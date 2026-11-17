"""Fixes for trial failures of TB-dev evals on E2B sandboxes that are not the policy's fault. Imported by
prl/rollout/harbor_agent.py; patches Harbor (``$PRL_EXT/pydeps_harbor``) in the rollout worker, nothing on disk.

1. tmux install (``TmuxSession._install_recording_tools``): 19 of the 70 TB-dev templates have no tmux, so every
   trial installs it with apt (2-12 s as root, checked on all 19). Harbor tries once; a transient mirror/lock failure
   (e.g. broken-python) then fails the trial before the agent starts. Retry the package-manager install. A slow
   mirror can also outlast Harbor's 120 s per install command: e2b then raises TimeoutException but does not kill the
   command (``CommandHandle``: "the command is not killed"), and the trial failed (in one eval all 30 trials of 4
   templates). Then wait while a package manager still runs in the sandbox, check tmux again and install again only
   if it is still missing, within a whole-step budget raised from Harbor's 240 s to TOOL_INSTALL_BUDGET_SEC (below
   Harbor's 360 s agent setup).
2. Dropped command streams (``E2BEnvironment.exec``): the sandbox service's gateway can close a command's event stream before its
   end event ("Command ended without an end event", 2-3 trials per eval). Harbor does not retry, rightly, since the
   command is running; reconnect to the same pid and wait for its end instead (never re-runs the command). A short
   command can end before the reconnect (process not found, its output gone: e.g. a ``tmux capture-pane``);
   the read-only tmux queries Terminus-2 polls with are then run again, anything else still raises.
3. Context accounting for summarization (``Terminus2._count_total_tokens``): litellm counts ``content`` with a
   generic tokenizer and ignores the retained ``reasoning_content``, so with thinking kept in the history the free
   context is overestimated and the next reply runs into the context end (SingleTurnMaxSeqLenExceededError in 21 of
   552 and 7 of 560 trials of two evals). Use the server's exact prompt + completion count of the last call while the
   history has only grown since. The proactive threshold (free tokens below which Terminus-2 summarizes; Harbor's
   default 8000) is HARBOR_TERMINUS_2_SUMMARY_THRESHOLD, forwarded by harbor_agent._terminus_kwargs.
4. An unawaited litellm coroutine (``sys.unraisablehook``): Miles makes every "coroutine ... was never awaited" fatal
   (``miles/utils/logging_utils.configure_strict_async_warnings``: ``os._exit(1)``). litellm can leave its own
   ``OpenAIChatCompletion.acompletion`` coroutine unawaited when a call is cancelled; that killed a run's rollout
   worker after 14 updates. Those are logged; every other unawaited coroutine stays fatal.
5. Stack dumps of the rollout worker: two runs stopped mid-rollout without a word (one for 3 h; the last lines
   "Unclosed client session") on a cluster whose compute nodes cannot be reached to look. Every STACK_DUMP_SEC
   a thread appends every thread's Python stack to ``$RUN_DIR/stack_dumps/<host>_<pid>.txt``, read under the GIL
   (``sys._current_frames``); a fatal signal also dumps there (``faulthandler.enable``). Not
   ``faulthandler.dump_traceback_later``: its watchdog reads the other threads without the GIL and killed two rollout
   workers in the middle of a dump.
Not patched, as in Harbor: a model that ends its shell (``exit`` / Ctrl-D) gets the RuntimeError of the next send-keys,
so the trial is not verified: evaluation counts it 0 (TB-dev's own scoring), training aborts and resamples it
(harbor_training.run: no eval_report).
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import re
import socket
import sys
import threading
import time
import traceback

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import ExecResult
from harbor.environments.e2b import E2BEnvironment
from harbor.llms.chat import Chat

logger = logging.getLogger(__name__)

TMUX_INSTALL_ATTEMPTS = 3
STREAM_RECONNECTS = 3
TOOL_INSTALL_BUDGET_SEC = 330  # Harbor's whole-step cap (240) leaves no room after one 120 s install times out
INSTALL_POLL_SEC = 10

try:
    from e2b.exceptions import TimeoutException as _InstallTimeout
except ImportError:  # e2b extra missing: nothing raises it
    _InstallTimeout = ()

# 1. tmux install --------------------------------------------------------------------------------------------------
_miles_install = TmuxSession._install_recording_tools
# A package manager running in the sandbox, read from /proc (pgrep is not in every image)
_INSTALLING = "grep -qsxE 'apt-get|apt|dpkg|yum|dnf|microdnf|apk' /proc/[0-9]*/comm"


async def _install_step(self: TmuxSession, install) -> None:
    try:
        await install
    except _InstallTimeout as exc:  # the command outlived its timeout and may still be installing
        logger.warning("tmux install command timed out; waiting for it and checking again: %s", exc)
    while (await self.environment.exec(command=_INSTALLING, user="root")).return_code == 0:
        await asyncio.sleep(INSTALL_POLL_SEC)  # Harbor's step budget (TOOL_INSTALL_BUDGET_SEC) ends a hung one


async def _install_recording_tools(self: TmuxSession) -> None:
    await _install_step(self, _miles_install(self))
    for attempt in range(1, TMUX_INSTALL_ATTEMPTS):
        if (await self.environment.exec(command="tmux -V", user="root")).return_code == 0:
            return
        system_info = await self._detect_system_info()
        command = self._get_combined_install_command(system_info, ["tmux"]) if system_info["package_manager"] else ""
        if not command:
            return
        logger.warning("tmux missing after install; retrying (%d/%d)", attempt, TMUX_INSTALL_ATTEMPTS - 1)
        await asyncio.sleep(5 * attempt)
        await _install_step(self, self.environment.exec(command=command, user="root",
                                                        timeout_sec=self._TOOL_INSTALL_TIMEOUT_SEC))


TmuxSession._install_recording_tools = _install_recording_tools
TmuxSession._TOOL_INSTALL_BUDGET_SEC = max(TmuxSession._TOOL_INSTALL_BUDGET_SEC, TOOL_INSTALL_BUDGET_SEC)

# 2. dropped command streams ---------------------------------------------------------------------------------------
try:
    from e2b.exceptions import NotFoundException
    from e2b.sandbox.commands.command_handle import CommandExitException
except ImportError:  # e2b extra missing: nothing to patch
    NotFoundException = CommandExitException = None


# Commands without side effects, safe to run again when their result was lost
_READ_ONLY = re.compile(r"tmux (capture-pane|has-session|display-message|-V)\b")


def _stream_dropped(exc: BaseException) -> bool:
    return "without an end event" in str(exc)


async def _exec(
    self: E2BEnvironment, command, cwd=None, env=None, timeout_sec=None, user=None, _reruns=STREAM_RECONNECTS
) -> ExecResult:
    from harbor.environments.e2b import effective_exec_cwd

    run_user = self._resolve_user(user)
    handle = await self._dispatch_command(
        command,
        cwd=effective_exec_cwd(cwd, self.task_env_config.workdir, self._workdir),
        env=self._merge_env(env),
        timeout_sec=timeout_sec,
        user=str(run_user) if run_user is not None else "root",
    )
    try:
        result = await handle.wait()
    except CommandExitException as e:
        result = e
    except Exception as e:
        if not _stream_dropped(e):
            raise
        try:
            return await _reconnect(self, handle, timeout_sec, e)
        except Exception as gone:
            if gone is not e or not _reruns or not _READ_ONLY.match(command):
                raise
            logger.warning("E2B command stream dropped and the process ended; re-running read-only %r", command[:80])
            return await _exec(self, command, cwd, env, timeout_sec, user, _reruns=_reruns - 1)
    return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.exit_code)


async def _reconnect(self: E2BEnvironment, handle, timeout_sec, error: BaseException) -> ExecResult:
    """Wait for the command's end over a new stream to the same pid; output before the drop is kept."""
    before_out, before_err = "".join(handle._stdout_chunks), "".join(handle._stderr_chunks)
    for attempt in range(1, STREAM_RECONNECTS + 1):
        logger.warning("E2B command stream dropped (pid %s); reconnecting (%d/%d)", handle.pid, attempt, STREAM_RECONNECTS)
        try:
            again = await self._sandbox.commands.connect(handle.pid, timeout=timeout_sec or 0)
            result = await again.wait()
        except CommandExitException as e:
            result = e
        except NotFoundException:
            raise error  # the process already ended: its exit code is gone
        except Exception as e:
            if not _stream_dropped(e) or attempt == STREAM_RECONNECTS:
                raise
            await asyncio.sleep(attempt)
            continue
        return ExecResult(
            stdout=before_out + (result.stdout or ""), stderr=before_err + (result.stderr or ""), return_code=result.exit_code
        )
    raise error


if CommandExitException is not None:
    E2BEnvironment.exec = _exec

# 3. context accounting --------------------------------------------------------------------------------------------
_miles_chat = Chat.chat


async def _chat(self: Chat, prompt, logging_path=None, **kwargs):
    response = await _miles_chat(self, prompt, logging_path=logging_path, **kwargs)
    usage = response.usage
    if usage is not None:
        # (history length after this call, exact tokens of that history as the server saw it)
        self._tw_last_seq = (len(self._messages), usage.prompt_tokens + usage.completion_tokens)
    return response


Chat.chat = _chat
_miles_count_total_tokens = Terminus2._count_total_tokens


def _count_total_tokens(self: Terminus2, chat: Chat) -> int:
    estimate = _miles_count_total_tokens(self, chat)
    n_messages, exact = getattr(chat, "_tw_last_seq", (None, 0))
    # A summary resets the history (chat._messages = messages[:1]): the last call's count no longer applies
    if n_messages is not None and len(chat.messages) >= n_messages:
        return max(estimate, exact)
    return estimate


Terminus2._count_total_tokens = _count_total_tokens


# 4. An unawaited litellm coroutine --------------------------------------------------------------------------------
LITELLM_UNAWAITED = re.compile(r"coroutine '(OpenAIChatCompletion\.acompletion|[\w.]*litellm[\w.]*)' was never awaited")


def _tolerate_litellm(hook):
    if getattr(hook, "_prl_litellm", False):
        return hook

    def tolerant(unraisable):
        exc = unraisable.exc_value
        if isinstance(exc, RuntimeWarning) and LITELLM_UNAWAITED.search(str(exc)):
            logger.warning("[prl] ignored an unawaited litellm coroutine (harbor_fixes 4): %s", exc)
            return
        hook(unraisable)

    tolerant._prl_litellm = True
    return tolerant


sys.unraisablehook = _tolerate_litellm(sys.unraisablehook)  # Miles' hook already installed: ours goes in front
try:  # ...and if Miles installs it later, it wraps ours: put ours in front again
    from miles.utils import logging_utils as _miles_logging

    _miles_strict = _miles_logging.configure_strict_async_warnings

    def _strict_then_tolerant():
        _miles_strict()
        sys.unraisablehook = _tolerate_litellm(sys.unraisablehook)

    _miles_logging.configure_strict_async_warnings = _strict_then_tolerant
except ImportError:  # outside the Miles container
    pass


# 5. Stack dumps of the rollout worker -----------------------------------------------------------------------------
STACK_DUMP_SEC = 900
_stack_dump_file = None


def start_stack_dumps(run_dir: str | None = os.environ.get("RUN_DIR"), every: float = STACK_DUMP_SEC) -> str | None:
    """Append all threads' stacks to <run_dir>/stack_dumps/<host>_<pid>.txt every ``every`` seconds; the path."""
    global _stack_dump_file
    if not run_dir or _stack_dump_file is not None:
        return None
    os.makedirs(os.path.join(run_dir, "stack_dumps"), exist_ok=True)
    path = os.path.join(run_dir, "stack_dumps", f"{socket.gethostname()}_{os.getpid()}.txt")
    _stack_dump_file = open(path, "a", buffering=1)  # noqa: SIM115 -- open for the life of the process
    faulthandler.enable(file=_stack_dump_file, all_threads=True)  # only on a fatal signal

    def dump_forever() -> None:
        while True:
            time.sleep(every)
            names = {t.ident: t.name for t in threading.enumerate()}
            lines = [f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}"]
            for ident, frame in sys._current_frames().items():  # holding the GIL: frames cannot change under us
                lines.append(f"--- thread {ident} {names.get(ident, '')}")
                lines += [line.rstrip() for line in traceback.format_stack(frame)]
            _stack_dump_file.write("\n".join(lines) + "\n")

    threading.Thread(target=dump_forever, name="stack-dumps", daemon=True).start()
    return path


start_stack_dumps()
