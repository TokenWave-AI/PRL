"""The three Harbor fixes of prl/rollout/harbor_fixes.py, against fakes (no sandbox, no model)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")
pytest.importorskip("e2b")

from e2b.exceptions import NotFoundException
from e2b.sandbox.commands.command_handle import CommandExitException
from harbor.environments.base import ExecResult

from prl.rollout import harbor_fixes as fx


def _run(coro):
    return asyncio.run(coro)


# 1. tmux install retry
class _InstallEnv:
    def __init__(self, fail_installs: int, running_polls: int = 0):
        self.fail_installs, self.installed, self.commands = fail_installs, False, []
        self.running_polls = running_polls  # polls during which a timed-out install still runs, then it has finished

    async def exec(self, command, user=None, timeout_sec=None):
        self.commands.append(command)
        if command == fx._INSTALLING:
            if self.running_polls:
                self.running_polls -= 1
                self.installed = self.running_polls == 0
                return ExecResult(stdout="", stderr="", return_code=0)
            return ExecResult(stdout="", stderr="", return_code=1)
        if command == "tmux -V":
            return ExecResult(stdout="tmux 3.3a" if self.installed else "", stderr="", return_code=0 if self.installed else 127)
        if "apt-get install" in command:
            if self.fail_installs:
                self.fail_installs -= 1
                return ExecResult(stdout="", stderr="E: Could not get lock", return_code=100)
            self.installed = True
            return ExecResult(stdout="", stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=0)


def _tmux_session(env):
    async def detect():
        return {"package_manager": "apt-get"}

    session = SimpleNamespace(environment=env, _detect_system_info=detect, _TOOL_INSTALL_TIMEOUT_SEC=120)
    session._get_combined_install_command = lambda info, tools: f"apt-get update && apt-get install -y {' '.join(tools)}"
    return session


def test_tmux_install_retried_after_a_transient_failure(monkeypatch):
    env = _InstallEnv(fail_installs=1)

    async def harbor_install(self):  # Harbor's single attempt
        await self.environment.exec(command="apt-get update && apt-get install -y tmux", user="root")

    monkeypatch.setattr(fx, "_miles_install", harbor_install)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(fx.asyncio, "sleep", lambda s: real_sleep(0))
    _run(fx._install_recording_tools(_tmux_session(env)))
    assert env.installed
    assert sum("apt-get install" in c for c in env.commands) == 2


def test_tmux_already_present_is_not_reinstalled(monkeypatch):
    env = _InstallEnv(fail_installs=0)
    env.installed = True

    async def harbor_install(self):
        pass

    monkeypatch.setattr(fx, "_miles_install", harbor_install)
    _run(fx._install_recording_tools(_tmux_session(env)))
    assert not any("apt-get install" in c for c in env.commands)


@pytest.mark.parametrize("finishes", [True, False])
def test_a_timed_out_tmux_install_is_waited_for_then_retried_if_needed(monkeypatch, finishes):
    """In an eval, apt outlived Harbor's 120 s; e2b raised TimeoutException but the install kept running in the
    sandbox. Wait for it; install again only if tmux is still missing."""
    from e2b.exceptions import TimeoutException

    env = _InstallEnv(fail_installs=0, running_polls=2)

    async def harbor_install(self):
        await self.environment.exec(command="apt-get update", user="root")  # not the install the fake env completes
        if not finishes:
            env.running_polls = 0  # the timed-out install died: nothing runs, tmux missing
        raise TimeoutException("Request timed out")

    monkeypatch.setattr(fx, "_miles_install", harbor_install)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(fx.asyncio, "sleep", lambda s: real_sleep(0))
    _run(fx._install_recording_tools(_tmux_session(env)))
    assert env.installed
    assert sum("apt-get install" in c for c in env.commands) == (0 if finishes else 1)


def test_the_tool_install_budget_leaves_room_for_a_second_install():
    from harbor.agents.terminus_2.tmux_session import TmuxSession

    assert TmuxSession._TOOL_INSTALL_BUDGET_SEC >= 2 * TmuxSession._TOOL_INSTALL_TIMEOUT_SEC + 30
    assert TmuxSession._TOOL_INSTALL_BUDGET_SEC < 360  # Harbor's agent setup timeout (trial.py)


# 2. dropped command stream -> reconnect to the same pid
class _Handle:
    def __init__(self, outcome, out="", pid=42):
        self.outcome, self.pid, self._stdout_chunks, self._stderr_chunks = outcome, pid, [out], []

    async def wait(self):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _Commands:
    def __init__(self, reconnects):
        self.reconnects, self.connected = list(reconnects), []

    async def connect(self, pid, timeout=None):
        self.connected.append(pid)
        nxt = self.reconnects.pop(0)
        if isinstance(nxt, type) and issubclass(nxt, BaseException):
            raise nxt("gone")
        return nxt


def _environment(first, reconnects):
    dispatched = []

    async def dispatch(command, **kw):
        dispatched.append(command)
        return first

    env = SimpleNamespace(
        _resolve_user=lambda u: u, _merge_env=lambda e: e, _dispatch_command=dispatch,
        task_env_config=SimpleNamespace(workdir=None), _workdir="/app",
        _sandbox=SimpleNamespace(commands=_Commands(reconnects)),
    )
    return env, dispatched


DROP = Exception("Command ended without an end event")


def test_dropped_stream_waits_on_the_same_process():
    done = SimpleNamespace(stdout=" world", stderr="", exit_code=0)
    env, dispatched = _environment(_Handle(DROP, out="hello"), [_Handle(done)])
    result = _run(fx._exec(env, "long command"))
    assert (result.stdout, result.return_code) == ("hello world", 0)
    assert dispatched == ["long command"]  # never re-run
    assert env._sandbox.commands.connected == [42]


def test_dropped_stream_keeps_a_nonzero_exit():
    failed = CommandExitException(stdout="", stderr="boom", exit_code=3, error=None)
    env, _ = _environment(_Handle(DROP), [_Handle(failed)])
    result = _run(fx._exec(env, "cmd"))
    assert (result.stderr, result.return_code) == ("boom", 3)


def test_process_gone_raises_the_original_error():
    env, _ = _environment(_Handle(DROP), [NotFoundException])
    with pytest.raises(Exception, match="without an end event"):
        _run(fx._exec(env, "cmd"))


def test_other_errors_are_not_retried():
    env, _ = _environment(_Handle(TimeoutError("command timed out")), [])
    with pytest.raises(TimeoutError):
        _run(fx._exec(env, "cmd"))
    assert env._sandbox.commands.connected == []


# 3. context accounting uses the server's count while the history only grew
def test_count_uses_exact_server_count(monkeypatch):
    monkeypatch.setattr(fx, "_miles_count_total_tokens", lambda self, chat: 20_000)  # content-only estimate
    chat = SimpleNamespace(messages=[0] * 5, _tw_last_seq=(5, 50_000))
    assert fx._count_total_tokens(None, chat) == 50_000
    chat.messages = [0] * 7  # grown since: the exact count is still a lower bound
    assert fx._count_total_tokens(None, chat) == 50_000


def test_count_after_summary_reset_falls_back(monkeypatch):
    monkeypatch.setattr(fx, "_miles_count_total_tokens", lambda self, chat: 3_000)
    chat = SimpleNamespace(messages=[0], _tw_last_seq=(9, 60_000))
    assert fx._count_total_tokens(None, chat) == 3_000
    assert fx._count_total_tokens(None, SimpleNamespace(messages=[0, 1])) == 3_000


def test_read_only_tmux_query_is_rerun_when_the_process_is_gone():
    env, dispatched = _environment(_Handle(DROP), [NotFoundException])
    redo = SimpleNamespace(stdout="pane", stderr="", exit_code=0)
    first = env._dispatch_command

    async def dispatch(command, **kw):
        handle = await first(command, **kw)
        return handle if len(dispatched) == 1 else _Handle(redo)

    env._dispatch_command = dispatch
    result = _run(fx._exec(env, "tmux capture-pane -p -t terminus-2"))
    assert (result.stdout, len(dispatched)) == ("pane", 2)
