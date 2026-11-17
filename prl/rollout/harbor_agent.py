"""Miles agent function for Harbor's Terminus-2, set up like OpenThinker-Agent's RL (SkyRL ``dcagent`` branch).

Wired in by scripts/lib/container_main.sh when ``HARNESS=harbor``:

    --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
    --custom-agent-function-path    prl.rollout.harbor_agent.run
    --custom-rm-path                generate.reward_func   (examples/swe-agent-harbor-docker)

Everything else is Miles' in-process Harbor path (``examples/experimental/harbor/harbor_agent_function.py``):
the trial runs in the rollout worker, the sandbox is reached over the E2B API (``HARBOR_ENV_TYPE=e2b`` pointed at
the site's ``E2B_API_URL``), and the verifier's ``reward`` is the reward. Three Terminus-2 kwargs
differ from Miles' binding:

- ``parser_name``: Terminus-2's default ``json`` (what OpenThinker trained and evaluated with); Miles hard-codes
  ``xml``. ``HARBOR_TERMINUS_2_PARSER`` overrides.
- ``record_terminal_session=False``: otherwise every trial installs asciinema in the sandbox (OpenThinker's
  configs turn it off too).
- no ``max_tokens`` per call: Miles forwards ``--rollout-max-response-len`` (30720) as every call's ``max_tokens``, and
  from the second turn on prompt + 30720 exceeds the 32768 context, which SGLang rejects (400). SkyRL sends none, so
  each reply may use whatever context remains; the trajectory cap is ``--max-seq-len``.
Configuration additions: HARBOR_ENV_KWARGS may contain prl_template_manifest
(a ready-template manifest path) and prl_max_concurrent_trials (per-worker
concurrency). They are consumed before constructing the E2B environment. Cloud
AgentError results raise instead of being silently counted as unsuccessful answers.
The opt-in harbor_training adapter selects TrainingTerminus2 through per-request
metadata; prl_force_summary_once in HARBOR_ENV_KWARGS enables its smoke probe (true, or a fraction of trials).
A sample with ``prl_reroll_file`` in its metadata (prl/rollout/branching.py) gets RerollTerminus2
(prl/rollout/reroll_agent.py) instead.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
from typing import Any

import harbor_agent_function as _haf  # examples/experimental/harbor on PYTHONPATH

import prl.rollout.harbor_fixes  # noqa: F401  # Harbor fixes for sandbox trial failures (see the module)

_miles_terminus_kwargs = _haf.HARNESS_BINDINGS["terminus-2"].kwargs


def _terminus_kwargs(session_url, api_key, sampling_params, model, max_seq_len):
    kwargs = _miles_terminus_kwargs(session_url, api_key, sampling_params, model, max_seq_len)
    kwargs["parser_name"] = os.getenv("HARBOR_TERMINUS_2_PARSER", "json")
    kwargs["record_terminal_session"] = False
    kwargs["llm_call_kwargs"].pop("max_tokens", None)
    # Free tokens below which Terminus-2 summarizes (Harbor default 8000); see harbor_fixes.py (3)
    if threshold := os.getenv("HARBOR_TERMINUS_2_SUMMARY_THRESHOLD"):
        kwargs["proactive_summarization_threshold"] = int(threshold)
    return kwargs


for _name in ("terminus-2", "terminus"):
    _haf.HARNESS_BINDINGS[_name] = dataclasses.replace(_haf.HARNESS_BINDINGS[_name], kwargs=_terminus_kwargs)

# Carry custom options inside Miles' already-forwarded HARBOR_ENV_KWARGS.
# Resolve per trial: a process-global template ID would cross-contaminate tasks.
_miles_trial_config = _haf.build_trial_config


def _trial_config(metadata: dict[str, Any], session_url: str, request_kwargs: dict[str, Any]) -> Any:
    from prl.sandbox.template_manifest import resolve_template

    config = _miles_trial_config(metadata, session_url, request_kwargs)
    options = config.environment.kwargs
    manifest = options.pop("prl_template_manifest", None)
    options.pop("prl_max_concurrent_trials", None)
    force_summary = options.pop("prl_force_summary_once", False)
    if metadata.get("prl_training_contract"):
        config.agent.name = None
        config.agent.import_path = "prl.rollout.training_terminus:TrainingTerminus2"
        config.agent.kwargs["force_summary_once"] = _forced(force_summary, session_url) and not metadata["prl_evaluation"]
        if metadata.get("prl_reroll_file"):  # a branch (rollout/branching.py): replay, then the policy
            config.agent.import_path = "prl.rollout.reroll_agent:RerollTerminus2"
            config.agent.kwargs["reroll_file"] = metadata["prl_reroll_file"]
    if manifest:
        options["prebuilt_template_id"] = resolve_template(manifest, config.task.path, os.environ["E2B_API_URL"])
    return config


def _forced(setting: Any, key: str) -> bool:
    """prl_force_summary_once: true, false, or the fraction of trials to split (picked by a hash of the trial's
    session). A transform smoke splits half: rewriting needs a trial in one piece, summary masks need split ones."""
    if isinstance(setting, bool) or setting in (0, 1):
        return bool(setting)
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < float(setting)


_haf.build_trial_config = _trial_config
_miles_trial_metadata = _haf.trial_result_to_metadata


def _trial_metadata(result: Any) -> dict[str, Any]:
    options = json.loads(os.getenv("HARBOR_ENV_KWARGS", "{}") or "{}")
    exc = getattr(result, "exception_info", None)
    policy_limits = {"AgentTimeoutError", "MaxSeqLenExceededError", "SingleTurnMaxSeqLenExceededError"}
    if options.get("prl_template_manifest") and exc and exc.exception_type not in policy_limits:
        # the type and the start of the message: harbor_training tells a failure of the sandbox service by them
        message = " ".join(str(getattr(exc, "exception_message", "") or "").split())[:200]
        raise RuntimeError(f"Harbor {exc.exception_type}: {message}; inspect trial {result.trial_uri}")
    return _miles_trial_metadata(result)


_haf.trial_result_to_metadata = _trial_metadata
_trial_slots: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


async def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
    options = json.loads(os.getenv("HARBOR_ENV_KWARGS", "{}") or "{}")
    cap = int(options.get("prl_max_concurrent_trials", 0))

    async def invoke() -> dict[str, Any]:
        result = await _haf.run(*args, **kwargs)
        if options.get("prl_template_manifest") and result["exit_status"] == "AgentError":
            raise RuntimeError(f"Harbor trial failed; inspect infrastructure/agent logs: {result.get('trial_dir')}")
        return result

    if cap <= 0:
        return await invoke()
    slots = _trial_slots.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(cap))
    async with slots:
        return await invoke()
