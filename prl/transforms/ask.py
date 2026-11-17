"""ask: a transform's call to the run's model (the judge / teacher profile, prl/algo/llm_client.py). Every call
is saved with the trial by the framework (prl/transforms/core.py collects them through _calls).
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from typing import Any


class Unanswered(Exception):
    """``ask`` got no acceptable answer: the trial is left unchanged (status skipped)."""


_calls: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar("transform_calls",
                                                                                      default=None)


async def ask(prompt: str | list[dict[str, str]], check: Callable[[str], Any] | None = None,
              prefix: str = "PRL_JUDGE_") -> Any:
    """The run's model (the judge profile, e.g. configs/arms/ota_judge.env) on ``prompt``; returns
    ``check(answer)``, or the answer. ``check`` raises to reject an answer, and the model is asked again with the
    reason (the profile's attempts). Retries, rate limits, refusals and concurrency are prl/algo/llm_client.py's; every call is saved
    with the trial (``<trial_dir>/transform_calls.json``). No acceptable answer: raises ``Unanswered``, which leaves
    the trial unchanged (skipped) unless the transform catches it."""
    from prl.algo import llm_client

    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    record: dict[str, Any] = {}
    # Boxed, so that a check returning None still counts as an answer
    result = await llm_client.ask(messages, lambda content: ((check or str)(content),), prefix=prefix, record=record,
                                  feedback=True)
    calls = _calls.get()
    if calls is not None:
        calls.append({"messages": messages, **record})
    if result is None:
        raise Unanswered(f"no acceptable answer from the model: {(record.get('errors') or ['?'])[-1]}"[:1000])
    return result[0]
