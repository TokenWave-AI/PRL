"""A trial that failed on the sandbox service's rate limit holds its slot before it is replaced."""

import asyncio
import sys
import types

import pytest

from prl.rollout import harbor_training


@pytest.mark.parametrize("error, waits", [("Harbor RateLimitException; inspect trial x", True), ("Harbor Other; x", False)])
def test_only_rate_limit_failures_wait(monkeypatch, error, waits):
    async def failing_run(*args, **kwargs):
        raise RuntimeError(error)

    monkeypatch.setitem(sys.modules, "prl.rollout.harbor_agent", types.SimpleNamespace(run=failing_run))
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(harbor_training.asyncio, "sleep", sleep)
    monkeypatch.setenv("PRL_RATE_LIMIT_WAIT", "60")
    result = asyncio.run(harbor_training.run(metadata={"prl_evaluation": False}))
    assert result["prl_aborted"] is True and result["reward"] is None
    assert bool(slept) is waits
    assert all(30 <= s <= 90 for s in slept)
