"""Check summary call tags are instance-local and restored on exceptions."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("harbor")  # the Harbor agent (installed on the cluster, see docs/setup.md)
from prl.rollout.training_terminus import ACTION_TAG, SUMMARY_TAG, Terminus2, TrainingTerminus2


def test_summary_tag_is_request_metadata_and_restored(monkeypatch):
    seen = []
    async def subagent(self, **kwargs):
        seen.append(self._llm_call_kwargs.copy())
        raise RuntimeError("transport failure")
    monkeypatch.setattr(Terminus2, "_run_subagent", subagent)
    agent = object.__new__(TrainingTerminus2)
    original = {"temperature": 1.0, "extra_body": {"user": ACTION_TAG, "return_token_ids": True}}
    agent._llm_call_kwargs = original
    with pytest.raises(RuntimeError):
        asyncio.run(agent._run_subagent())
    assert seen == [{"temperature": 1.0, "extra_body": {"user": SUMMARY_TAG, "return_token_ids": True}}]
    assert agent._llm_call_kwargs is original
    assert original["extra_body"]["user"] == ACTION_TAG


def test_harbor_forwards_tag_outside_messages(monkeypatch):
    from harbor.llms.lite_llm import LiteLLM
    from litellm import ModelResponse

    seen = []
    async def completion(**kwargs):
        seen.append(kwargs)
        return ModelResponse(model="model", choices=[{"message": {"role": "assistant", "content": "summary"},
                                                     "finish_reason": "stop", "index": 0}],
                             usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
    monkeypatch.setattr("harbor.llms.lite_llm.litellm.acompletion", completion)
    llm = LiteLLM(model_name="openai/model", api_base="http://localhost/v1", api_key="dummy",
                  model_info={"max_input_tokens": 65536, "max_output_tokens": 63488,
                              "input_cost_per_token": 0, "output_cost_per_token": 0})
    asyncio.run(llm.call(prompt="Summarize", message_history=[], extra_body={"user": SUMMARY_TAG}))
    assert seen[0]["extra_body"]["user"] == SUMMARY_TAG
    assert seen[0]["messages"] == [{"role": "user", "content": "Summarize"}]
