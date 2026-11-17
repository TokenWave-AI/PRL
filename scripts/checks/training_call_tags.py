"""Probe training tags through real Harbor/LiteLLM/OpenAI HTTP serialization.

Called by grpo_preflight on the compute node, and directly on any host without
GPU/model/sandbox access. Only the HTTP transport is fake; main and auxiliary calls
use the real Terminus methods. Temporary subagent trajectories are removed.
Returns wire tags/messages for the preflight receipt. No environment settings.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
from harbor.llms.chat import Chat
from openai import AsyncOpenAI

from prl.rollout.training_terminus import ACTION_TAG, SUMMARY_TAG, TrainingTerminus2


async def check_wire_tags() -> dict[str, Any]:
    requests: list[dict[str, Any]] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, request=request, json={
            "id": f"chatcmpl-probe-{len(requests)}", "object": "chat.completion", "created": 1,
            "model": "Qwen3.5-9B", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "probe response"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        })

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http,
        AsyncOpenAI(api_key="probe-only", base_url="http://wire-probe.invalid/v1",
                    http_client=http, max_retries=0) as client,
    ):
        with TemporaryDirectory(prefix="call-tags-") as directory:
            agent = TrainingTerminus2(
                logs_dir=Path(directory), model_name="openai/Qwen3.5-9B",
                api_base="http://wire-probe.invalid/v1", record_terminal_session=False,
                model_info={"max_input_tokens": 65536, "max_output_tokens": 63488,
                            "input_cost_per_token": 0, "output_cost_per_token": 0},
                llm_kwargs={"api_key": "probe-only"},
                llm_call_kwargs={"client": client, "extra_body": {"return_token_ids": True}},
            )
            # Regression control: the old top-level tag is silently dropped.
            await agent._llm.call(prompt="control", message_history=[], client=client, user=ACTION_TAG)
            await agent._chat_with_retries(Chat(agent._llm), "action before summary")
            await agent._run_subagent(
                prompt="summarize", message_history=[], steps=[], session_id="probe-summary",
                agent_name="terminus-2-summarization-summary", filename_suffix="summary",
                summary_text="wire probe",
            )
            await agent._chat_with_retries(Chat(agent._llm), "action after summary")

    tags = [request.get("user") for request in requests]
    assert tags == [None, ACTION_TAG, SUMMARY_TAG, ACTION_TAG], tags
    for request, prompt in zip(requests, ("control", "action before summary", "summarize", "action after summary"),
                               strict=True):
        assert request["messages"] == [{"role": "user", "content": prompt}], request["messages"]
        assert "extra_body" not in request  # OpenAI serializes it into top-level JSON.
    assert all(request["return_token_ids"] for request in requests[1:])
    return {"passed": True, "tags": tags, "requests": requests}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(check_wire_tags()), indent=2))
