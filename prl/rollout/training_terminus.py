"""Terminus-2 subclass that identifies auxiliary generations for training.

Selected through Harbor AgentConfig.import_path by harbor_agent._trial_config
only for the training adapter. The OpenAI ``user`` field labels calls in TITO
v2's node.turn_args; it is not part of the model's messages/chat template.
Send it through extra_body: LiteLLM drops top-level user for non-OpenAI models
when Harbor sets drop_params=True, before the request reaches TITO.
Action generations (including thinking) are trainable. Summary, questions,
answers and fallback summary generations are context only.

Configuration: constructor ``force_summary_once=False`` is a smoke-only option
passed through HARBOR_ENV_KWARGS.prl_force_summary_once (true, or the
fraction of trials that get it; harbor_agent._forced picks them). It requests one
real summary after the first successful action if the task continues. Normal
proactive summarization remains unchanged; no process-global mutation is used.
"""

from __future__ import annotations

from typing import Any

from harbor.agents.terminus_2 import Terminus2

ACTION_TAG = "prl/action-v1"
SUMMARY_TAG = "prl/summary-v1"


class TrainingTerminus2(Terminus2):
    def __init__(self, *args: Any, force_summary_once: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._llm_call_kwargs = self._tagged_kwargs(ACTION_TAG)
        self._prl_force_summary = force_summary_once
        self._prl_action_calls = 0

    async def _run_subagent(self, *args: Any, **kwargs: Any) -> Any:
        previous = self._llm_call_kwargs
        self._llm_call_kwargs = self._tagged_kwargs(SUMMARY_TAG)
        try:
            return await super()._run_subagent(*args, **kwargs)
        finally:
            self._llm_call_kwargs = previous

    def _tagged_kwargs(self, tag: str) -> dict[str, Any]:
        return {**self._llm_call_kwargs,
                "extra_body": {**self._llm_call_kwargs.get("extra_body", {}), "user": tag}}

    async def _chat_with_retries(self, *args: Any, **kwargs: Any) -> Any:
        response = await super()._chat_with_retries(*args, **kwargs)
        self._prl_action_calls += 1
        return response

    async def _check_proactive_summarization(self, chat: Any, original_instruction: str, session: Any) -> Any:
        if self._prl_force_summary and self._prl_action_calls:
            self._prl_force_summary = False
            self.logger.info("[prl] smoke: forcing one summary after a successful action")
            return await self._summarize_with_fallback(chat, original_instruction, session)
        return await super()._check_proactive_summarization(chat, original_instruction, session)
