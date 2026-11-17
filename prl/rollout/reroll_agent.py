"""Terminus-2 that replays a trial's first turns, then lets the policy continue: the agent of a reroll.

Selected through Harbor AgentConfig.import_path by prl/rollout/harbor_agent.py when the sample's metadata carries
``prl_reroll_file`` (written by prl/rollout/harbor_training.py for a ``step.reroll()`` of the trajectory
transform, prl/transforms/core.py). The file holds the request ``Trajectory._reroll_request`` built: the trial's
first user message and, for each turn before the rerolled one, its thinking and reply (possibly rewritten by the
teacher) and the message that followed it as generated.

How: Terminus-2's own loop runs, but its first ``len(turns)`` model calls return the recorded replies instead of
calling the model. Terminus-2 parses them, runs their commands in this trial's fresh sandbox and writes the next message
as for any reply (terminal output, warnings, parse errors, the completion check), so the context the policy then sees is
the one Terminus-2 would have built, with real terminal output. From the rerolled turn on, calls go to the policy
through the trial's own TITO session; its first request carries the whole history, which the session server renders
with the model's chat template (Qwen3.5/3.6: ``preserve_thinking``, so the replayed turns keep their thinking, as the
policy saw them). Those turns are the sample's prompt, so only what the policy generates from the rerolled turn on is
trained.

The first message is the recorded one (a fresh sandbox's first screen may differ, e.g. its host name). No summary during
the replayed turns: the trial being rerolled had none (core refuses trials split by a summary). Each replayed turn's
executed message is compared with the recorded one; the counts go to ``reroll_replay.json`` in the trial directory
(``same`` / ``differ``, and ``differ_detail``: the end of both messages as compared), and harbor_training copies them
into the branch's outcome. The comparison leaves out the shell prompt's host name and blank lines: every fresh sandbox
has its own host name and tmux pads the screen with blank lines, which alone made 186 of 188 replayed turns differ in a
smoke run. What still differs is the environment itself (file times, inodes, commit hashes) or a command that really
behaved otherwise. Until the policy's first call, the context count Terminus-2 uses to decide on a summary
(prl/rollout/harbor_fixes.py (3)) is the trial's own (``context_tokens``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from harbor.llms.base import LLMResponse

from prl.rollout.training_terminus import TrainingTerminus2

DETAIL_CHARS = 1500  # per message in reroll_replay.json differ_detail (the end of a screen holds the latest output)
_PROMPT_HOST = re.compile(r"\b([\w.-]+)@[\w.-]+:(?=\S*[#$])")  # root@c-6af1dda6-...:/workspace#


def _normal(text: str) -> str:
    """A terminal message without what differs from sandbox to sandbox by itself: the shell prompt's host name and
    blank lines."""
    text = _PROMPT_HOST.sub(r"\1@HOST:", text)
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())


class RerollTerminus2(TrainingTerminus2):
    def __init__(self, *args: Any, reroll_file: str, **kwargs: Any) -> None:
        kwargs["force_summary_once"] = False
        super().__init__(*args, **kwargs)
        self._reroll_file = Path(reroll_file)
        request = json.loads(self._reroll_file.read_text())
        self._instruction = request["instruction"]
        self._turns = list(request["turns"])
        self._context_tokens = int(request.get("context_tokens") or 0)
        self._replayed = 0
        self._compared = {"turns": len(self._turns), "same": 0, "differ": [], "differ_detail": [],
                          "first_message_same": None}

    async def _chat_with_retries(self, chat: Any, prompt: str) -> Any:
        if self._replayed == 0:
            self._compared["first_message_same"] = _normal(prompt) == _normal(self._instruction)
            prompt = self._instruction  # the policy's first message, as in the trial being rerolled
        elif self._replayed <= len(self._turns):
            self._compare(prompt)
        if self._replayed >= len(self._turns):
            self._replayed += 1
            self._save()
            return await super()._chat_with_retries(chat, prompt)
        turn = self._turns[self._replayed]
        self._replayed += 1
        # As Chat.chat records a call (interleaved_thinking: the reasoning travels with the reply)
        reply: dict[str, Any] = {"role": "assistant", "content": turn["content"]}
        if turn["reasoning"]:
            reply["reasoning_content"] = turn["reasoning"]
        chat._messages.extend([{"role": "user", "content": prompt}, reply])
        if self._replayed == len(self._turns) and self._context_tokens:
            # harbor_fixes (3): Terminus-2 decides on a summary from the last call's exact token count, which a replayed
            # reply does not have; the trial's own count through this reply stands in until the policy's first call
            chat._tw_last_seq = (len(chat._messages), self._context_tokens)
        return LLMResponse(content=turn["content"], reasoning_content=turn["reasoning"] or None)

    def _compare(self, message: str) -> None:
        """The message Terminus-2 built after replayed turn ``self._replayed - 1``, against the recorded one."""
        recorded = self._turns[self._replayed - 1].get("message")
        if recorded is not None and _normal(message) == _normal(recorded):
            self._compared["same"] += 1
        else:
            step = self._turns[self._replayed - 1]["step"]
            self._compared["differ"].append(step)
            # What differed, normalized as compared (the step numbers alone did not tell environment noise from a
            # command that behaved otherwise)
            self._compared["differ_detail"].append({"step": step, "recorded": _normal(recorded or "")[-DETAIL_CHARS:],
                                                    "executed": _normal(message)[-DETAIL_CHARS:]})

    def _save(self) -> None:
        if self._replayed == len(self._turns) + 1:  # once, when the policy takes over
            (self._reroll_file.parent / "reroll_replay.json").write_text(json.dumps(self._compared) + "\n")

    async def _check_proactive_summarization(self, chat: Any, original_instruction: str, session: Any) -> Any:
        if self._replayed < len(self._turns):
            return None
        return await super()._check_proactive_summarization(chat, original_instruction, session)
