"""Terminus-2 without a model: runs recorded keystrokes turn by turn and saves what the terminal shows.

Selected through Harbor AgentConfig.import_path by prl/rollout/replay.py. Same session, keystroke timing and output limit
as a generated trial, so a rewritten trial is scored as a generated one is: by executing it and running the task's verifier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_2 import Command


class ReplayAgent(Terminus2):
    def __init__(self, *args: Any, replay_file: str, force_summary_once: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._replay_file = Path(replay_file)

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        request = json.loads(self._replay_file.read_text())
        await self._session.get_incremental_output()
        outputs = []
        for commands in request["turns"]:
            batch = [Command(keystrokes=c["keystrokes"], duration_sec=min(float(c.get("duration", 1.0)), 60))
                     for c in commands]
            timed_out, output = await self._execute_commands(batch, self._session)
            outputs.append({"timed_out": timed_out, "output": output})
            # Each turn: an agent timeout still leaves the verifier a reward, and these
            Path(request["outputs"]).write_text(json.dumps(outputs, ensure_ascii=False))
