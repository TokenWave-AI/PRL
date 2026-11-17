"""Keep one evaluation result per agent trial when TITO v2 branches history.

Loaded by Miles ``--session-sample-picker-path`` inside SessionCoreV2, before
its default postprocessor attaches the agent's verifier reward. Terminus-2
summarization creates multiple history branches; counting every leaf would
weight trials by their number of summaries. Select the latest available leaf
by commit sequence, preserving the input object and its token metadata.

This is an evaluation-only reduction, not a training trajectory policy: earlier
branches are omitted from the returned Sample, while Harbor keeps the full
trajectory on disk. Token/length metrics describe the selected branch only.
An empty input stays empty for Miles' existing aborted-sample handling.

Configuration: requires ``MODE=rollout``, already forwarded by scripts/run.sh.
No new environment variables. Tree metadata follows upstream
``miles/rollout/session/v2/utils.py:build_leaf_material``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miles.utils.types import Sample


def latest_eval_leaf(leaf_samples: list[Sample], session_metadata: dict[str, Any]) -> list[Sample]:
    """Select one representative leaf without duplicating a trial's reward."""
    if os.environ.get("MODE") != "rollout":
        raise ValueError("latest_eval_leaf requires MODE=rollout; it discards branches needed for training")
    if not leaf_samples:
        return []
    return [max(leaf_samples, key=lambda sample: sample.metadata["leaf"]["node_id"])]
