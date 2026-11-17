"""Collect compact Harbor trajectories without training auxiliary summary calls.

Miles SessionCoreV2 loads ``pick_samples`` via --session-sample-picker-path and
``postprocess_samples`` via --session-sample-postprocessor-path. The generate
adapter passes per-request evaluation state in agent metadata, so periodic eval
works even when MODE=train. Eval counts one leaf/trial; training retains distinct
history branches and reuses Miles' default shared-completion ownership masks.
Only auxiliary completion spans are additionally zeroed; the trajectory transform
(PRL_TRANSFORM) runs later, in the async generate adapter (harbor_training.py), since this runs
on the session server's event loop. Observations/context already have zero loss in Miles. All retained leaves share their trial's reward and rollout_id;
GRPO normalization counts rollouts, not leaves.

Inputs follow miles/rollout/session/v2/utils.py. Unknown/missing call tags fail
closed; missing agent metadata produces an aborted sample upstream. When the
adapter sets agent.prl_reject_mismatch, annotate every selected leaf with
the checks from ALL raw leaves, before deduplication/masking. The generate adapter
rejects and resamples the whole trial. No direct environment reads here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def pick_samples(leaf_samples: list[Sample], session_metadata: dict[str, Any]) -> list[Sample]:
    agent = session_metadata.get("agent") or {}
    if not agent.get("prl_training_contract") or agent.get("prl_aborted"):
        return []
    if agent["prl_evaluation"]:
        return [max(leaf_samples, key=lambda s: s.metadata["leaf"]["node_id"])] if leaf_samples else []
    if agent.get("prl_reject_mismatch"):
        # Inspect BEFORE retry deduplication and auxiliary masking: a discarded
        # summary branch must not hide a mismatch in this trial.
        checks = {
            "checked_leaves": len(leaf_samples),
            "rejected": [
                {"leaf": s.metadata["leaf"], "mismatch": s.metadata.get("tito_session_mismatch"),
                 "reason": "mismatch" if s.metadata.get("tito_session_mismatch") else "missing_check"}
                for s in leaf_samples
                if s.metadata.get("tito_session_mismatch") != []
            ],
        }
        for sample in leaf_samples:
            sample.metadata["prl_mismatch_checks"] = checks
    from miles.rollout.session.v2.picker_hub.drop_same_prompt_retries import drop_same_prompt_retries

    return drop_same_prompt_retries(leaf_samples, session_metadata)


def mask_auxiliary_completions(samples: list[Sample], session_metadata: dict[str, Any]) -> list[Sample]:
    """Mask summary spans after upstream deduplication, preserving main actions."""
    nodes = {node["id"]: node for node in session_metadata["tree"]["nodes"]}
    kept = []
    masked = 0
    for sample in samples:
        response_start = len(sample.tokens) - sample.response_length
        for node_id in sample.metadata["leaf"]["path_node_ids"]:
            node = nodes[node_id]
            tag = node["turn_args"].get("user")
            if tag not in ("prl/action-v1", "prl/summary-v1"):
                raise ValueError(f"Missing/unknown training call tag at node {node_id}: {tag!r}")
            if tag == "prl/summary-v1":
                start, end = node["completion_span"]
                start = max(start - response_start, 0)
                end = min(end - response_start, sample.response_length)
                if end > start:
                    masked += sum(sample.loss_mask[start:end])
                    sample.loss_mask[start:end] = [0] * (end - start)
        if any(sample.loss_mask):
            kept.append(sample)
    logger.info("[prl] training samples: %d/%d leaves; %d auxiliary tokens masked", len(kept), len(samples), masked)
    return kept


def postprocess_samples(leaf_samples: list[Sample], session_metadata: dict[str, Any]) -> list[Sample]:
    from miles.rollout.session.v2.postprocessor_hub.default_postprocess import default_postprocess

    samples = default_postprocess(leaf_samples, session_metadata)
    if (session_metadata.get("agent") or {}).get("prl_evaluation"):
        return samples
    return mask_auxiliary_completions(samples, session_metadata)
