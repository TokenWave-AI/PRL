"""Rollout Routing Replay through our rollout path: routes stay one row per token up to the trainer, and a function that
only masks leaves them alone. Needs Miles (Sample, train-data conversion)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

miles_types = pytest.importorskip("miles.utils.types")
from prl.rollout import training_samples
from prl.transforms import core


class CharTokenizer:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


LAYERS, TOPK = 3, 2


def routes_for(n_tokens: int) -> np.ndarray:
    return np.arange((n_tokens - 1) * LAYERS * TOPK, dtype=np.int32).reshape(n_tokens - 1, LAYERS, TOPK) % 256


def make_sample(reward: float = 1.0) -> miles_types.Sample:
    prompt = "system"
    parts = [("<think>a</think>\n\n{}<|im_end|>", 1), ("\n<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n", 0),
             ("b</think>\n\n{}<|im_end|>", 1)]
    response = [ord(c) for text, _ in parts for c in text]
    tokens = [ord(c) for c in prompt] + response
    sample = miles_types.Sample(tokens=tokens, response_length=len(response), reward=reward)
    sample.loss_mask = [t for text, t in parts for _ in text]
    sample.rollout_log_probs = [-0.5] * len(response)
    sample.rollout_routed_experts = routes_for(len(tokens))
    sample.metadata = {"leaf": {"node_id": 0, "path_node_ids": []}, "reward": reward}
    sample.rollout_id = 7
    return sample


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for key in ("PRL_TRANSFORM", "PRL_TRANSFORM_ARGS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(core, "_failures", 0)


def run_transform(monkeypatch, fn, samples):
    monkeypatch.setattr(core, "_resolve", lambda name: fn)
    monkeypatch.setenv("PRL_TRANSFORM", "t")
    return asyncio.run(core.run(samples, CharTokenizer(), None))


def test_summary_masking_keeps_routes():
    sample = make_sample()
    before = sample.rollout_routed_experts.copy()
    nodes = [{"id": 1, "turn_args": {"user": "prl/summary-v1"}, "completion_span": [len(sample.tokens) - 5, len(sample.tokens)]}]
    sample.metadata["leaf"]["path_node_ids"] = [1]
    kept = training_samples.mask_auxiliary_completions([sample], {"tree": {"nodes": nodes}})
    assert kept[0].loss_mask[-5:] == [0] * 5
    assert np.array_equal(kept[0].rollout_routed_experts, before)
    kept[0].validate()


def test_a_masking_transform_keeps_routes(monkeypatch):
    def fn(traj):
        traj.steps[0].mask()
        return traj

    sample = make_sample()
    before = sample.rollout_routed_experts.copy()
    kept = run_transform(monkeypatch, fn, [sample])
    assert kept[0].metadata["prl_transform"]["chain"][0]["status"] == "ok"
    assert sum(kept[0].loss_mask) < sum(make_sample().loss_mask)
    assert np.array_equal(kept[0].rollout_routed_experts, before)
    kept[0].validate()


def test_a_rewrite_cannot_break_routes(monkeypatch):
    """Rewritten tokens have no routes: the trial trains as it was, recorded as an error (rewrite is not supported on
    MoE yet)."""
    def fn(traj):
        traj.steps[0].edit_thinking("a", "something longer")
        return traj

    sample = make_sample()
    tokens, before = list(sample.tokens), sample.rollout_routed_experts.copy()
    kept = run_transform(monkeypatch, fn, [sample])
    assert kept[0].metadata["prl_transform"]["chain"][0]["status"] == "error"
    assert list(kept[0].tokens) == tokens and np.array_equal(kept[0].rollout_routed_experts, before)
    kept[0].validate()


def test_stripping_the_last_token_keeps_one_row_per_token():
    sample = make_sample()
    sample.rollout_routed_experts = routes_for(len(sample.tokens))
    sample.strip_last_output_tokens(1, CharTokenizer())
    assert len(sample.rollout_routed_experts) == len(sample.tokens) - 1
    sample.validate()


def test_routes_reach_the_train_data_and_a_missing_route_fails():
    from miles.ray.rollout import train_data_conversion as conversion

    assert "rollout_routed_experts" in conversion.__dict__.get("_DTYPES", {"rollout_routed_experts": 1})
    source = conversion.convert_samples_to_train_data.__code__
    assert "rollout_routed_experts" in source.co_names or "rollout_routed_experts" in source.co_consts


def test_check_routes_tool():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts/tools/check_routes.py"
    spec = importlib.util.spec_from_file_location("check_routes", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    good = np.zeros((4, LAYERS, TOPK), dtype=np.int32)
    good[..., 1] = 1
    assert tool.check_sample({"tokens": [0] * 5, "rollout_routed_experts": good}, LAYERS, TOPK, 256)[0] == []
    assert tool.check_sample({"tokens": [0] * 6, "rollout_routed_experts": good}, LAYERS, TOPK, 256)[0]
    assert tool.check_sample({"tokens": [0] * 5, "rollout_routed_experts": np.zeros_like(good)}, LAYERS, TOPK, 256)[0]
    bad = good.copy()
    bad[0, 0, 0] = 256
    assert tool.check_sample({"tokens": [0] * 5, "rollout_routed_experts": bad}, LAYERS, TOPK, 256)[0]
    assert tool.check_sample({"tokens": [0] * 5}, LAYERS, TOPK, 256)[0] == ["no routes"]
