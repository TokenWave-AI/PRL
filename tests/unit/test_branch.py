"""transforms/branch.py: the branch template's plug-ins (select, allocate, credit) on a recorded-like trial, through the
framework (prl/transforms/core.py); no GPU, judge or sandbox (fake tokenizer, judge and teacher)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from prl.transforms import core

PROMPT = "<|im_start|>user\nTASK<|im_end|>\n<|im_start|>assistant\n<think>\n"


class Tok:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _trial(reward=0.0, logprobs=(-0.1, -0.1, -0.1)):
    def turn(thinking, keys):
        return f"{thinking}\n</think>\n\n{json.dumps({'commands': [{'keystrokes': keys}]})}<|im_end|>"

    def message(text):
        return f"\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    parts = [(turn("look", "ls\n"), 1), (message("a b"), 0), (turn("read", "cat a\n"), 1), (message("x"), 0),
             (turn("fix", "rm a\n"), 1)]
    lps = iter(logprobs)
    response, mask, lp = [], [], []
    for text, trained in parts:
        value = next(lps) if trained else 0.0
        response += [ord(c) for c in text]
        mask += [trained] * len(text)
        lp += [value] * len(text)
    return [SimpleNamespace(tokens=[ord(c) for c in PROMPT] + response, response_length=len(response), loss_mask=mask,
                            reward=reward, rollout_id=7, rollout_log_probs=lp, train_metadata=None,
                            metadata={"leaf": {"node_id": 0}, "reward": reward, "trial_dir": "/trials/t"})]


@pytest.fixture
def run(monkeypatch):
    for key in ("PRL_TRANSFORM_MAX_FAILURES", "PRL_TRANSFORM_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(core, "_failures", 0)
    monkeypatch.setenv("PRL_TRANSFORM", "branch")

    def go(args, samples=None):
        monkeypatch.setenv("PRL_TRANSFORM_ARGS", json.dumps(args))
        out = asyncio.run(core.run(samples or _trial(), Tok()))
        record = out[0].metadata["prl_transform"]
        return record, record["chain"][0]
    return go


def test_default_branches_a_failed_trial_from_its_middle_step(run):
    record, entry = run({})
    assert entry["status"] == "ok" and entry["record"]["branch"]["anchors"] == [{"step": 1, "n": 1}]
    [anchor] = record["reroll"]["anchors"]
    assert (anchor["step"], anchor["n"], anchor["credit"]) == (1, 1, "member")


def test_a_passed_trial_is_left_out_before_any_selection(run):
    record, entry = run({}, _trial(reward=1.0))
    assert "reroll" not in record and "only failed" in entry["record"]["branch"]["why"]
    record, _ = run({"allocate": {"name": "fixed", "only": "passed", "n": 2}}, _trial(reward=1.0))
    assert record["reroll"]["anchors"][0]["n"] == 2


def test_group_view_is_refused_for_now(run):
    record, entry = run({"allocate": {"name": "fixed", "view": "group"}})
    assert entry["status"] == "error" and "planned" in entry["reason"] and "reroll" not in record


def test_surprisal_picks_the_least_likely_steps(run):
    record, _ = run({"select": {"name": "surprisal", "top": 2}, "max_anchors": 2, "credit": "subgroup"},
                        _trial(logprobs=(-0.1, -0.2, -3.0)))
    assert [(a["step"], a["credit"]) for a in record["reroll"]["anchors"]] == [(1, "subgroup"), (2, "subgroup")]


def test_surprisal_ranks_past_the_steps_before_min_step(run):
    # step 0 is the least likely, but a branch from it is just a new trial: the next one is taken instead
    record, _ = run({"select": "surprisal"}, _trial(logprobs=(-5.0, -0.2, -3.0)))
    assert [a["step"] for a in record["reroll"]["anchors"]] == [2]


def test_first_bad_uses_the_judge_and_min_step_drops_step_zero(run, monkeypatch):
    from prl.algo import labeler

    async def grade(traj):
        return {0}, {2}
    monkeypatch.setattr(labeler, "grade", grade)
    record, _ = run({"select": "first_bad", "allocate": {"name": "fixed", "n": 3}, "credit": "subgroup"})
    assert [(a["step"], a["n"]) for a in record["reroll"]["anchors"]] == [(2, 3)]

    async def grade_first(traj):
        return set(), {0, 2}
    monkeypatch.setattr(labeler, "grade", grade_first)
    record, entry = run({"select": "first_bad"})
    assert "reroll" not in record and "min_step" in entry["record"]["branch"]["why"]

    async def no_answer(traj):
        return None
    monkeypatch.setattr(labeler, "grade", no_answer)
    _, entry = run({"select": "first_bad"})
    assert entry["status"] == "skipped"


def test_the_teacher_selection_and_a_users_own_plugins(run, monkeypatch):
    from transforms import branch, rewrite

    async def teacher_choice(traj, version, obs_chars, change_share):
        assert version == "reroll_v2"
        traj.steps[0].edit_thinking("look", "look closer")  # a correction before the step reaches the replay only
        return 2
    monkeypatch.setattr(rewrite, "teacher_choice", teacher_choice)
    record, _ = run({"select": {"name": "teacher", "prompt": "reroll_v2"}})
    [anchor] = record["reroll"]["anchors"]
    assert anchor["step"] == 2 and anchor["rewritten_steps"] == [0] and anchor["turns"][0]["reasoning"] == "look closer"

    mine = {"last": lambda traj: [traj.steps[-1]], "twice_each": lambda traj, anchors, k=2: [(a, k) for a in anchors]}
    monkeypatch.setattr(branch, "_resolve", lambda name: mine[name])
    record, _ = run({"select": "last", "allocate": {"name": "twice_each", "k": 4}})
    assert [(a["step"], a["n"]) for a in record["reroll"]["anchors"]] == [(2, 4)]
