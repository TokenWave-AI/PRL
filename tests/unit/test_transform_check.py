"""prl/transforms/check.py: recorded trials as samples, and what fails the pre-run check."""

from __future__ import annotations

import asyncio
import gzip
import json

import pytest

from prl.transforms import check, core


class CharTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


TRIALS = [{"name": "ok_trial", "reward": 1.0, "rollout_id": 3, "samples": [
    {"node_id": 4, "prompt": "TASK", "response": [{"text": "A", "trained": 1}, {"text": "o1", "trained": 0},
                                                  {"text": "BB", "trained": 1}]}]},
          {"name": "fail_trial", "reward": 0.0, "rollout_id": 4, "samples": [
    {"node_id": 9, "prompt": "P2", "response": [{"text": "C", "trained": 1}]},
    {"node_id": 4, "prompt": "TASK", "response": [{"text": "D", "trained": 1}, {"text": "o", "trained": 0}]}]}]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    for key in ("PRL_TRANSFORM", "PRL_TRANSFORM_ARGS"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "trials.json.gz"
    path.write_bytes(gzip.compress(json.dumps({"trials": TRIALS}).encode()))
    fns = {}
    monkeypatch.setattr(core, "_resolve", lambda name: fns[name])

    def select(fn):
        fns["t"] = fn
        monkeypatch.setenv("PRL_TRANSFORM", "t")
    return path, select


def _check(path, tmp_path):
    return asyncio.run(check.check(CharTokenizer(), path, tmp_path / "out", 10))


def test_trials_become_samples():
    s = check.samples(TRIALS[0], CharTokenizer(), "/t")
    assert len(s) == 1 and s[0].tokens == [ord(c) for c in "TASKAo1BB"] and s[0].loss_mask == [1, 0, 0, 1, 1]
    assert (s[0].response_length, s[0].metadata["trial_dir"], s[0].reward) == (5, "/t", 1.0)


def test_passes_and_records(fixture, tmp_path):
    path, select = fixture

    def keep_first(traj):
        for step in traj.steps[1:]:
            step.mask()

    select(keep_first)
    summary = _check(path, tmp_path)
    assert [(r["trial"], r["status"], r["trained_tokens"], r["changed_steps"]) for r in summary["trials"]] == [
        ("ok_trial", ["ok"], [3, 1], {"masked": [1]}), ("fail_trial", ["ok"], [2, 1], {"masked": [1]})]
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["transform"] == "t"
    assert (tmp_path / "out" / "ok_trial" / "transform.json").exists()


@pytest.mark.parametrize("fn, error", [(lambda traj: 1 / 0, "ZeroDivisionError"),
                                       (lambda traj: traj.skip("no"), "skipped all 2"),
                                       (lambda traj: [s.mask() for s in traj.steps] and None, "no trained token")])
def test_failures_stop_the_run(fixture, tmp_path, fn, error):
    path, select = fixture
    select(fn)
    with pytest.raises(check.CheckFailed, match=error):
        _check(path, tmp_path)


def test_nothing_selected(fixture, tmp_path):
    path, _ = fixture
    assert _check(path, tmp_path) == {"transform": "none"}


def test_the_shipped_trials_are_readable():
    data = json.loads(gzip.decompress(check.FIXTURE.read_bytes()))
    assert len(data["trials"]) == 8 and {t["reward"] for t in data["trials"]} == {0.0, 1.0}
    assert any(len(t["samples"]) > 1 for t in data["trials"])  # one trial split by a summary
    assert all(seg["trained"] in (0, 1) for t in data["trials"] for leaf in t["samples"] for seg in leaf["response"])


def test_the_line_names_weight_changes(fixture, tmp_path, monkeypatch, capsys):
    path, select = fixture

    def push_down(traj):
        traj.steps[0].weight = -1.0  # trained tokens stay as they are: only the line's step counts show it

    select(push_down)
    monkeypatch.setattr(check, "Tokenizer", lambda _: CharTokenizer())
    monkeypatch.setenv("MODEL_HF_DIR", "unused")
    monkeypatch.setenv("PRL_TRANSFORM_CHECK_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("PRL_TRANSFORM_FIXTURE", str(path))
    assert check.main() == 0
    assert "ok_trial: reward 1.0, 2 steps, ok, steps changed: 1 weighted, trained tokens 3 -> 3" in capsys.readouterr().out


def test_a_skip_names_its_reason_and_a_reroll_needs_its_arm_in_a_job(fixture, tmp_path, monkeypatch, capsys):
    path, select = fixture

    def skip_failures(traj, why: str = "a failure"):
        if not traj.reward:
            traj.skip(why)

    select(skip_failures)
    monkeypatch.setattr(check, "Tokenizer", lambda _: CharTokenizer())
    monkeypatch.delenv("EXTRA_ARGS", raising=False)
    monkeypatch.setenv("MODEL_HF_DIR", "unused")
    monkeypatch.setenv("PRL_TRANSFORM_CHECK_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("PRL_TRANSFORM_FIXTURE", str(path))
    assert check.main() == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if "fail_trial:" in x)
    assert ", skipped, steps changed: none," in line and line.endswith(" s; skipped: a failure")
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["params"] == [{"why": "a failure"}]

    async def rerolls(*_args, **_kwargs):
        return {"transform": "t", "trials": [{"trial": "x", "reward": 0.0, "steps": 2, "status": ["ok"], "seconds": 0,
                                              "trained_tokens": [1, 1], "changed_steps": {"rerolled": [1]},
                                              "skipped": []}]}

    monkeypatch.setattr(check, "check", rerolls)
    assert check.main() == 0  # no EXTRA_ARGS: the try on recorded trials, outside a job
    monkeypatch.setenv("EXTRA_ARGS", "--use-tis")
    assert check.main() == 1 and "ARM=ota_train_transform_reroll" in capsys.readouterr().err
    monkeypatch.setenv("EXTRA_ARGS", "--custom-reward-post-process-path prl.algo.added_rewards.group_rewards")
    assert check.main() == 0
