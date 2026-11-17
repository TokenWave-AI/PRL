"""Trajectory transforms: the judge profile, the transform arms (configs/arms/ota_*transform*), and
the smoke acceptance of a transform (scripts/checks/grpo_acceptance.py check_transform). Bash + stdlib only."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_ppo_config import arm_env, diff, load_acceptance


def test_judge_profile() -> None:
    j = arm_env("ota_judge")  # no endpoint in the repo: the site profile or the environment names it
    assert (j["PRL_JUDGE_API_BASE"], j["PRL_JUDGE_MODEL"], j["PRL_JUDGE_EXTRA_BODY"]) == ("", "", "")
    assert j["PRL_JUDGE_KEY_FILE"].endswith("/private/judge_api_key")
    assert (j["PRL_JUDGE_PROMPT"], j["PRL_JUDGE_CONCURRENCY"], j["PRL_JUDGE_FALLBACK_MODELS"]) == ("paper", "32", "")
    assert "PRL_JUDGE_KEY" not in j  # only the file holds the key
    site = arm_env("ota_judge", PRL_JUDGE_API_BASE="https://x/v1", PRL_JUDGE_MODEL="m",
                   PRL_JUDGE_EXTRA_BODY='{"reasoning_effort": "medium"}', PRL_JUDGE_FALLBACK_MODELS="a,b")
    assert (site["PRL_JUDGE_API_BASE"], site["PRL_JUDGE_MODEL"], site["PRL_JUDGE_FALLBACK_MODELS"]) == (
        "https://x/v1", "m", "a,b")
    assert json.loads(site["PRL_JUDGE_EXTRA_BODY"]) == {"reasoning_effort": "medium"}


def test_transform_arms() -> None:
    grpo = arm_env("ota_train_full_noturncap_b32", PRL_RUN_ID="r1")
    tf = arm_env("ota_train_transform", PRL_RUN_ID="r1", TRANSFORM="my_mask")
    judge = {k for k in tf if k.startswith("PRL_JUDGE_")}
    assert diff(grpo, tf) == judge | {"PRL_TRANSFORM", "PRL_TRANSFORM_ARGS", "TRANSFORM", "RUN_DIR",
                                      "HARBOR_TRIALS_DIR"}
    assert (tf["PRL_TRANSFORM"], tf["PRL_TRANSFORM_ARGS"]) == ("my_mask", "")
    assert tf["RUN_DIR"].endswith("/runs/grpo_ota_train_tf_my_mask_r1")
    assert (tf["ROLLOUT_BATCH"], tf["NUM_ROLLOUT"]) == ("32", "68") and "HARBOR_AGENT_MAX_ITERATIONS" not in tf
    ppo = arm_env("ota_train_transform", PRL_RUN_ID="r1", TRANSFORM="my_mask", ALGO="ppo")
    base = arm_env("ota_train_full_ppo_b32", PRL_RUN_ID="r1")
    assert diff(base, ppo) == judge | {"PRL_TRANSFORM", "PRL_TRANSFORM_ARGS", "TRANSFORM", "ALGO",
                                       "RUN_DIR", "HARBOR_TRIALS_DIR"}
    assert ppo["RUN_DIR"].endswith("/runs/ppo_ota_train_tf_my_mask_r1") and ppo["NUM_ROLLOUT"] == "88"


def test_transform_smoke_arms_and_args_file(tmp_path) -> None:
    smoke = arm_env("ota_train_smoke_transform", TRANSFORM="atlas")
    assert diff(arm_env("ota_train_smoke_noturncap"), smoke) == {
        k for k in smoke if k.startswith("PRL_JUDGE_")} | {"PRL_TRANSFORM",
                                                                 "PRL_TRANSFORM_ARGS", "TRANSFORM",
                                                                 "HARBOR_ENV_KWARGS"}
    # half of the trials stay whole, for transforms that rewrite (the plain smoke splits every one)
    assert json.loads(smoke["HARBOR_ENV_KWARGS"])["prl_force_summary_once"] == 0.5
    assert json.loads(arm_env("ota_train_smoke_noturncap")["HARBOR_ENV_KWARGS"])[
        "prl_force_summary_once"] is True
    ppo = arm_env("ota_train_smoke_transform", TRANSFORM="atlas", ALGO="ppo", PRL_RUN_ID="s1",
                  PRL_PPO_SMOKE_ROLLOUTS="2")
    assert (ppo["ADVANTAGE_ESTIMATOR"], ppo["RUN_DIR"].rsplit("/", 1)[1]) == ("ppo", "ppo_ota_train_smoke_s1")
    args = tmp_path / "args.json"
    args.write_text('{"a": 1, "b": 2}')  # commas: qsub -v cannot carry it, the file can
    tf = arm_env("ota_train_transform", PRL_RUN_ID="r1", TRANSFORM="x",
                 PRL_TRANSFORM_ARGS_FILE=str(args))
    assert json.loads(tf["PRL_TRANSFORM_ARGS"]) == {"a": 1, "b": 2}


def test_reroll_smoke_arm() -> None:
    smoke = arm_env("ota_train_smoke_transform", TRANSFORM="reroll_smoke")
    reroll = arm_env("ota_train_smoke_reroll")
    assert diff(smoke, reroll) == {"EXTRA_ARGS"} and reroll["PRL_TRANSFORM"] == "reroll_smoke"
    assert reroll["EXTRA_ARGS"] == smoke["EXTRA_ARGS"] + \
        " --custom-reward-post-process-path prl.algo.added_rewards.group_rewards"


def test_async_smoke_arm() -> None:
    reroll = arm_env("ota_train_smoke_reroll")
    async_ = arm_env("ota_train_smoke_async")
    assert diff(reroll, async_) == {"ASYNC", "EXTRA_ARGS", "PRL_REJECT_MISMATCH"}
    assert async_["ASYNC"] == "1" and async_["COLOCATE"] == "0"
    # mismatch refill is synchronous only (prl/rollout/harbor_training.py)
    assert reroll["PRL_REJECT_MISMATCH"] == "1" and async_["PRL_REJECT_MISMATCH"] == "0"
    assert async_["EXTRA_ARGS"].startswith(reroll["EXTRA_ARGS"] + " --fully-async --max-weight-staleness 2 ")
    assert "--use-tis" in async_["EXTRA_ARGS"]  # Miles: async needs rollout log probs, TIS or the old actor


def test_reroll_training_arm() -> None:
    for algo in ("grpo", "ppo"):
        base = arm_env("ota_train_transform", PRL_RUN_ID="r1", TRANSFORM="rewrite", ALGO=algo)
        arm = arm_env("ota_train_transform_reroll", PRL_RUN_ID="r1", TRANSFORM="rewrite", ALGO=algo)
        assert diff(base, arm) == {"EXTRA_ARGS"} and arm["RUN_DIR"] == base["RUN_DIR"]
        assert arm["EXTRA_ARGS"].endswith("--custom-reward-post-process-path prl.algo.added_rewards.group_rewards")
    with pytest.raises(subprocess.CalledProcessError):  # the arm refuses REINFORCE++
        arm_env("ota_train_transform_reroll", PRL_RUN_ID="r1", TRANSFORM="x", ALGO="reinforce_plus_plus")


def _leaf(trial: str, trained: int, *, before: int = 9, after: int = 3, status: str = "ok", chain=("atlas",),
          weights=None) -> dict:
    record = {"chain": [{"transform": name, "status": status} for name in chain], "trained_tokens_before": before,
              "trained_tokens_after": after, "tokens_changed": False}
    return {"loss_mask": [1] * trained + [0] * 2, "train_metadata": {"step_weights": weights} if weights else {},
            "metadata": {"trial_dir": trial, "prl_transform": record}}


@pytest.mark.parametrize("case", ["ok", "unmasked_leaf", "too_many_skipped", "error", "other_transform",
                                  "nothing_changed", "weights_only", "no_pre_check"])
def test_transform_acceptance(tmp_path: Path, monkeypatch, case: str) -> None:
    acc = load_acceptance()
    a, b = tmp_path / "a", tmp_path / "b"
    for trial in (a, b):
        trial.mkdir()
        (trial / "atlas_labels.json").write_text(json.dumps({"served_model": "gpt-6.1-sol"}))
    if case != "no_pre_check":
        (tmp_path / "transform_check").mkdir()
        (tmp_path / "transform_check" / "summary.json").write_text(json.dumps({"transform": "atlas", "trials": [{}]}))
    batch = [_leaf(str(a), 1), _leaf(str(a), 2), _leaf(str(b), 3)]  # trial a: two leaves (a summary), 1 + 2 = 3 kept
    batch = {"unmasked_leaf": [_leaf(str(a), 3), _leaf(str(b), 9)],
             "too_many_skipped": [_leaf(str(a), 3), _leaf(str(b), 9, after=9, status="skipped")],
             "error": [_leaf(str(a), 3, status="error")],
             "other_transform": [_leaf(str(a), 3, chain=("x",))],
             "nothing_changed": [_leaf(str(a), 9, after=9)],
             "weights_only": [_leaf(str(a), 9, after=9, weights=[1.0, 0.3])]}.get(case, batch)
    monkeypatch.setattr(acc, "load_samples", lambda run, n: batch)
    if case in ("ok", "weights_only"):
        out = acc.check_transform(tmp_path, (0,), "atlas")
        assert out["judge_served_models"] == ["gpt-6.1-sol"] and out["pre_check_trials"] == 1
        if case == "ok":
            assert (out["transform_trials"], out["transform_skipped"], out["kept_token_share"]) == (2, 0, 0.3333)
    else:
        with pytest.raises((AssertionError, FileNotFoundError)):
            acc.check_transform(tmp_path, (0,), "atlas")


@pytest.mark.parametrize("case", ["ok", "reroll_missing", "wrong_group_reward", "tokens_differ", "none_ok",
                                  "subgroup_unmarked", "wrong_added_id"])
def test_reroll_acceptance(tmp_path: Path, monkeypatch, case: str) -> None:
    acc = load_acceptance()
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "reroll" / "s2_b0").mkdir(parents=True)
    b.mkdir()
    (a / "reroll" / "s2_b0" / "request.json").write_text(json.dumps({"turns": [{}, {}]}))
    (tmp_path / "transform_check").mkdir()
    (tmp_path / "transform_check" / "summary.json").write_text(json.dumps({"transform": "reroll_smoke", "trials": [{}]}))
    outcome = {"step": 2, "branch": 0, "credit": "subgroup", "status": "ok", "reward": 1.0, "trained_tokens": 4,
               "replay": {"turns": 2, "same": 1, "differ": [1], "first_message_same": True}}
    trial = _leaf(str(a), 9, after=9, chain=("reroll_smoke",))
    trial["metadata"]["prl_transform"]["reroll"] = [outcome]
    if case != "subgroup_unmarked":
        trial["train_metadata"]["branch_anchors"] = [{"step": 2, "start": 5, "credit": "subgroup"}]
    trial["reward"] = 0.0
    failed = _leaf(str(b), 9, after=9, chain=("reroll_smoke",))
    failed["metadata"]["prl_transform"]["reroll"] = [{"step": 1, "branch": 0, "credit": "member",
                                                            "status": "failed", "reason": "no verifier reward"}]
    marks = {"group_reward": 1.0, "added_id": "reroll:2:0", "branch": {"anchor": 2, "credit": "subgroup"}}
    leaf = {"loss_mask": [1, 1, 0], "reward": 0.0, "train_metadata": marks,
            "metadata": {"trial_dir": "/trials/r", "prl_reroll": outcome, "prl_reroll_of": str(a)}}
    batch = [trial, failed, leaf, dict(leaf)]  # the branch in two leaves, 2 + 2 trained tokens
    if case == "reroll_missing":
        batch = [trial, failed]
    elif case == "wrong_group_reward":
        batch = [trial, failed, {**leaf, "train_metadata": {**marks, "group_reward": 0.0}}, dict(leaf)]
    elif case == "wrong_added_id":
        batch = [trial, failed, {**leaf, "train_metadata": {**marks, "added_id": "reroll"}}, dict(leaf)]
    elif case == "tokens_differ":
        batch = [trial, failed, leaf]
    elif case == "none_ok":
        batch = [failed]
    monkeypatch.setattr(acc, "load_samples", lambda run, n: batch)
    if case == "ok":
        out = acc.check_transform(tmp_path, (0,), "reroll_smoke")
        assert (out["reroll_requested_trials"], out["reroll_branches"], out["reroll_ok_branches"],
                out["reroll_subgroup_branches"], out["reroll_rewards"]) == (2, 2, 1, 1, [1.0])
        assert (out["reroll_replayed_turns_same"], out["reroll_replayed_turns_differ"]) == (1, 1)
        assert out["reroll_failed"] == ["no verifier reward"]
    else:
        with pytest.raises(AssertionError):
            acc.check_transform(tmp_path, (0,), "reroll_smoke")


def _batch_sample(trial: Path, group: int, rollout: int, reward: float, **meta) -> dict:
    trial.mkdir(parents=True, exist_ok=True)
    (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": reward}}}))
    return {"status": "completed", "reward": reward, "tokens": [1, 2, 3], "response_length": 2, "loss_mask": [1, 1],
            "weight_versions": [[{"version": 1, "abs_start": 1, "abs_end": 3}]], "group_index": group,
            "rollout_id": rollout,
            "metadata": {"tito_session_mismatch": [], "prl_reject_mismatch": True,
                         "prl_mismatch_checks": {"rejected": [], "checked_leaves": 1},
                         "prl_mismatch_retries": 0, "accumulated_token_ids": [1, 2, 3], "trial_dir": str(trial),
                         **meta}}


@pytest.mark.parametrize("case", ["ok", "wrong_group_reward", "reroll_mismatch"])
def test_the_batch_check_takes_rerolls_as_added_members(tmp_path: Path, monkeypatch, case: str) -> None:
    """A reroll shares its trial's rollout_id but has its own trial directory and reward, and no mismatch retries of
    its own; in a smoke, check_batch failed on it before check_rerolls could look."""
    acc = load_acceptance()
    batch = [_batch_sample(tmp_path / f"t{g}_{r}", g, 8 * g + r, 0.0) for g in range(8) for r in range(8)]
    reroll = _batch_sample(tmp_path / "reroll", 0, 3, 0.0, prl_reroll={"status": "ok"},
                           prl_reroll_of=str(tmp_path / "t0_3"))
    del reroll["metadata"]["prl_mismatch_retries"]
    (tmp_path / "reroll" / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}))
    reroll["train_metadata"] = {"group_reward": 0.0 if case == "wrong_group_reward" else 1.0, "added_id": "reroll"}
    if case == "reroll_mismatch":
        reroll["metadata"]["tito_session_mismatch"] = [{"leaf": 0}]
    monkeypatch.setattr(acc, "load_samples", lambda run, n: batch + [reroll])
    if case == "ok":
        out = acc.check_batch(tmp_path, 0)
        assert (out["groups"], out["rollouts"], out["leaves"], out["reroll_samples"]) == (8, 64, 65, 1)
    else:
        with pytest.raises(AssertionError):
            acc.check_batch(tmp_path, 0)


@pytest.mark.parametrize("batch_no, versions, staleness, passed", [
    (0, {1}, 0, True), (1, {1, 2}, 0, False), (1, {1, 2}, 2, True), (1, {1, 2}, 1, True), (2, {1, 3}, 1, False),
    (0, {2}, 2, False)])
def test_the_async_batch_check_allows_versions_within_the_staleness(tmp_path: Path, monkeypatch, batch_no, versions,
                                                                    staleness, passed) -> None:
    """ASYNC=1: batch b (the trainer at version b + 1) may hold rollouts of up to --max-weight-staleness older weights."""
    acc = load_acceptance()
    ordered = sorted(versions)
    batch = [_batch_sample(tmp_path / f"t{g}_{r}", g, 8 * g + r, 0.0) for g in range(8) for r in range(8)]
    for i, sample in enumerate(batch):
        sample["weight_versions"] = [[{"version": ordered[i % len(ordered)], "abs_start": 1, "abs_end": 3}]]
    monkeypatch.setattr(acc, "load_samples", lambda run, n: batch)
    if passed:
        assert acc.check_batch(tmp_path, batch_no, staleness)["weight_versions"] == ordered
    else:
        with pytest.raises(AssertionError):
            acc.check_batch(tmp_path, batch_no, staleness)


def test_the_batch_check_refuses_a_trained_token_outside_every_model_call(tmp_path: Path, monkeypatch) -> None:
    """A trained token lies in a model call's span (weight_versions), never in a tool output or user message (one
    smoke: 324107 trained tokens, none outside)."""
    acc = load_acceptance()
    batch = [_batch_sample(tmp_path / f"t{g}_{r}", g, 8 * g + r, 0.0) for g in range(8) for r in range(8)]
    monkeypatch.setattr(acc, "load_samples", lambda run, n: batch)
    acc.check_batch(tmp_path, 0)
    batch[5]["weight_versions"] = [[{"version": 1, "abs_start": 2, "abs_end": 3}]]  # token 1 came from the environment
    with pytest.raises(AssertionError):
        acc.check_batch(tmp_path, 0)
