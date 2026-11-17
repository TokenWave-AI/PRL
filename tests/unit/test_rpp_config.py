"""REINFORCE++ wiring that lives in our scripts, not in Miles: the 32 x 8 arm and its smoke against the
no-turn-cap GRPO arms they overlay, the transform arm's ALGO=reinforce_plus_plus, the launcher passing
--normalize-advantages (Miles refuses the estimator without it) and smoke receipts of this algorithm. Bash + stdlib."""

from __future__ import annotations

import subprocess

from test_find_smoke import ROOT, _smoke, find
from test_ppo_config import arm_env, diff

RPP_KEYS = {"ADVANTAGE_ESTIMATOR", "NORMALIZE_ADVANTAGES", "RUN_DIR", "HARBOR_TRIALS_DIR"}


def test_rpp_arm_only_changes_the_estimator() -> None:
    grpo = arm_env("ota_train_full_noturncap_b32", PRL_RUN_ID="r1")
    rpp = arm_env("ota_train_full_rpp_b32", PRL_RUN_ID="r1")
    assert diff(grpo, rpp) == RPP_KEYS
    assert (rpp["ADVANTAGE_ESTIMATOR"], rpp["NORMALIZE_ADVANTAGES"]) == ("reinforce_plus_plus", "1")
    assert (rpp["ROLLOUT_BATCH"], rpp["NUM_ROLLOUT"], rpp["KL_LOSS_COEF"]) == ("32", "68", "0.001")
    assert "HARBOR_AGENT_MAX_ITERATIONS" not in rpp and not {"CRITIC_LR", "CRITIC_WARMUP_STEPS"} & rpp.keys()
    assert rpp["RUN_DIR"].endswith("/runs/reinforce_plus_plus_ota_train_full_r1")


def test_transform_and_smoke_arms() -> None:
    tf = arm_env("ota_train_transform", PRL_RUN_ID="r1", TRANSFORM="rewrite", ALGO="reinforce_plus_plus")
    base = arm_env("ota_train_full_rpp_b32", PRL_RUN_ID="r1")
    judge = {k for k in tf if k.startswith("PRL_JUDGE_")}
    assert diff(base, tf) == judge | {"PRL_TRANSFORM", "PRL_TRANSFORM_ARGS", "TRANSFORM", "ALGO",
                                      "RUN_DIR", "HARBOR_TRIALS_DIR"}
    assert tf["RUN_DIR"].endswith("/runs/reinforce_plus_plus_ota_train_tf_rewrite_r1")
    smoke = arm_env("ota_train_smoke_rpp")
    assert diff(arm_env("ota_train_smoke_noturncap"), smoke) == RPP_KEYS
    assert smoke["RUN_DIR"].endswith("/runs/reinforce_plus_plus_ota_train_smoke_123")  # find_smoke's glob
    assert smoke["MAX_CONTEXT"] == tf["MAX_CONTEXT"]  # the chain's smoke step requires the training arm's context


def test_launcher_whitens_rpp_advantages_only() -> None:
    lines = (ROOT / "scripts/lib/container_main.sh").read_text().splitlines()
    rule = next(line for line in lines if "= reinforce_plus_plus" in line)
    for estimator, normalize, whiten in (("reinforce_plus_plus", "1", True), ("grpo", "1", False),
                                         ("reinforce_plus_plus", "0", False)):
        env = {"PATH": "/usr/bin:/bin", "ADVANTAGE_ESTIMATOR": estimator, "NORMALIZE_ADVANTAGES": normalize}
        out = subprocess.run(["bash", "-c", f'GRPO_ARGS=(); {rule}; echo "${{GRPO_ARGS[@]}}"'], capture_output=True,
                             text=True, check=True, env=env).stdout
        assert ("--normalize-advantages" in out) == whiten, (estimator, normalize, out)


def test_rpp_receipts_are_their_own(tmp_path) -> None:
    _smoke(tmp_path, "grpo_ota_train_smoke_1", 100, model="qwen3.5-9B", context="65536")
    assert find(ROOT, tmp_path, "reinforce_plus_plus", "qwen3.5-9B", "65536") == ""  # a GRPO receipt does not vouch
    _smoke(tmp_path, "reinforce_plus_plus_ota_train_smoke_2", 200, algo="reinforce_plus_plus",
           model="qwen3.5-9B", context="65536")
    assert find(ROOT, tmp_path, "reinforce_plus_plus", "qwen3.5-9B", "65536").endswith("_smoke_2")
    assert find(ROOT, tmp_path, "grpo", "qwen3.5-9B", "65536").endswith("grpo_ota_train_smoke_1")
