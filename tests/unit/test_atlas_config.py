"""ATLAS: the ATLAS arm against the no-turn-cap GRPO arm it overlays (configs/arms/ota_atlas_judge.env).
The smoke acceptance of transforms is in test_transform_config.py. Bash + stdlib only."""

from __future__ import annotations

import json

from test_ppo_config import arm_env, diff

JUDGE = {"PRL_TRANSFORM", "PRL_TRANSFORM_ARGS", "PRL_JUDGE_API_BASE", "PRL_JUDGE_KEY_FILE",
         "PRL_JUDGE_MODEL", "PRL_JUDGE_PROMPT", "PRL_JUDGE_EXTRA_BODY", "PRL_JUDGE_CONCURRENCY",
         "PRL_JUDGE_FALLBACK_MODELS"}


def test_atlas_arms_only_add_the_judge() -> None:
    grpo = arm_env("ota_train_full_noturncap_b32", PRL_RUN_ID="r1")
    atlas = arm_env("ota_train_atlas_noturncap_b32", PRL_RUN_ID="r1")
    assert diff(grpo, atlas) == JUDGE | {"RUN_DIR", "HARBOR_TRIALS_DIR"}
    assert atlas["RUN_DIR"].endswith("/runs/grpo_ota_train_atlas_r1")
    assert "HARBOR_AGENT_MAX_ITERATIONS" not in atlas  # no turn cap, as the GRPO arm


def test_atlas_judge_settings() -> None:
    a = arm_env("ota_train_atlas_noturncap_b32", PRL_RUN_ID="r1", PRL_JUDGE_MODEL="m",
                PRL_JUDGE_EXTRA_BODY='{"enable_thinking": true}')
    assert (a["PRL_TRANSFORM"], a["PRL_TRANSFORM_ARGS"]) == ("atlas", "")  # hard mask
    assert (a["PRL_JUDGE_MODEL"], a["PRL_JUDGE_PROMPT"]) == ("m", "paper")  # the site's judge, the paper's prompt
    assert json.loads(a["PRL_JUDGE_EXTRA_BODY"]) == {"enable_thinking": True}
    assert (a["ROLLOUT_BATCH"], a["NUM_ROLLOUT"]) == ("32", "68")


def test_signed_atlas_arm_is_the_transform_arm_with_a_low_effort_judge() -> None:
    tf = arm_env("ota_train_transform", PRL_RUN_ID="r1", TRANSFORM="atlas_signed")
    signed = arm_env("ota_train_atlas_signed", PRL_RUN_ID="r1")
    assert diff(tf, signed) == {"PRL_JUDGE_EXTRA_BODY"}
    assert json.loads(signed["PRL_JUDGE_EXTRA_BODY"]) == {"reasoning_effort": "low"}
    assert signed["PRL_TRANSFORM"] == "atlas_signed" and signed["PRL_JUDGE_MODEL"] == tf["PRL_JUDGE_MODEL"]
    assert signed["RUN_DIR"].endswith("/runs/grpo_ota_train_tf_atlas_signed_r1")
    assert (signed["ROLLOUT_BATCH"], signed["NUM_ROLLOUT"]) == ("32", "68") and "HARBOR_AGENT_MAX_ITERATIONS" not in signed


def test_signed_atlas_smoke_arm_is_the_transform_smoke_with_a_low_effort_judge() -> None:
    smoke = arm_env("ota_train_smoke_transform", TRANSFORM="atlas_signed", JOB_ID="7")
    signed = arm_env("ota_train_smoke_atlas_signed", JOB_ID="7")
    assert diff(smoke, signed) == {"PRL_JUDGE_EXTRA_BODY"}
    assert json.loads(signed["PRL_JUDGE_EXTRA_BODY"]) == {"reasoning_effort": "low"}
    assert signed["PRL_TRANSFORM"] == "atlas_signed"
