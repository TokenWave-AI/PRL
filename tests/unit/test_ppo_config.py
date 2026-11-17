"""PPO wiring that lives in our scripts, not in Miles: the two PPO arms against the GRPO arms they overlay, the
critic resume choice (scripts/lib/relay.sh critic_load, used by scripts/lib/container_main.sh), checkpoint
pruning of ckpt_critic/ (scripts/tools/prune_ckpts.sh) and the PPO smoke acceptance's log checks
(scripts/checks/grpo_acceptance.py). Bash + stdlib only."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PPO_KEYS = {"ADVANTAGE_ESTIMATOR", "CRITIC_WARMUP_STEPS", "CRITIC_LR", "GAE_GAMMA", "GAE_LAMBDA",
            "NORMALIZE_ADVANTAGES", "PRL_VALUE_HEAD_INIT"}


def arm_env(arm: str, **env: str) -> dict[str, str]:
    script = (f'set -euo pipefail; set -a; source configs/sites/example.env; '
              f'source configs/arms/common_ota_single_node.env; source configs/arms/{arm}.env; set +a; env -0')
    out = subprocess.run(["bash", "-c", script], cwd=ROOT, capture_output=True, check=True,
                         env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "USER": "u", "PRL_ROOT": str(ROOT),
                              "JOB_ID": "123", **env}).stdout.decode()
    skip = {"PWD", "SHLVL", "_", "OLDPWD"}
    return {k: v for k, v in (kv.split("=", 1) for kv in out.split("\0") if kv) if k not in skip}


def diff(a: dict[str, str], b: dict[str, str]) -> set[str]:
    return {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}


def test_full_ppo_arm_only_changes_the_algorithm() -> None:
    grpo = arm_env("ota_train_full", PRL_RUN_ID="r1")
    ppo = arm_env("ota_train_full_ppo", PRL_RUN_ID="r1")
    assert diff(grpo, ppo) == PPO_KEYS | {"NUM_ROLLOUT", "PRL_KEEP_ITERS", "RUN_DIR", "HARBOR_TRIALS_DIR"}
    assert "ADVANTAGE_ESTIMATOR" not in grpo  # run.sh passes only set knobs: GRPO launches are unchanged
    assert (ppo["ADVANTAGE_ESTIMATOR"], ppo["CRITIC_WARMUP_STEPS"], ppo["NUM_ROLLOUT"]) == ("ppo", "10", "44")
    assert (ppo["CRITIC_LR"], ppo["GAE_GAMMA"], ppo["GAE_LAMBDA"], ppo["NORMALIZE_ADVANTAGES"]) == (
        "1e-5", "1.0", "1.0", "1")
    assert ppo["RUN_DIR"].endswith("/runs/ppo_ota_train_full_r1")
    assert ppo["HARBOR_TRIALS_DIR"] == ppo["RUN_DIR"] + "/harbor_trials/job_123"
    # Miles' value head init unless asked (patches miles/0006, 0008): what the earlier PPO runs had
    assert ppo["PRL_VALUE_HEAD_INIT"] == "upstream"
    zero = arm_env("ota_train_full_ppo", PRL_RUN_ID="r1", PRL_VALUE_HEAD_INIT="zero")
    assert diff(ppo, zero) == {"PRL_VALUE_HEAD_INIT"}


def test_b32_arm_halves_the_batch_over_the_same_prompts() -> None:
    ppo = arm_env("ota_train_full_ppo", PRL_RUN_ID="r1")
    half = arm_env("ota_train_full_ppo_b32", PRL_RUN_ID="r1")
    assert diff(ppo, half) == {"ROLLOUT_BATCH", "OVER_SAMPLING_BATCH", "CRITIC_WARMUP_STEPS", "NUM_ROLLOUT",
                               "PRL_KEEP_ITERS"}
    assert (half["ROLLOUT_BATCH"], half["OVER_SAMPLING_BATCH"], half["N_SAMPLES"]) == ("32", "32", "8")
    assert (half["CRITIC_WARMUP_STEPS"], half["NUM_ROLLOUT"], half["PRL_KEEP_ITERS"]) == ("20", "88", "59")

    def prompts(env: dict[str, str], rollouts: str) -> int:
        return int(env[rollouts]) * int(env["ROLLOUT_BATCH"])

    # Same prompts for the critic-only phase and in total, so the actor's too
    assert prompts(half, "CRITIC_WARMUP_STEPS") == prompts(ppo, "CRITIC_WARMUP_STEPS")
    assert prompts(half, "NUM_ROLLOUT") == prompts(ppo, "NUM_ROLLOUT") == 2816


def test_smoke_ppo_arm_runs_as_two_jobs() -> None:
    grpo = arm_env("ota_train_smoke")
    first = arm_env("ota_train_smoke_ppo", PRL_RUN_ID="s1", PRL_PPO_SMOKE_ROLLOUTS="2")
    second = arm_env("ota_train_smoke_ppo", PRL_RUN_ID="s1", PRL_PPO_SMOKE_ROLLOUTS="3")
    assert diff(grpo, first) == PPO_KEYS | {"NUM_ROLLOUT", "EXTRA_ARGS", "RUN_DIR", "HARBOR_TRIALS_DIR",
                                            "PRL_RUN_ID", "PRL_PPO_SMOKE_ROLLOUTS"}
    # One 3-rollout run (the LR scheduler checkpoint pins the run length); job A only exits early
    assert (first["NUM_ROLLOUT"], second["NUM_ROLLOUT"], first["CRITIC_WARMUP_STEPS"]) == ("3", "3", "1")
    assert first["EXTRA_ARGS"] == grpo["EXTRA_ARGS"] + " --debug-exit-after-rollout 2"
    assert second["EXTRA_ARGS"] == grpo["EXTRA_ARGS"]
    assert first["RUN_DIR"] == second["RUN_DIR"]
    for bad in ({}, {"PRL_PPO_SMOKE_ROLLOUTS": "4"}):
        with pytest.raises(subprocess.CalledProcessError):
            arm_env("ota_train_smoke_ppo", PRL_RUN_ID="s1", **bad)


def critic_load(run: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'source scripts/lib/relay.sh; critic_load "{run}" /base'], cwd=ROOT,
                          capture_output=True, text=True, check=False)


def mark(run: Path, sub: str, iteration: int) -> None:
    (run / sub / f"iter_{iteration:07d}").mkdir(parents=True, exist_ok=True)
    (run / sub / "latest_checkpointed_iteration.txt").write_text(f"{iteration}\n")


def test_critic_resumes_only_with_the_actor(tmp_path: Path) -> None:
    assert critic_load(tmp_path).stdout.strip() == "/base"            # fresh run
    mark(tmp_path, "ckpt_critic", 3)                                   # inside value pretraining: actor not saved yet
    assert critic_load(tmp_path).stdout.strip() == "/base"
    mark(tmp_path, "ckpt", 10)
    assert critic_load(tmp_path).returncode != 0                       # actor 10, critic 3: refuse
    mark(tmp_path, "ckpt_critic", 10)
    assert critic_load(tmp_path).stdout.strip() == f"{tmp_path}/ckpt_critic"


def test_prune_keeps_only_the_newest_critic(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for i in range(4):
        mark(run, "ckpt", i)
        mark(run, "ckpt_critic", i)
    (run / "ckpt" / "iter_0000004").mkdir()                            # being written
    (run / "ckpt" / "latest_checkpointed_iteration.txt").write_text("3")
    subprocess.run(["bash", "scripts/tools/prune_ckpts.sh", str(run)], cwd=ROOT, check=True, capture_output=True,
                   env={**os.environ, "LIVE": "1", "KEEP_ITERS": "1"})
    assert sorted(p.name for p in (run / "ckpt").glob("iter_*")) == ["iter_0000001", "iter_0000003", "iter_0000004"]
    assert sorted(p.name for p in (run / "ckpt_critic").glob("iter_*")) == ["iter_0000003"]


def load_acceptance():
    spec = importlib.util.spec_from_file_location("acc", ROOT / "scripts/checks/grpo_acceptance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ppo_log(*, cold_start: bool = False, actor_at_0: bool = False) -> str:
    critic = "[x] actor.py:1 - critic-step {r}: {{'train/critic-value_loss': 0.2{r}, 'train/critic-grad_norm': 1.5}}"
    actor = "[x] actor.py:1 - step {r}: {{'train/loss': 0.01, 'train/grad_norm': 0.3, 'train/step': {r}}}"
    rollouts = ([0, 1] if cold_start else []) + [0, 1, 2]
    lines = [critic.format(r=r) for r in rollouts] + [actor.format(r=r) for r in rollouts if r or actor_at_0]
    return "\n".join(lines)


@pytest.mark.parametrize(("kwargs", "ok"), [({}, True), ({"cold_start": True}, False), ({"actor_at_0": True}, False)])
def test_ppo_acceptance_log_checks(tmp_path: Path, monkeypatch, kwargs: dict, ok: bool) -> None:
    acc = load_acceptance()
    monkeypatch.setattr(acc, "check_batch", lambda run, batch: {"batch": batch})
    for sub in ("ckpt", "ckpt_critic"):
        mark(tmp_path, sub, 2)
    settings = {"CRITIC_WARMUP_STEPS": "1", "NUM_ROLLOUT": "3"}
    if ok:
        out = acc.accept_ppo(tmp_path, settings, ppo_log(**kwargs))
        assert out["critic_updates"][0]["train/critic-value_loss"] == 0.2 and sorted(out["actor_updates"]) == [1, 2]
    else:
        with pytest.raises(AssertionError):
            acc.accept_ppo(tmp_path, settings, ppo_log(**kwargs))
