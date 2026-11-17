"""Relay of the full GRPO run across 24 h jobs: scripts/lib/relay.sh, live checkpoint pruning
(scripts/tools/prune_ckpts.sh) and the full-run arm configs/arms/ota_train_full.env. Bash only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def bash(script: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], check=False, capture_output=True, text=True,
                          env={**os.environ, **env})


def make_ckpt(run: Path, iters: list[int], latest: int | None, cursors: list[int] | None = None) -> None:
    """Checkpoints, the tracker and the dataset cursors Miles saves with them (by default one per checkpoint)."""
    ckpt = run / "ckpt"
    ckpt.mkdir(parents=True)
    (ckpt / "rollout").mkdir()
    for i in iters:
        (ckpt / f"iter_{i:07d}").mkdir()
    for i in iters if cursors is None else cursors:
        (ckpt / "rollout" / f"global_dataset_state_dict_{i}.pt").write_text(str(i))
    if latest is not None:
        (ckpt / "latest_checkpointed_iteration.txt").write_text(f"{latest}\n")


def begin(run: Path, job: str, **env: str) -> subprocess.CompletedProcess:
    return bash(f'source {ROOT}/scripts/lib/relay.sh; relay_begin "{run}" 34', JOB_ID=job, **env)


def starts(run: Path) -> list[list[str]]:
    return [line.split("\t")[:2] for line in (run / "relay.tsv").read_text().splitlines()]


def test_first_job_trains_and_records_its_start(tmp_path):
    run = tmp_path / "run"
    assert begin(run, "11").returncode == 0
    assert starts(run) == [["11", "-1"]]


def test_successor_resumes_after_progress(tmp_path):
    run = tmp_path / "run"
    assert begin(run, "11").returncode == 0
    make_ckpt(run, [12], latest=12)
    assert begin(run, "12").returncode == 0
    assert starts(run) == [["11", "-1"], ["12", "12"]]


def test_chain_stops_without_progress_until_reset(tmp_path):
    run = tmp_path / "run"
    make_ckpt(run, [5], latest=5)
    assert begin(run, "11").returncode == 0
    assert begin(run, "12").returncode == 1  # predecessor saved nothing new
    assert begin(run, "13").returncode == 1  # the rest of the chain stops too
    assert begin(run, "14", PRL_RELAY_RESET="1").returncode == 0  # manual resubmission after a fix


def test_first_job_failing_before_any_checkpoint_stops_successor(tmp_path):
    run = tmp_path / "run"
    assert begin(run, "11").returncode == 0
    assert begin(run, "12").returncode == 1


def test_complete_run_exits_without_recording(tmp_path):
    run = tmp_path / "run"
    make_ckpt(run, [33], latest=33)
    result = begin(run, "15")
    assert result.returncode == 3 and "complete" in result.stdout
    assert not (run / "relay.tsv").exists()


def remaining(run: Path) -> list[str]:
    return sorted(p.name for p in (run / "ckpt").glob("iter_*"))


def test_live_prune_never_touches_the_checkpoint_being_written(tmp_path):
    run = tmp_path / "run"
    make_ckpt(run, [3, 4, 5, 6], latest=5)  # iter_6 is still being saved
    assert bash(f'bash {ROOT}/scripts/tools/prune_ckpts.sh "{run}"', LIVE="1", KEEP_ITERS="3").returncode == 0
    assert remaining(run) == ["iter_0000003", "iter_0000005", "iter_0000006"]


def test_start_prune_drops_partial_checkpoint_of_killed_job(tmp_path):
    run = tmp_path / "run"
    make_ckpt(run, [3, 4, 5, 6], latest=5)  # iter_6 was cut off by the job's time limit
    assert bash(f'bash {ROOT}/scripts/tools/prune_ckpts.sh "{run}"', KEEP_ITERS="3").returncode == 0
    assert remaining(run) == ["iter_0000003", "iter_0000005"]


def arm_env(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    script = (f"set -a; source {ROOT}/configs/arms/common_ota_single_node.env; "
              f"source {ROOT}/configs/arms/ota_train_full.env; set +a; "
              f"{sys.executable} -c 'import json, os; print(json.dumps(dict(os.environ)))'")
    base = {"PRL_ROOT": str(ROOT), "PRL_EXT": str(tmp_path), "PRL_SCRATCH": str(tmp_path), "JOB_ID": "77",
            "USER": "u", "PRL_REJECT_MISMATCH": "", "PRL_MISMATCH_MAX_RETRIES": ""}
    return bash(script, **{**base, **env})


def test_full_arm_settings(tmp_path):
    result = arm_env(tmp_path, PRL_RUN_ID="r1")
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)
    assert env["PRL_REJECT_MISMATCH"] == "0"
    assert env["EVAL_DATA"] == "" and env["TBDEV_DATA"].endswith("ot_tbdev/tbdev.jsonl")
    assert env["SAVE_INTERVAL"] == "1" and env["PRL_KEEP_ITERS"] == "19"
    assert env["RUN_DIR"] == f"{tmp_path}/runs/grpo_ota_train_full_r1"
    assert env["HARBOR_TRIALS_DIR"] == f"{env['RUN_DIR']}/harbor_trials/job_77"
    extra = env["EXTRA_ARGS"].split()
    assert extra[extra.index("--sglang-server-concurrency") + 1] == "64"
    assert json.loads(env["HARBOR_ENV_KWARGS"])["prl_max_concurrent_trials"] == 64
    assert extra[extra.index("--use-session-server") + 1] == "v2"


def test_full_arm_requires_shared_run_id(tmp_path):
    result = arm_env(tmp_path)
    assert result.returncode != 0 and "PRL_RUN_ID" in result.stderr


def test_a_checkpoint_without_its_dataset_cursor_stops_until_told_to_use_the_previous(tmp_path):
    run = tmp_path / "run"
    make_ckpt(run, [12], latest=12, cursors=[11])  # killed between the weights and the cursor
    result = begin(run, "13")
    assert result.returncode == 1 and "no dataset cursor" in result.stderr
    assert begin(run, "14", PRL_RESUME_DATA_FROM_PREVIOUS="1").returncode == 0
    cursor = run / "ckpt/rollout/global_dataset_state_dict_12.pt"
    assert cursor.read_text() == "11"


def claim(run: Path, job: str) -> subprocess.CompletedProcess:
    # the claiming shell lives 1 s; its heartbeat then removes the claim within 10 s
    return bash(f'source {ROOT}/scripts/lib/relay.sh; relay_claim "{run}" && sleep 1', JOB_ID=job)


def test_a_run_in_use_by_another_job_is_refused(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "active_job").write_text("1234567\n")  # fresh: another job is training this run
    result = bash(f'source {ROOT}/scripts/lib/relay.sh; seq() {{ echo 1; echo 18; }}; sleep() {{ :; }}; '
                  f'relay_claim "{run}"', JOB_ID="1234568")
    assert result.returncode == 1 and "in use by job 1234567" in result.stderr


def test_a_stale_claim_is_taken_over_and_released(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "active_job").write_text("old\n")
    os.utime(run / "active_job", (0, 0))  # its job was killed long ago
    assert claim(run, "12").returncode == 0
    for _ in range(30):
        if not (run / "active_job").exists():
            break
        import time
        time.sleep(0.5)
    assert not (run / "active_job").exists()
