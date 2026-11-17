"""scripts/checks/find_smoke.py: a training job uses the newest smoke whose receipt passed for these sources and
algorithm and whose preflight ran its model at its context and schedule (what scripts/chain/grpo_full.sh checks
again)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "checks"))
os.environ.setdefault("PRL_ROOT", str(ROOT))
from find_smoke import find
from grpo_acceptance import source_hashes


def _smoke(ext: Path, name: str, mtime: int, *, passed=True, hashes=None, algo="grpo", model="qwen3-8B",
           context="40960", schedule=None) -> None:
    run = ext / "runs" / name
    run.mkdir(parents=True)
    (run / "acceptance.json").write_text(json.dumps(
        {"passed": passed, "source_hashes": source_hashes(ROOT) if hashes is None else hashes, "algorithm": algo}))
    settings = {"MODEL": model, "MAX_CONTEXT": context, "ADVANTAGE_ESTIMATOR": algo} | ({"ASYNC": schedule} if schedule else {})
    (run / "preflight.json").write_text(json.dumps({"settings": settings}))
    os.utime(run / "acceptance.json", (mtime, mtime))


def test_newest_matching_smoke(tmp_path) -> None:
    _smoke(tmp_path, "grpo_ota_train_smoke_40k_qwen3-8B_1", 100)
    _smoke(tmp_path, "grpo_ota_train_smoke_40k_qwen3-8B_2", 200)
    _smoke(tmp_path, "grpo_ota_train_smoke_40k_qwen3-8B_3", 300, passed=False)
    _smoke(tmp_path, "grpo_ota_train_smoke_40k_qwen3-8B_4", 400, hashes={"scripts/run.sh": "old"})
    _smoke(tmp_path, "grpo_ota_train_smoke_5", 500, model="qwen3.5-9B", context="65536")
    _smoke(tmp_path, "grpo_ota_train_smoke_40k_qwen3.5-9B_6", 600, model="qwen3.5-9B")
    _smoke(tmp_path, "ppo_ota_train_smoke_x", 700, algo="ppo")
    (tmp_path / "runs" / "grpo_ota_train_smoke_7").mkdir()  # a smoke without a receipt
    found = find(ROOT, tmp_path, "grpo", "qwen3-8B", "40960")
    assert found == str(tmp_path / "runs" / "grpo_ota_train_smoke_40k_qwen3-8B_2")
    assert find(ROOT, tmp_path, "grpo", "qwen3.5-9B", "65536").endswith("grpo_ota_train_smoke_5")
    assert find(ROOT, tmp_path, "ppo", "qwen3-8B", "40960").endswith("ppo_ota_train_smoke_x")
    assert find(ROOT, tmp_path, "grpo", "qwen3-8B", "65536") == ""


def test_an_async_smoke_vouches_only_for_async_jobs(tmp_path) -> None:
    _smoke(tmp_path, "grpo_ota_train_smoke_1", 100)  # receipts before ASYNC was recorded: synchronous
    _smoke(tmp_path, "grpo_ota_train_smoke_2", 200, schedule="1")
    assert find(ROOT, tmp_path, "grpo", "qwen3-8B", "40960").endswith("_1")
    assert find(ROOT, tmp_path, "grpo", "qwen3-8B", "40960", "1").endswith("_2")
