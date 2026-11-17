"""Run official oracle/nop Harbor trials against prebuilt task templates.

Called by scripts/chain/task_smoke.sh (chain step task_check), before model evaluation. Each manifest task gets a fresh sandbox per agent; require oracle=1,
nop=0 and no infrastructure exceptions. Preserve complete Harbor trials and JSON.
No model is loaded. No task, oracle, or verifier files are modified.

Configuration:
    E2B_API_KEY_FILE / E2B_API_URL / E2B_DOMAIN: private key and explicit endpoint.
    PRL_TEMPLATE_MANIFEST: ready template manifest with context hashes.
    PRL_TASK_ROOT: Harbor task root.
    PRL_TASK_PROMPTS: optional prompt JSONL; only its tasks (metadata.instance_id) are checked, e.g. the smoke's 8
        (the smoke preflight reuses this result for them: <manifest dir>/contract).
    PRL_TASK_CHECK_OUTPUT: new directory for trials and summary.json.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from prl.sandbox.template_manifest import context_hash, context_inventory, resolve_template


async def main() -> None:
    key_file = Path(os.environ["E2B_API_KEY_FILE"])
    if key_file.stat().st_mode & 0o077:
        raise ValueError("Key file must be private")
    os.environ["E2B_API_KEY"] = key_file.read_text().strip()
    os.environ.pop("E2B_SANDBOX_URL", None)
    manifest_path = os.environ["PRL_TEMPLATE_MANIFEST"]
    manifest = json.loads(Path(manifest_path).read_text())
    output = Path(os.environ["PRL_TASK_CHECK_OUTPUT"])
    output.mkdir(parents=True, exist_ok=False)
    tasks = manifest["tasks"]
    if prompts := os.environ.get("PRL_TASK_PROMPTS"):
        tasks = [json.loads(line)["metadata"]["instance_id"] for line in Path(prompts).read_text().splitlines()]
        assert set(tasks) <= set(manifest["tasks"]), sorted(set(tasks) - set(manifest["tasks"]))
    # Bind the result to the full tasks, not just Docker contexts: changing an
    # oracle or verifier must invalidate the CPU gate before GPU training.
    inputs = {
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "task_hashes": {
            task: context_hash(context_inventory(Path(os.environ["PRL_TASK_ROOT"]) / task))
            for task in tasks
        },
    }
    (output / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    rows = []
    # Sequential validation keeps cloud resource use bounded and failures obvious.
    for task in tasks:
        task_path = Path(os.environ["PRL_TASK_ROOT"]) / task
        template = resolve_template(manifest_path, task_path, os.environ["E2B_API_URL"])
        for agent, expected in (("oracle", 1.0), ("nop", 0.0)):
            config = TrialConfig(
                task=TaskConfig(path=task_path), agent=AgentConfig(name=agent),
                environment=EnvironmentConfig(
                    type="e2b", override_cpus=2, override_memory_mb=2048,
                    kwargs={"prebuilt_template_id": template, "sandbox_timeout_secs": 1800, "renew_interval": 300},
                ),
                trials_dir=output / "trials", timeout_multiplier=1.0,
            )
            trial = await Trial.create(config)
            result = await trial.run()
            reward = result.verifier_result.rewards if result.verifier_result else None
            row = {"task": task, "agent": agent, "expected": expected, "rewards": reward,
                   "trial_dir": str(trial.paths.trial_dir),
                   "exception": result.exception_info.model_dump(mode="json") if result.exception_info else None}
            row["passed"] = row["exception"] is None and reward == {"reward": expected}
            rows.append(row)
            (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            if not row["passed"]:
                raise RuntimeError(f"Task contract failed: {task}/{agent}; inspect saved trial logs")


if __name__ == "__main__":
    asyncio.run(main())
