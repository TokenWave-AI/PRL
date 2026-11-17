"""Validate the training configuration and real Miles compact-sample contracts.

Run inside the Miles container on a compute node before the GRPO smoke. This
uses real Sample/filter/reward-normalization implementations, without loading a
model or CUDA tensors. Fail before expensive rollout if wiring or grouping is
wrong. Source/config snapshots and the JSON result go under RUN_DIR.

Configuration: RUN_DIR, PROMPT_DATA, PRL_TEMPLATE_MANIFEST, HARBOR_TASKS_DIR,
E2B_API_URL, the model profile (MODEL, MODEL_HF_DIR, MODEL_MAX_CONTEXT from configs/models/),
and the exact arm variables validated below. Never records API keys.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import shlex
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from grpo_acceptance import JUDGE_KEYS, source_hashes
from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.filter_hub.common_filters import apply_aborted_filter
from miles.utils.types import Sample
from training_call_tags import check_wire_tags

from prl.rollout.harbor_training import generate
from prl.rollout.training_samples import pick_samples, postprocess_samples
from prl.sandbox.template_manifest import context_hash, context_inventory, resolve_template
from prl.transforms.core import configured


def main() -> None:
    # Exercise the installed dependency versions through HTTP serialization,
    # not just kwargs passed into LiteLLM. No model or cloud request is made.
    wire = asyncio.run(asyncio.wait_for(check_wire_tags(), timeout=60))
    # One node: 2 training GPUs + 2 rollout GPUs; or colocated on NNODES nodes (the 2-node MoE smoke,
    # configs/arms/ota_train_smoke_moe.env: training and rollout share all GPUs)
    if os.environ.get("COLOCATE") == "1":
        assert int(os.environ["NNODES"]) >= 1, os.environ["NNODES"]
        placement = {"COLOCATE": "1", "NNODES": os.environ["NNODES"]}
    else:
        placement = {"COLOCATE": "0", "NNODES": "1", "ACTOR_GPUS_PER_NODE": "2", "ROLLOUT_GPUS": "2"}
    # ASYNC=1: the async smoke (ota_train_smoke_async): Miles' fully async trainer on the split placement
    assert os.environ["ASYNC"] in ("0", "1"), os.environ["ASYNC"]
    if os.environ["ASYNC"] == "1":
        assert placement["COLOCATE"] == "0" and "--fully-async" in os.environ["EXTRA_ARGS"], os.environ["EXTRA_ARGS"]
    # A mismatched trial runs again only in synchronous training (prl/rollout/harbor_training.py raises
    # otherwise); the async smoke trains mismatched trials instead (configs/arms/ota_train_smoke_async.env)
    reject = "0" if os.environ["ASYNC"] == "1" else "1"
    expected = {
        "MODE": "train", "NUM_GPUS": "4", **placement, "ASYNC": os.environ["ASYNC"], "ROLLOUT_BATCH": "8",
        "N_SAMPLES": "8",
        "NUM_ROLLOUT": "2", "HARBOR_TERMINUS_2_ENABLE_SUMMARIZE": "true",
        "KL_LOSS_COEF": "0.001", "LR": "1e-6", "EVAL_DATA": "",
        # the trainer's log-probs are at temperature 1 over the full vocabulary (the train/rollout comparison)
        "TEMPERATURE": "1.0", "TOP_P": "1.0",
        "PRL_REJECT_MISMATCH": reject, "PRL_MISMATCH_MAX_RETRIES": "2",
    }
    # The PPO smoke (ota_train_smoke_ppo) runs as two jobs of one 3-rollout run: the first exits after 2,
    # the second resumes for the third. The receipt records the algorithm; the full launcher requires the same one.
    algorithm = os.environ.get("ADVANTAGE_ESTIMATOR", "grpo")
    if algorithm == "ppo":
        expected |= {"NUM_ROLLOUT": "3", "CRITIC_WARMUP_STEPS": "1"}
        early_exit = "--debug-exit-after-rollout 2" in os.environ["EXTRA_ARGS"]
        assert early_exit == (os.environ["PRL_PPO_SMOKE_ROLLOUTS"] == "2"), os.environ["PRL_PPO_SMOKE_ROLLOUTS"]
        expected |= {key: os.environ[key] for key in ("CRITIC_LR", "GAE_GAMMA", "GAE_LAMBDA", "NORMALIZE_ADVANTAGES",
                                                       "PRL_PPO_SMOKE_ROLLOUTS")}
    elif algorithm == "reinforce_plus_plus":  # ota_train_smoke_rpp: Miles requires whitened advantages
        expected |= {"NORMALIZE_ADVANTAGES": "1"}
    else:
        assert algorithm == "grpo", algorithm
    for key, value in expected.items():
        assert os.environ[key] == value, (key, os.environ.get(key), value)
    expected["ADVANTAGE_ESTIMATOR"] = algorithm
    # Trajectory transform (prl/transforms/core.py): recorded with its judge profile, for the record only. The
    # smoke validates the framework; the transform itself is checked on recorded trials by every job before training
    # (scripts/lib/transform_check.sh, run by the launcher after this preflight).
    chain = configured()  # unknown names, params that do not fit: raise
    transform = ",".join(name for name, _, _ in chain) or "none"
    expected |= {"PRL_TRANSFORM": transform,
                 "PRL_TRANSFORM_ARGS": os.environ.get("PRL_TRANSFORM_ARGS", "")}
    if chain:
        expected |= {key: os.environ[key] for key in JUDGE_KEYS if key in os.environ}
    # The contract simulations below send fake trials through the real generate adapter: no transform on them (the
    # acceptance checks it on the smoke's real trials). With atlas left on, the adapter labeled a fake trial whose state
    # has no tokenizer, and the smoke failed here.
    os.environ["PRL_TRANSFORM"] = "none"
    os.environ.pop("PRL_TRANSFORM_ARGS", None)
    # Context is per arm (65536 for Qwen3.5-9B, 40960 = Qwen3-8B's max positions); its four budgets must agree
    # and fit the model. The receipt records model and context; the full launcher requires the same.
    context = int(os.environ["MAX_CONTEXT"])
    assert os.environ["HARBOR_MAX_SEQ_LEN"] == os.environ["AGENT_MAX_INPUT_TOKENS"] == str(context)
    assert os.environ["MAX_RESPONSE"] == os.environ["AGENT_MAX_OUTPUT_TOKENS"] == str(context - 2048)
    assert context <= int(os.environ["MODEL_MAX_CONTEXT"]), (context, os.environ["MODEL_MAX_CONTEXT"])
    expected |= {key: os.environ[key] for key in ("MODEL", "MODEL_HF_DIR", "MAX_CONTEXT", "MAX_RESPONSE")}
    extra = shlex.split(os.environ["EXTRA_ARGS"])
    for flag, value in {
        "--rollout-num-gpus-per-engine": "2", "--num-gpus-per-node": "4", "--sglang-server-concurrency": "16",
        "--use-session-server": "v2", "--optimizer-offload-fraction": "1.0",
        "--custom-generate-function-path": "prl.rollout.harbor_training.generate",
        "--custom-agent-function-path": "prl.rollout.harbor_training.run",
        "--session-sample-picker-path": "prl.rollout.training_samples.pick_samples",
        "--session-sample-postprocessor-path": "prl.rollout.training_samples.postprocess_samples",
    }.items():
        assert extra[extra.index(flag) + 1] == value, flag
    assert "--calculate-per-token-loss" in extra
    options = json.loads(os.environ["HARBOR_ENV_KWARGS"])
    force = options["prl_force_summary_once"]  # every trial, or a fraction (transform smokes keep some whole)
    assert options["prl_max_concurrent_trials"] == 16 and (force is True or (type(force) is float and 0 < force < 1))
    rows = [json.loads(line) for line in Path(os.environ["PROMPT_DATA"]).read_text().splitlines()]
    tasks = [row["metadata"]["instance_id"] for row in rows]
    manifest_path = os.environ["PRL_TEMPLATE_MANIFEST"]
    assert len(tasks) == len(set(tasks)) == 8
    # the training snapshot's manifest (configs/arms/ota_train_smoke.env): the smoke's 8 tasks among its 798
    assert set(tasks) <= set(json.loads(Path(manifest_path).read_text())["tasks"])
    for task in tasks:
        resolve_template(manifest_path, Path(os.environ["HARBOR_TASKS_DIR"]) / task, os.environ["E2B_API_URL"])
    # Reuse CPU oracle/nop results (chain step task_check, written to <manifest dir>/contract: the smoke's tasks or the
    # whole snapshot) only for exactly the same templates and tasks.
    contract_dir = Path(manifest_path).parent / "contract"
    assert (contract_dir / "summary.json").exists(), (
        f"no oracle/nop result for this snapshot: CHAIN=task_check+smoke_grpo PRL_TASK_CHECK_OUTPUT={contract_dir} "
        "PRL_TASK_PROMPTS=$PROMPT_DATA (scripts/chain/task_smoke.sh)")
    inputs = json.loads((contract_dir / "inputs.json").read_text())
    assert inputs["manifest_sha256"] == hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    for task in tasks:
        assert inputs["task_hashes"][task] == context_hash(
            context_inventory(Path(os.environ["HARBOR_TASKS_DIR"]) / task)), task
    contract = [r for r in json.loads((contract_dir / "summary.json").read_text()) if r["task"] in tasks]
    assert len(contract) == 16 and {(r["task"], r["agent"]) for r in contract} == {
        (task, agent) for task in tasks for agent in ("oracle", "nop")
    }
    assert all(r["passed"] and r["exception"] is None
               and r["rewards"] == {"reward": 1.0 if r["agent"] == "oracle" else 0.0} for r in contract)

    # One summarized rollout has two leaves; it must not become two GRPO draws.
    samples = [Sample(index=i, rollout_id=i, prompt="same task", reward=float(i >= 4)) for i in range(8)]
    samples.insert(1, Sample(index=0, rollout_id=0, prompt="same task", reward=0.0))
    args = Namespace(advantage_estimator="grpo", grpo_std_normalization=True, n_samples_per_prompt=8)
    normalized = _normalize_rewards_by_rollout(args, samples, [s.reward for s in samples], [9])
    rewards = torch.tensor([0.0] * 4 + [1.0] * 4)
    expected_adv = ((rewards - rewards.mean()) / (rewards.std() + 1e-6)).tolist()
    assert torch.allclose(torch.tensor(normalized), torch.tensor([expected_adv[0], *expected_adv]))
    samples[0].status = Sample.Status.ABORTED
    assert not apply_aborted_filter(args, [[s] for s in samples]).keep
    assert pick_samples(samples, {"agent": {"prl_training_contract": True, "prl_aborted": True}}) == []

    # A summary leaf can own a real pre-summary action. Keep that action, not
    # the summary tokens; retain the post-summary branch as another segment.
    leaves = [
        Sample(tokens=[1, 2, 3, 4, 5, 6], response_length=4, loss_mask=[1, 1, 1, 1],
               metadata={"leaf": {"node_id": 1, "parent": 0, "path_node_ids": [0, 1]}}),
        Sample(tokens=[7, 8, 9, 10], response_length=2, loss_mask=[1, 1],
               metadata={"leaf": {"node_id": 2, "parent": None, "path_node_ids": [2]}}),
    ]
    metadata = {"agent": {"prl_training_contract": True, "prl_evaluation": False, "reward": 0.5},
                "tree": {"nodes": [
                    {"id": 0, "parent": None, "completion_span": [2, 4],
                     "turn_args": wire["requests"][1]},
                    {"id": 1, "parent": 0, "completion_span": [4, 6],
                     "turn_args": wire["requests"][2]},
                    {"id": 2, "parent": None, "completion_span": [2, 4],
                     "turn_args": wire["requests"][3]},
                ], "leaves": [{"node_id": 1, "path_node_ids": [0, 1]}, {"node_id": 2, "path_node_ids": [2]}]}}
    for sample in leaves:
        sample.metadata["accumulated_token_ids"] = sample.tokens
    kept = postprocess_samples(pick_samples(leaves, metadata), metadata)
    assert len(kept) == 2 and sum(sum(s.loss_mask) for s in kept) == 4
    assert all(s.reward == 0.5 for s in kept)
    # Use real Miles inputs/samples while replacing only generation. This tests
    # that one bad draw is replaced, siblings remain one draw, and exhaustion
    # reaches the actual whole-group abort filter with no artificial reward.
    clean = {"tito_session_mismatch": [],
             "prl_mismatch_checks": {"checked_leaves": 1, "rejected": []}}
    bad = {"tito_session_mismatch": [{"type": "assistant_text"}],
           "prl_mismatch_checks": {"checked_leaves": 1, "rejected": [{"reason": "mismatch"}]}}
    refill_input = GenerateFnInput(state=SimpleNamespace(args=Namespace(use_session_server="v2")),
                                  sample=Sample(index=3, rollout_id=3, prompt="same task"),
                                  sampling_params={"sampling_seed": 42}, evaluation=False)
    if reject == "1":  # the async smoke has no mismatch refill to simulate
        calls = []
        async def fake_generation(request):
            calls.append(request)
            return GenerateFnOutput(samples=[Sample(status=Sample.Status.COMPLETED, reward=0,
                                                    metadata=bad if len(calls) == 1 else clean)])
        with patch("miles.rollout.generate_hub.agentic_tool_call.generate", fake_generation):
            retried = asyncio.run(generate(refill_input))
        assert len(calls) == 2 and calls[0].sample is not calls[1].sample
        assert calls[0].sampling_params != calls[1].sampling_params
        assert retried.samples[0].reward == 0 and retried.samples[0].rollout_id == 3
        assert retried.samples[0].metadata["prl_mismatch_retries"] == 1
        async def always_bad(request):
            return GenerateFnOutput(samples=[Sample(status=Sample.Status.COMPLETED, reward=1, metadata=bad)])
        with patch("miles.rollout.generate_hub.agentic_tool_call.generate", always_bad):
            exhausted = asyncio.run(generate(refill_input))
        assert exhausted.samples[0].reward is None
        assert not apply_aborted_filter(args, [exhausted.samples]).keep
    # An aborted trial (infrastructure failure) runs again on its own before Miles would refill its whole group
    # (harbor_training._retry_aborted, PRL_ABORT_MAX_RETRIES default 2)
    if int(os.getenv("PRL_ABORT_MAX_RETRIES", "2")) > 0:
        abort_calls = []
        async def abort_once(request):
            abort_calls.append(request)
            if len(abort_calls) == 1:
                return GenerateFnOutput(samples=[Sample(status=Sample.Status.ABORTED, reward=None)])
            return GenerateFnOutput(samples=[Sample(status=Sample.Status.COMPLETED, reward=0, metadata=dict(clean))])
        with patch("miles.rollout.generate_hub.agentic_tool_call.generate", abort_once):
            recovered = asyncio.run(generate(refill_input))
        assert len(abort_calls) == 2 and abort_calls[0].sample is not abort_calls[1].sample
        assert recovered.samples[0].metadata["prl_abort_retries"] == 1
        assert apply_aborted_filter(args, [recovered.samples]).keep
    out = Path(os.environ["RUN_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    result = {"passed": True, "tasks": tasks, "settings": expected, "extra_args": extra,
              "source_hashes": source_hashes(Path(os.environ["PRL_ROOT"])),
              "wire_probe": wire,
              "versions": {name: importlib.metadata.version(name) for name in ("harbor", "litellm", "openai")},
              "checks": ["8 ready context hashes", "CPU oracle/nop receipt matches all task files and manifest",
                         "rollout reward normalization", "abort group filter",
                         "real Miles picker + shared masks + summary masking",
                         "real Harbor/LiteLLM/OpenAI wire tags before/during/after summary",
                         "real Miles trial refill and exhausted group abort" if reject == "1" else
                         "async: mismatched trials train (no refill)"]}
    (out / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
