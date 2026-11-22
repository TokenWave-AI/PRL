"""Accept a completed two-update smoke before releasing full training.

Run on a CPU compute node after the smoke job (a job dependency on it). Read retained
rollout dictionaries, actual verifier results, update logs and checkpoint marker;
write acceptance.json only after all checks pass. A failed dependency alone does
not stop successors on every scheduler, so the full launcher also requires this receipt.

Configuration: PRL_SMOKE_RUN points to the completed smoke directory;
PRL_ROOT is the source checkout. No model loading or cloud requests.
"""
from __future__ import annotations

import codecs
import hashlib
import io
import itertools
import json
import math
import os
import pickle
import re
import zipfile
from collections import defaultdict
from pathlib import Path

# prl.algo.llm_client.REFUSED (this script runs on the host without the repo on its path)
REFUSED = "flagged as potentially violating"

# The framework a receipt vouches for: the rollout and training path, the Miles flags (container_main.sh) and our Miles
# patches (all of patches/miles, see source_hashes). Not the transforms (transforms/*.py) or what they use (the judge's
# labeler and LLM client): every training job checks its transform on recorded trials before training
# (scripts/lib/transform_check.sh), so a new transform needs no smoke.
SOURCE_FILES = (
    "prl/rollout/harbor_training.py", "prl/rollout/training_samples.py",
    "prl/rollout/training_terminus.py", "prl/rollout/reroll_agent.py", "prl/rollout/harbor_agent.py",
    "prl/rollout/branching.py", "prl/rollout/replay.py", "prl/rollout/replay_agent.py",
    "prl/rollout/harbor_fixes.py", "prl/rollout/rollout_metrics.py", "prl/rollout/refill.py",
    "prl/algo/step_advantage.py", "prl/algo/added_rewards.py", "prl/algo/fused_logprob.py",
    "scripts/run.sh", "scripts/lib/container_main.sh",
    "prl/transforms/core.py", "prl/transforms/check.py", "prl/transforms/api.py",
    "prl/transforms/tokens.py", "prl/transforms/ask.py",
    "scripts/lib/transform_check.sh",
)
# A sane smoke's trainer and engine agree on the sampled tokens' log-probs: Qwen3.5-9B 0.0077-0.0095 mean absolute
# difference, KL 3-4e-4 (two smokes); Qwen3-8B KL 7e-4. An engine with stale or scrambled weights (with
# --check-weight-update-equal: any tensor the sync missed) is off by orders of magnitude.
MAX_LOGPROB_ABS_DIFF = 0.05
MAX_TRAIN_ROLLOUT_KL = 0.01
# The judge profile of a transform: the smoke's preflight records it (for the record; configs/arms/
# ota_judge.env)
JUDGE_KEYS = ("PRL_JUDGE_API_BASE", "PRL_JUDGE_MODEL", "PRL_JUDGE_FALLBACK_MODELS",
              "PRL_JUDGE_PROMPT", "PRL_JUDGE_EXTRA_BODY", "PRL_JUDGE_CONCURRENCY")


def source_hashes(root: Path) -> dict[str, str]:
    patches = tuple(sorted(str(p.relative_to(root)) for p in (root / "patches/miles").glob("*.patch")))
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES + patches}


class OpaqueArray:
    """A numpy array of the dump (rollout_routed_experts with R3, MoE models), not built: only its shape is kept, so the
    check needs no numpy and runs no code from the file. Stands in for numpy's _reconstruct, ndarray and dtype."""
    def __init__(self, *args):
        self.shape: tuple[int, ...] | None = None

    def __setstate__(self, state):  # ndarray: (version, shape, dtype, fortran, data); dtype: its own tuple
        if isinstance(state, tuple) and len(state) == 5 and isinstance(state[1], tuple):
            self.shape = state[1]


NUMPY_ARRAY = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype")}


class PlainDataUnpickler(pickle.Unpickler):
    """Miles debug rollout dumps contain plain dictionaries, not model tensors (numpy arrays only as OpaqueArray)."""
    def find_class(self, module: str, name: str):
        if (module, name) in NUMPY_ARRAY:
            return OpaqueArray
        if (module, name) == ("_codecs", "encode"):  # pickle protocol 2 (torch.save's) writes an array's bytes so
            return codecs.encode
        raise ValueError(f"Unexpected object in rollout dump: {module}.{name}")


def finite(line: str, metrics: tuple[str, ...]) -> dict[str, float]:
    values = {metric: float(re.search(rf"'{metric}': ([^,}}]+)", line)[1]) for metric in metrics}
    assert all(math.isfinite(value) for value in values.values()), values
    return values


def load_samples(run: Path, batch: int) -> list[dict]:
    with zipfile.ZipFile(run / f"traces/rollout_data/{batch}.pt") as archive:
        name, = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        return PlainDataUnpickler(io.BytesIO(archive.read(name))).load()["samples"]


def check_batch(run: Path, batch: int, staleness: int = 0, reject: bool = True) -> dict:
    """``staleness`` > 0: the async smoke (ASYNC=1), whose batch ``b`` may mix weight versions, each at most that many
    updates behind the trainer's (version b + 1: one per update, from 1); 0: one version per batch. ``reject``:
    PRL_REJECT_MISMATCH=1, so a trial with a TITO mismatch ran again and none is left; the async smoke runs
    without it (prl/rollout/harbor_training.py refuses the refill in async training) and its mismatched trials
    train, counted in ``accepted_mismatches``."""
    samples = load_samples(run, batch)
    groups = defaultdict(lambda: defaultdict(list))
    versions = set()
    routed = rerolls = mismatched = 0
    for sample in samples:
        meta = sample["metadata"]
        # R3 (MoE): one row of routed experts per token but the last, on every sample or on none
        if (experts := sample.get("rollout_routed_experts")) is not None:
            assert experts.shape and experts.shape[0] == len(sample["tokens"]) - 1, (experts.shape, len(sample["tokens"]))
            routed += 1
        assert sample["status"] != "aborted" and sample["reward"] is not None
        if reject:
            assert meta["tito_session_mismatch"] == [] and meta["prl_reject_mismatch"] is True
            assert meta["prl_mismatch_checks"]["rejected"] == []
            assert meta["prl_mismatch_checks"]["checked_leaves"] > 0
        else:
            assert meta["prl_reject_mismatch"] is False and "prl_mismatch_retries" not in meta
            mismatched += bool(meta.get("tito_session_mismatch"))
        assert any(sample["loss_mask"]) and len(sample["loss_mask"]) == sample["response_length"]
        if not (meta.get("prl_transform") or {}).get("tokens_changed"):  # rewritten trials train teacher tokens
            # Only tokens the model generated train: each trained position lies in a span of a model call (the
            # call's weight_versions), never in a tool output, user message or the prompt
            generated = [(span["abs_start"], span["abs_end"]) for call in sample["weight_versions"] for span in call]
            prompt = len(sample["tokens"]) - sample["response_length"]
            outside = [i for i, m in enumerate(sample["loss_mask"])
                       if m and not any(a <= prompt + i < b for a, b in generated)]
            assert not outside, (meta.get("trial_dir"), len(outside), outside[:5])
        versions.update(span["version"] for call in sample["weight_versions"] for span in call)
        if "prl_reroll" in meta:
            # A reroll (an added group member): its own trial directory and verifier reward, which is its group
            # member's; it shares the rollout_id of the trial it came from (check_rerolls matches the two)
            assert sample["tokens"] == meta["accumulated_token_ids"]
            trial = json.loads((Path(meta["trial_dir"]) / "result.json").read_text())
            own = trial["verifier_result"]["rewards"]["reward"]
            assert (sample.get("train_metadata") or {}).get("group_reward") == own, (meta["trial_dir"], own)
            rerolls += 1
            continue
        assert not reject or 0 <= meta["prl_mismatch_retries"] <= 2
        record = meta.get("prl_transform") or {}
        if record.get("tokens_changed"):  # a rewriting transform trains other tokens than the session produced (it
            prompt = len(sample["tokens"]) - sample["response_length"]  # checks them itself): the prompt still matches
            assert sample["tokens"][:prompt] == meta["accumulated_token_ids"][:prompt]
        else:
            assert sample["tokens"] == meta["accumulated_token_ids"]
        trial = json.loads((Path(meta["trial_dir"]) / "result.json").read_text())
        verified = next((e["reward"][0] for e in record.get("chain", []) if e.get("reward")), sample["reward"])
        assert verified == trial["verifier_result"]["rewards"]["reward"]  # before a transform changed it
        groups[sample["group_index"]][sample["rollout_id"]].append(sample)
    assert len(groups) == 8 and all(len(group) == 8 for group in groups.values())
    if staleness:
        assert all(batch + 1 - staleness <= int(v) <= batch + 1 for v in versions), (batch, versions)
    else:
        assert len(versions) == 1, versions
    assert routed in (0, len(samples)), f"routed experts on {routed} of {len(samples)} samples"
    rollouts = [leaves for group in groups.values() for leaves in group.values()]
    assert all(len({s["reward"] for s in leaves}) == 1 for leaves in rollouts)
    assert all(len({s["metadata"]["trial_dir"] for s in leaves}) == 1 for leaves in rollouts)
    return {"batch": batch, "groups": len(groups), "rollouts": len(rollouts), "leaves": len(samples),
            "weight_versions": sorted(versions), "accepted_mismatches": mismatched, "routed_expert_samples": routed,
            "reroll_samples": rerolls,
            "retried_rollouts": sum(leaves[0]["metadata"].get("prl_mismatch_retries", 0) > 0 for leaves in rollouts)}


def check_transform(run: Path, batches: tuple[int, ...], transform: str) -> dict:
    """A trajectory transform: every trained trial carries its record (prl_transform, the same on every leaf); its
    trained tokens, summed over its leaves, are the record's; no transform failed and at most 10% were skipped; it
    changed what trains (tokens removed or rewritten, steps weighted); the job's pre-run check passed. atlas: the
    judge's label files. rewrite: at least one trial rewritten, and the teacher's served model. Rerolls
    (``step.reroll()``): see check_rerolls."""
    trials: dict[tuple[int, str], dict] = {}
    rerolls: list[tuple[int, dict]] = []
    for batch in batches:
        for sample in load_samples(run, batch):
            if "prl_reroll" in sample["metadata"]:  # an added trial, without a transform record of its own
                rerolls.append((batch, sample))
                continue
            record = sample["metadata"]["prl_transform"]
            trial = trials.setdefault((batch, sample["metadata"]["trial_dir"]),
                                      {"record": record, "trained": 0, "weighted": False, "reward": sample.get("reward"),
                                       "anchors": (sample.get("train_metadata") or {}).get("branch_anchors") or []})
            assert trial["record"] == record, sample["metadata"]["trial_dir"]  # one record per trial, on every leaf
            trial["trained"] += sum(sample["loss_mask"])
            train_metadata = sample.get("train_metadata") or {}  # step or token level (token_weight_spans)
            trial["weighted"] |= bool(train_metadata.get("step_weights") or train_metadata.get("token_weight_spans"))
    assert trials
    for (_, trial_dir), trial in trials.items():
        record = trial["record"]
        assert [e["transform"] for e in record["chain"]] == transform.split(","), (trial_dir, record["chain"])
        assert all(e["status"] != "error" for e in record["chain"]), (trial_dir, record["chain"])
        assert trial["trained"] == record["trained_tokens_after"], (trial_dir, trial["trained"], record)
        assert record["tokens_changed"] or record["trained_tokens_after"] <= record["trained_tokens_before"], record
    skips = [e.get("reason") or "" for t in trials.values() for e in t["record"]["chain"] if e["status"] == "skipped"]
    skipped = len(skips)
    # The model provider's content filter refuses some prompts: the transform's business, not the framework's
    refused = sum(REFUSED in reason for reason in skips)
    assert skipped - refused <= 0.1 * len(trials), f"{skipped} of {len(trials)} trials skipped ({refused} refused)"
    before = sum(t["record"]["trained_tokens_before"] for t in trials.values())
    after = sum(t["record"]["trained_tokens_after"] for t in trials.values())
    assert after != before or any(t["weighted"] or t["record"]["tokens_changed"] or t["record"].get("reroll")
                                  for t in trials.values()), "the transform changed nothing"
    check = json.loads((run / "transform_check" / "summary.json").read_text())
    assert check["transform"] == transform and check["trials"], check
    result = {"transform": transform, "transform_trials": len(trials), "transform_skipped": skipped,
              "transform_refused": refused,
              "kept_token_share": round(after / before, 4), "pre_check_trials": len(check["trials"])}
    if "atlas" in transform.split(","):
        served = set()
        for (_, trial_dir), trial in trials.items():
            if any(e["transform"] == "atlas" and e["status"] == "ok" for e in trial["record"]["chain"]):
                served.add(str(json.loads((Path(trial_dir) / "atlas_labels.json").read_text()).get("served_model")))
        result["judge_served_models"] = sorted(served)
    if "rewrite" in transform.split(","):  # the smoke must reach the renderer, not only pass trials through
        rewritten = {d for (_, d), t in trials.items()
                     if any(e["transform"] == "rewrite" and e.get("rewritten_steps") for e in t["record"]["chain"])}
        assert rewritten, "no trial was rewritten"
        assert all(t["record"]["tokens_changed"] for (_, d), t in trials.items() if d in rewritten)
        result["rewritten_trials"] = len(rewritten)
        result["teacher_served_models"] = sorted({str(c.get("served_model")) for d in rewritten
                                                  for c in json.loads((Path(d) / "transform_calls.json").read_text())})
    result |= check_rerolls(trials, rerolls)
    if "reroll_smoke" in transform.split(","):
        assert result.get("reroll_ok_branches"), "no reroll got a reward"
    return result


def check_rerolls(trials: dict[tuple[int, str], dict], rerolls: list[tuple[int, dict]]) -> dict:
    """Each branch sample belongs to a trial of its batch whose record lists that branch (step, branch) as successful;
    it carries that outcome, the trial's reward, its own reward (group_reward) and one added_id per branch; its trained
    tokens, summed over its samples, are the outcome's; the replay compared every replayed turn. Every successful branch
    reached the batch. A trial with a subgroup or verified branch carries the step in branch_anchors; a branch with a
    hint (step.reroll(hint=...)) showed the policy all of it (reroll_replay.json hint_chars)."""
    requested = [(key, o) for key, t in trials.items() for o in (t["record"].get("reroll") or [])
                 if isinstance(o, dict)]
    if not requested and not rerolls:
        return {}
    trained: dict[tuple[int, str, int, int], int] = {}
    for batch, sample in rerolls:
        md, tm = sample["metadata"], sample.get("train_metadata") or {}
        key = (batch, md["prl_reroll_of"])
        assert key in trials, f"reroll of {key} without its trial in the batch"
        mine = md["prl_reroll"]
        assert mine in trials[key]["record"]["reroll"] and mine["status"] == "ok", (key, mine)
        assert tm.get("group_reward") == mine["reward"], (key, tm)
        assert tm.get("added_id") == f"reroll:{mine['step']}:{mine['branch']}", (key, tm)
        assert tm.get("branch") == {"anchor": mine["step"], "credit": mine["credit"]}, (key, tm)
        if mine["credit"] in ("subgroup", "verified"):
            assert mine["step"] in [a["step"] for a in trials[key]["anchors"]], (key, trials[key]["anchors"])
        assert trials[key]["reward"] is None or sample.get("reward") == trials[key]["reward"], key
        branch = (*key, mine["step"], mine["branch"])
        trained[branch] = trained.get(branch, 0) + sum(sample["loss_mask"])
    ok = [(key, o) for key, o in requested if o["status"] == "ok"]
    hinted = 0
    for key, outcome in ok:
        assert trained.get((*key, outcome["step"], outcome["branch"])) == outcome["trained_tokens"], (key, outcome)
        folder = Path(key[1]) / "reroll" / f"s{outcome['step']}_b{outcome['branch']}"
        request = json.loads((folder / "request.json").read_text())
        replay = outcome["replay"]
        assert replay and replay["turns"] == len(request["turns"]) == replay["same"] + len(replay["differ"]), outcome
        if request.get("hint"):
            assert replay.get("hint_chars") == len(request["hint"]), (outcome, request["hint"])
            hinted += 1
    failed = [o.get("reason", "") for _, o in requested if o["status"] == "failed"]
    return {"reroll_requested_trials": len({key for key, _ in requested}), "reroll_branches": len(requested),
            "reroll_ok_branches": len(ok), "reroll_subgroup_branches": sum(o["credit"] == "subgroup" for _, o in ok),
            "reroll_verified_branches": sum(o["credit"] == "verified" for _, o in ok), "reroll_hinted_branches": hinted,
            "reroll_failed": failed[:5], "reroll_rewards": [o["reward"] for _, o in ok],
            "reroll_replayed_turns_same": sum(o["replay"]["same"] for _, o in ok),
            "reroll_replayed_turns_differ": sum(len(o["replay"]["differ"]) for _, o in ok),
            "reroll_first_message_same": sum(bool(o["replay"]["first_message_same"]) for _, o in ok)}


def check_weight_sync(run: Path, sync: bool) -> list[dict]:
    """The engine's weights after each sync (Miles' per-parameter checksums, traces/events/main.jsonl): every update
    changes language-model weights and no vision-tower weight (the trainer has none). Synchronous smokes record
    one before the first rollout and one after update 0."""
    events = [json.loads(line) for line in (run / "traces/events/main.jsonl").read_text().splitlines()]
    events = sorted((e for e in events if e["type"] == "inference_engine_weight_checksum"), key=lambda e: e["rollout_id"])
    if sync:
        assert [e["rollout_id"] for e in events] == [-1, 0], [e["rollout_id"] for e in events]
    changes = []
    for before, after in itertools.pairwise(events):
        old, new = before["engine_checksums"], after["engine_checksums"]
        assert len(old) == len(new)
        changed = [k for engine, later in zip(old, new) for k in engine if engine[k] != later.get(k)]
        vision = [k for k in changed if k.split("/", 1)[-1].startswith("visual.")]
        assert changed and not vision, (after["rollout_id"], len(changed), vision[:5])
        changes.append({"rollout_id": after["rollout_id"], "changed_tensors": len(changed),
                        "tensors": sum(len(engine) for engine in new)})
    return changes


def accept_ppo(run: Path, settings: dict, log: str) -> dict:
    """Two jobs on one RUN_DIR (ota_train_smoke_ppo): rollout 0 trains only the critic, 1 is PPO and
    saves both models, the second job resumes them for rollout 2. preflight.json is the second job's."""
    assert settings["CRITIC_WARMUP_STEPS"] == "1" and settings["NUM_ROLLOUT"] == "3"
    lines = log.splitlines()
    critic, actor = {}, {}
    for step in (0, 1, 2):
        found = [line for line in lines if f"critic-step {step}: {{" in line]
        assert found, f"Missing critic update {step}"
        critic[step] = (len(found), finite(found[-1], ("train/critic-value_loss", "train/critic-grad_norm")))
        found = [line for line in lines if re.search(rf"(?<![-\w])step {step}: {{'train/", line)]
        if step == 0:
            assert not found, "rollout 0 must train only the critic (value pretraining)"
            continue
        assert found, f"Missing actor update {step}"
        actor[step] = (len(found), finite(found[-1], ("train/loss", "train/grad_norm")))
    # A second job that cold started instead of resuming would log rollouts 0 and 1 a second time.
    assert len({count for count, _ in critic.values()} | {count for count, _ in actor.values()}) == 1, (critic, actor)
    for sub in ("ckpt", "ckpt_critic"):
        assert (run / sub / "latest_checkpointed_iteration.txt").read_text().strip() == "2", sub
    loads = sorted(set(re.findall(r"successfully loaded checkpoint from (\S+) .*?at iteration (\d+)", log)))
    return {"algorithm": "ppo", "batches": [check_batch(run, batch) for batch in (0, 1, 2)],
            "critic_updates": {step: metrics for step, (_, metrics) in critic.items()},
            "actor_updates": {step: metrics for step, (_, metrics) in actor.items()}, "checkpoint_loads": loads}


def accept(run: Path, root: Path) -> dict:
    preflight = json.loads((run / "preflight.json").read_text())
    assert preflight["passed"] and preflight["source_hashes"] == source_hashes(root)
    reject = preflight["settings"]["ASYNC"] != "1"  # mismatch refill is synchronous only (harbor_training.py)
    assert preflight["settings"]["PRL_REJECT_MISMATCH"] == ("1" if reject else "0")
    log = (run / "train.log").read_text()
    assert "Missing/unknown training call tag" not in log
    assert "Task raised exception" not in log
    assert any(int(n) > 0 for n in re.findall(r"(\d+) auxiliary tokens masked", log))
    counts = {"rejected_attempts": log.count("mismatch_refill rejected"),
              "exhausted_rollouts": log.count("mismatch_refill exhausted")}
    settings = preflight["settings"]
    transform = settings.get("PRL_TRANSFORM") or "none"
    if settings.get("ADVANTAGE_ESTIMATOR", "grpo") == "ppo":
        transformed = check_transform(run, (0, 1, 2), transform) if transform != "none" else {"transform": "none"}
        return {"passed": True, "smoke_run": str(run), "source_hashes": preflight["source_hashes"],
                **accept_ppo(run, settings, log), **transformed, **counts}
    staleness = 0
    if settings["ASYNC"] == "1":  # Miles' --max-weight-staleness, as the preflight recorded the run's arguments
        extra = preflight["extra_args"]
        staleness = int(extra[extra.index("--max-weight-staleness") + 1])
        assert staleness > 0, staleness
    updates = {}
    for step in (0, 1):
        lines = [line for line in log.splitlines() if f"'train/step': {step}" in line]
        assert lines, f"Missing optimizer update {step}"
        updates[step] = values = finite(lines[-1], ("train/loss", "train/grad_norm", "train/kl_loss", "train/ppo_kl",
                                                    "train/pg_clipfrac", "train/train_rollout_logprob_abs_diff",
                                                    "train/train_rollout_kl"))
        # trainer and engine agree (the engine got the trainer's weights, under the same sampling settings)
        assert values["train/train_rollout_logprob_abs_diff"] < MAX_LOGPROB_ABS_DIFF, (step, values)
        assert values["train/train_rollout_kl"] < MAX_TRAIN_ROLLOUT_KL, (step, values)
        if not staleness:  # one optimizer step per rollout: the policy it starts from is the one that sampled
            assert values["train/ppo_kl"] == 0 and values["train/pg_clipfrac"] == 0, (step, values)
    assert abs(updates[0]["train/kl_loss"]) < 1e-6, updates[0]  # a fresh run starts at the reference policy
    assert (run / "ckpt/latest_checkpointed_iteration.txt").read_text().strip() == "1"
    synced = check_weight_sync(run, sync=not staleness)
    batches = [check_batch(run, batch, staleness, reject) for batch in (0, 1)]
    if staleness:  # the second batch may still hold rollouts of the first weights, but no update goes backwards
        assert max(map(int, batches[1]["weight_versions"])) >= max(map(int, batches[0]["weight_versions"]))
    else:
        assert batches[0]["weight_versions"] != batches[1]["weight_versions"]
    transformed = check_transform(run, (0, 1), transform) if transform != "none" else {"transform": "none"}
    # the algorithm, which find_smoke.py and the full launcher match (receipts without it are GRPO)
    return {"passed": True, "smoke_run": str(run), "source_hashes": preflight["source_hashes"], "batches": batches,
            "algorithm": settings.get("ADVANTAGE_ESTIMATOR", "grpo"), "async": staleness > 0, "updates": updates,
            "weight_sync": synced, **transformed, **counts}


if __name__ == "__main__":
    run_dir = Path(os.environ["PRL_SMOKE_RUN"])
    result = accept(run_dir, Path(os.environ["PRL_ROOT"]))
    output = run_dir / "acceptance.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(result), flush=True)
