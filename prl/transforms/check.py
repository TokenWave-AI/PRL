"""Pre-run check of the selected trajectory transform on recorded trials: ``python -m prl.transforms.check``.

The training launchers run it inside the Miles container before training starts (scripts/chain/grpo_*.sh), so a
transform that raises, returns an invalid result, cannot reach its model, hangs, or skips every trial stops the job
before any GPU time is spent; the user never runs a smoke for a new transform. The trials are real training
trajectories stored as text (tests/fixtures/transform_trials.json.gz, made by scripts/tools/make_transform_fixture.py:
successes and failures, short and long, one split by a summary) and tokenized with the run's tokenizer, so the check
works for any model. The transform runs exactly as in training (prl.transforms.core.run) but strict: the first
failure raises. Its records (transform.json, the transform's own files such as atlas_labels.json) and summary.json go
to the output directory.

Configuration (env): PRL_TRANSFORM(_ARGS) as for training (nothing to check without one);
    MODEL_HF_DIR                         the tokenizer (a Hugging Face model dir, or a tokenizer.json)
    PRL_TRANSFORM_CHECK_OUT   output directory (default $RUN_DIR/transform_check)
    PRL_TRANSFORM_FIXTURE     trials file (default tests/fixtures/transform_trials.json.gz)
    PRL_TRANSFORM_CHECK_TIMEOUT  seconds per trial (default: PRL_TRANSFORM_TIMEOUT, as in
                                         training, else 1800)
    EXTRA_ARGS                           --use-tis / --use-rollout-logprobs forbid rewritten tokens, as in training
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from prl.transforms import core

logger = logging.getLogger(__name__)

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "transform_trials.json.gz"


class CheckFailed(RuntimeError):
    pass


class Tokenizer:
    """The run's tokenizer: encode without adding special tokens, decode keeping them (as Miles' samples are)."""

    def __init__(self, path: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError:  # outside the container: the tokenizers library and the model's tokenizer.json
            from tokenizers import Tokenizer as Fast

            file = Path(path) / "tokenizer.json" if Path(path).is_dir() else Path(path)
            fast = Fast.from_file(str(file))
            self.encode = lambda text: fast.encode(text, add_special_tokens=False).ids
            self.decode = lambda ids: fast.decode(list(ids), skip_special_tokens=False)
        else:
            hf = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.encode = lambda text: hf.encode(text, add_special_tokens=False)
            self.decode = lambda ids: hf.decode(list(ids), skip_special_tokens=False)


def samples(trial: dict[str, Any], tokenizer: Any, trial_dir: str) -> list[Any]:
    """One recorded trial as training samples: each text segment of a response tokenized on its own, so the trained
    (model output) and untrained (tool output, template) spans keep their boundaries."""
    out = []
    for leaf in trial["samples"]:
        response, mask = [], []
        for segment in leaf["response"]:
            ids = tokenizer.encode(segment["text"])
            response += ids
            mask += [segment["trained"]] * len(ids)
        out.append(SimpleNamespace(
            tokens=tokenizer.encode(leaf["prompt"]) + response, response_length=len(response), loss_mask=mask,
            reward=trial["reward"], rollout_id=trial["rollout_id"], rollout_log_probs=[0.0] * len(response),
            train_metadata=None, metadata={"leaf": {"node_id": leaf["node_id"]}, "reward": trial["reward"],
                                           "trial_dir": trial_dir}))
    return out


async def check(tokenizer: Any, fixture: Path, out: Path, timeout: float, args: Any = None) -> dict[str, Any]:
    chain = core.configured()  # configuration errors raise here
    if not chain:
        return {"transform": "none"}
    spec = ",".join(item for item, _, _ in chain)
    trials = json.loads(gzip.decompress(fixture.read_bytes()))["trials"]

    async def one(trial: dict[str, Any]) -> dict[str, Any]:
        trial_dir = out / trial["name"]
        trial_dir.mkdir(parents=True, exist_ok=True)
        steps = sum(seg["trained"] for leaf in trial["samples"] for seg in leaf["response"])
        about = f"{trial['name']} (reward {trial['reward']}, {steps} steps, {len(trial['samples'])} leaves)"
        t0 = time.time()
        try:
            kept = await asyncio.wait_for(core.run(samples(trial, tokenizer, str(trial_dir)), tokenizer, args,
                                                   strict=True), timeout)
        except Exception as exc:
            raise CheckFailed(f"transform {spec} failed on recorded trial {about}: {type(exc).__name__}: {exc}") from exc
        record = kept[0].metadata["prl_transform"]
        return {"trial": trial["name"], "reward": trial["reward"], "steps": steps,
                "status": [e["status"] for e in record["chain"]], "seconds": round(time.time() - t0, 1),
                "trained_tokens": [record["trained_tokens_before"], record["trained_tokens_after"]]}

    results = await asyncio.gather(*(one(trial) for trial in trials))
    if all("skipped" in r["status"] for r in results):
        raise CheckFailed(f"transform {spec} skipped all {len(results)} recorded trials (see {out}/*/transform.json)")
    summary = {"transform": spec, "params": [params for _, _, params in chain], "fixture": str(fixture),
               "trials": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        chain = core.configured()
    except (ValueError, ImportError, SyntaxError) as exc:  # unknown name, params that do not fit, a broken file
        print(f"[transform check] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    if not chain:
        print("[transform check] no trajectory transform selected; nothing to check", flush=True)
        return 0
    out = Path(os.getenv("PRL_TRANSFORM_CHECK_OUT") or Path(os.environ["RUN_DIR"]) / "transform_check")
    out.mkdir(parents=True, exist_ok=True)
    extra = os.getenv("EXTRA_ARGS", "").split()
    args = SimpleNamespace(use_tis="--use-tis" in extra, use_rollout_logprobs="--use-rollout-logprobs" in extra)
    fixture = Path(os.getenv("PRL_TRANSFORM_FIXTURE") or FIXTURE)
    try:
        summary = asyncio.run(check(Tokenizer(os.environ["MODEL_HF_DIR"]), fixture, out,
                                    float(os.getenv("PRL_TRANSFORM_CHECK_TIMEOUT")
                                          or os.getenv("PRL_TRANSFORM_TIMEOUT") or 1800) or None, args))
    except CheckFailed as exc:
        logger.exception("[transform check] FAILED")
        print(f"[transform check] FAILED: {exc}\n  records: {out}", file=sys.stderr, flush=True)
        return 1
    for r in summary["trials"]:
        print(f"[transform check] {r['trial']}: reward {r['reward']}, {r['steps']} steps, {'/'.join(r['status'])}, "
              f"trained tokens {r['trained_tokens'][0]} -> {r['trained_tokens'][1]}, {r['seconds']} s", flush=True)
    print(f"[transform check] passed: {summary['transform']} on {len(summary['trials'])} recorded trials; {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
