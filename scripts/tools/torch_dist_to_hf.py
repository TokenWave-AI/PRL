"""Export a Miles/Megatron torch_dist checkpoint to a complete, verified HF directory for SGLang evaluation.

Pipeline position: between training (`$RUN_DIR/ckpt/iter_N`, written by Miles `--save`) and an eval arm whose
`--hf-checkpoint` is this output. `scripts/chain/ckpt_to_hf.sh` runs it on a CPU node inside the Miles container;
the eval launcher then gets the output with `qsub -v MODEL_HF_DIR=<out>` (or `QWEN35_9B=<out>` for the Qwen3.5-9B
launchers): the model profile (configs/models/) passes it to `--hf-checkpoint`.

Why not Miles' `tools/convert_torch_dist_to_hf.py` as is:
  - it reads the Megatron args from `<ckpt>/common.pt`; our checkpoints (the base `<model>_torch_dist/release`
    and every `iter_*` of a run) have no `common.pt` (the common state is inside the DCP as `common_state/...`),
    so it stops before converting anything;
  - it writes only what the checkpoint holds. Qwen3.5-9B is published as `Qwen3_5ForConditionalGeneration`: besides
    427 language-model tensors the HF directory has a vision tower (333) and an MTP head (15) that our Megatron
    model does not contain (text only, `mtp_num_layers=None`). During training SGLang keeps those from the base
    model and receives only language-model weights, so they are copied here unchanged to rebuild that same model.
    Dense text models (Qwen3-8B) have nothing to copy.

Reused unchanged from upstream: the DCP reader/planner of that tool (model tensors only, optimizer skipped), its
layer-name expansion, and Miles' `convert_to_hf` (the mapping the live weight sync to SGLang uses, chosen by the
same model name upstream derives from the HF config class: qwen3_5config -> qwen3_5, qwen3config -> qwen2). The
Megatron args the mappings and padding removal read (hidden_size, num_attention_heads, num_query_groups,
kv_channels, num_layers, vocab_size) are taken from the base HF config (its text_config if it has one); for
Qwen3.5-9B they match a training run's arg dump (4096 / 16 / 4 / 256 / 32 / 248320, no vocab padding).

Checks (a failure exits non-zero, and the output directory is not created):
  - exported names == base HF names minus the copied ones, which may only be vision (`model.visual.`) or MTP
    (`mtp.`) tensors; shapes equal;
  - each tensor is compared with the base at the checkpoint's precision (the base cast to the checkpoint dtype:
    24 linear_attn.norm.weight are fp32 in the base HF but bf16 in Megatron), then written in the base dtype
    (casts are counted per dtype pair);
  - `--expect-identical` (round trip of the base torch_dist): every exported tensor is bitwise equal to the base
    at that precision; otherwise at least one tensor must differ, and a per-tensor relative-change summary is
    recorded.
The export is written to `<out>.partial` and renamed at the end; `<out>/export_verified.json` records the source,
base, counts and checks. Eval launchers require that file. `--check-only` runs the checks without writing.

Usage (inside the Miles container; PRL_MILES_DIR = the build tree that trained the checkpoint):
  python3 scripts/tools/torch_dist_to_hf.py --input-dir RUN_DIR/ckpt/iter_0000019 --origin-hf-dir $MODEL_HF_DIR \
      --output-dir $PRL_EXT/ckpt/hf/<run>_iter_0000019
  python3 scripts/tools/torch_dist_to_hf.py --input-dir $MODEL_TORCH_DIST/release \
      --origin-hf-dir $MODEL_HF_DIR --output-dir /unused --expect-identical --check-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from argparse import Namespace

import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.environ["PRL_MILES_DIR"], "tools"))
# Importing the upstream tool also installs its megatron-tolerant unpickler for the DCP metadata.
import convert_torch_dist_to_hf as upstream
from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf
from miles.utils.hf_utils.config import load_hf_config

SHARD_BYTES = 5 * 1024**3
# Base tensors the Megatron text model does not hold (Qwen3.5: vision tower, MTP head); anything else missing fails.
COPYABLE_PREFIXES = ("model.visual.", "mtp.")


def megatron_args(origin_hf_dir: str) -> Namespace:
    with open(os.path.join(origin_hf_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)  # Qwen3.5 nests the text model; Qwen3 is flat
    return Namespace(
        hidden_size=tc["hidden_size"],
        num_attention_heads=tc["num_attention_heads"],
        num_query_groups=tc["num_key_value_heads"],
        kv_channels=tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"],
        num_layers=tc["num_hidden_layers"],
        vocab_size=tc["vocab_size"],
        num_experts=None,
        sglang_enable_ep_moe=False,  # upstream save_tensors sets this before convert_to_hf
    )


def load_model_tensors(input_dir: str) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=upstream.WrappedStorageReader(input_dir),
        planner=upstream.EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}


class BaseModel:
    """Lazy (mmap) access to the base HF shards, one tensor at a time."""

    def __init__(self, hf_dir: str):
        self.dir = hf_dir
        with open(os.path.join(hf_dir, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._files = {}

    def get(self, name: str) -> torch.Tensor:
        f = self.weight_map[name]
        if f not in self._files:
            self._files[f] = safe_open(os.path.join(self.dir, f), framework="pt")
        return self._files[f].get_tensor(name)


class ShardWriter:
    def __init__(self, out_dir: str | None):
        self.out_dir, self.buf, self.size, self.files, self.weight_map, self.total = out_dir, {}, 0, [], {}, 0

    def add(self, name: str, t: torch.Tensor) -> None:
        n = t.numel() * t.element_size()
        if self.buf and self.size + n > SHARD_BYTES:
            self.flush()
        self.buf[name] = t.contiguous()
        self.size += n
        self.total += n

    def flush(self) -> None:
        if not self.buf:
            return
        tmp = f"shard-{len(self.files):05d}.tmp"
        if self.out_dir:
            safetensors.torch.save_file(self.buf, os.path.join(self.out_dir, tmp), metadata={"format": "pt"})
        self.files.append((tmp, list(self.buf)))
        self.buf, self.size = {}, 0

    def finish(self) -> None:
        self.flush()
        n = len(self.files)
        for i, (tmp, names) in enumerate(self.files):
            final = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            if self.out_dir:
                os.rename(os.path.join(self.out_dir, tmp), os.path.join(self.out_dir, final))
            self.weight_map.update({k: final for k in names})
        if self.out_dir:
            index = {"metadata": {"total_size": self.total}, "weight_map": dict(sorted(self.weight_map.items()))}
            with open(os.path.join(self.out_dir, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input-dir", required=True, help="torch_dist checkpoint dir (iter_* or release)")
    p.add_argument("--origin-hf-dir", required=True, help="base HF model the run started from")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--expect-identical", action="store_true", help="require bitwise equality with the base")
    p.add_argument("--check-only", action="store_true", help="convert and check, write nothing")
    a = p.parse_args()
    if not a.check_only and os.path.exists(a.output_dir):
        raise SystemExit(f"{a.output_dir} exists; refusing to overwrite")

    t0 = time.time()
    margs = megatron_args(a.origin_hf_dir)
    model_name = type(load_hf_config(a.origin_hf_dir)).__name__.lower()  # as upstream main() derives it
    base = BaseModel(a.origin_hf_dir)
    tensors = load_model_tensors(a.input_dir)
    print(f"[export] loaded {len(tensors)} model tensors from {a.input_dir} in {time.time() - t0:.0f}s", flush=True)

    partial = None if a.check_only else a.output_dir.rstrip("/") + ".partial"
    if partial:
        shutil.rmtree(partial, ignore_errors=True)
        os.makedirs(partial)
    writer = ShardWriter(partial)
    exported: set[str] = set()
    casts: dict[str, int] = {}
    identical, rel = 0, {}
    for mname in sorted(tensors):
        param = tensors.pop(mname)
        for lname, lparam in upstream.get_layer_param(margs, f"module.module.{mname}", param):
            for hname, h in convert_to_hf(margs, model_name, lname, lparam):
                if hname not in base.weight_map:
                    raise SystemExit(f"exported name not in base HF model: {hname} (from {lname})")
                if hname in exported:
                    raise SystemExit(f"exported twice: {hname}")
                ref = base.get(hname)
                if tuple(h.shape) != tuple(ref.shape):
                    raise SystemExit(f"shape mismatch {hname}: {tuple(h.shape)} vs base {tuple(ref.shape)}")
                # Compare at the checkpoint's precision: the base HF stores the 24 linear_attn.norm.weight in fp32,
                # the HF -> torch_dist conversion rounded them to bf16, so a bitwise match to the fp32 original is
                # impossible (round trip of the base: relative L2 1.4e-3). SGLang casts them to bf16
                # at load anyway (RMSNormGated is built with the bf16 default dtype), so the served model is the same.
                ref_c = ref.to(h.dtype)
                if torch.equal(h, ref_c):
                    identical += 1
                else:
                    d = (h.float() - ref_c.float()).norm() / ref_c.float().norm().clamp_min(1e-12)
                    rel[hname] = float(d)
                    if a.expect_identical:
                        raise SystemExit(f"--expect-identical: {hname} differs (relative L2 {float(d):.3e})")
                if h.dtype != ref.dtype:  # written in the base dtype, so config.json and the index stay valid
                    key = f"{h.dtype}->{ref.dtype}".replace("torch.", "")
                    casts[key] = casts.get(key, 0) + 1
                    h = h.to(ref.dtype)
                writer.add(hname, h)
                exported.add(hname)
        del param

    copied = sorted(set(base.weight_map) - exported)
    prefixes = sorted({".".join(k.split(".")[:2]) for k in copied})
    lm_missing = [k for k in copied if not k.startswith(COPYABLE_PREFIXES)]
    if lm_missing:
        raise SystemExit(f"base language-model tensors missing from the export: {lm_missing[:10]}")
    if not a.expect_identical and not rel:
        raise SystemExit("every exported tensor equals the base: is this really a trained checkpoint?")
    for k in copied:  # Qwen3.5 vision tower + MTP head: not in the Megatron model, kept from the base as SGLang does
        writer.add(k, base.get(k))
    writer.finish()

    vals = sorted(rel.values())
    report = {
        "input_dir": os.path.abspath(a.input_dir),
        "origin_hf_dir": os.path.abspath(a.origin_hf_dir),
        "miles_dir": os.environ["PRL_MILES_DIR"],
        "model_name": model_name,
        "megatron_args": vars(margs),
        "exported": len(exported),
        "copied_from_base": len(copied),
        "copied_prefixes": prefixes,
        "dtype_casts": casts,
        "identical_to_base": identical,
        "changed": len(rel),
        "changed_rel_l2": (
            {"median": statistics.median(vals), "max": vals[-1], "top5": sorted(rel.items(), key=lambda x: -x[1])[:5]}
            if vals
            else None
        ),
        "expect_identical": a.expect_identical,
        "seconds": round(time.time() - t0),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("[export] " + json.dumps(report, indent=1), flush=True)
    if partial:
        upstream.copy_assets(a.origin_hf_dir, partial)  # config.json, tokenizer, chat template, preprocessors
        with open(os.path.join(partial, "export_verified.json"), "w") as f:
            json.dump(report, f, indent=1)
        os.rename(partial, a.output_dir)
        print(f"[export] wrote {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
