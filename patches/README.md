# patches/: every modification we make to vendored upstream code

The vendored trees are kept **byte-identical to upstream**:

| Vendored dir | Upstream | Pinned commit | Notes |
|---|---|---|---|
| `third_party/miles/` | https://github.com/radixark/miles (fork of THUDM/slime) | `49576fd` (2026-10-02) | Runs from `build/miles` in the container `radixark/miles:v0.1.2` (amd64 and arm64; CUDA 13, needs host driver ≥ 580), which provides the environment (SGLang, Megatron-LM, torch) |

All our changes live here as `git format-patch` files, one change per patch; the rationale is in the commit message or in the patch's `# prl:` comments:

| Patch | What / why |
|---|---|
| `miles/0001-…ship-sample.train_metadata-to-the-trainer.patch` | Adds `"metadata"` to the per-DP-rank key whitelist. Upstream collects `train_data["metadata"]` from `sample.train_metadata` but never ships it to the trainer, so custom advantage/loss code cannot see per-sample metadata (e.g. step weights). |
| `miles/0002-…add-custom-advantage-postprocess-path-hook.patch` | New flag `--custom-advantage-postprocess-path`: `fn(args, rollout_data, advantages, returns) -> (advantages, returns)`, called right after the advantage estimator (before OPD / normalization). Used by `prl.algo.step_advantage`. |
| `miles/0003-…fused-chunked-linear-log-prob….patch` | `PRL_FUSED_LOGPROB=1`: forward steps pass next-token labels so the policy log-prob comes from `prl.algo.fused_logprob` (chunked, recomputed in backward) instead of full-vocabulary logits, which do not fit one GPU for 64k-token trajectories. A PPO critic keeps `labels=None` (`fused_logprob.install()` skips a 1-wide value head; before, the critic's first forward raised `TypeError`). |
| `miles/0004-…per-group-gradient-norm-diagnostic.patch` | `PRL_LOG_GRAD_GROUPS=1`: logs the gradient norm per parameter group. |
| `miles/0005-…rollout-gpus-on-the-ray-head-node.patch` | `PRL_ROLLOUT_ON_HEAD=1`: ranks the head node's GPUs last so they become rollout GPUs: the RolloutExecutor, router and session servers (pinned to the head) then share a rollout node, not a trainer node's host memory. |
| `miles/0006-…zero-initialised-ppo-value-head.patch` | `PRL_VALUE_HEAD_INIT=zero`: the PPO value head starts at zero, so the critic starts at V = 0: the default init gave values of about ±6 for 0/1 rewards, and the first ten updates went into shrinking them. The default `upstream` keeps Miles' init; `zero` is opt-in. Unknown values raise. |
| `miles/0007-…per-role-train-disk-offload-dirs.patch` | Actor and critic get their own `--offload-train-disk-dir` subdirectory: they share cells and ranks, and one's startup wipe deleted the other's backup. |
| `miles/0008-…zero-ppo-value-head-loaded-from-an-lm-checkpoint.patch` | Same zero init when the critic is loaded from an LM (release) checkpoint, which skips the finetune path; only with `PRL_VALUE_HEAD_INIT=zero`. Without it, such a critic's head is the checkpoint's output-layer row 0 (value loss ~41 at rollout 0). |
| `miles/0009-…malloc-trim-when-clearing-host-memory.patch` | `malloc_trim` after clearing host memory: glibc keeps freed heap in its arenas, and a PPO relay's sleeping actor kept ~8 GB more RSS on one node, which pushed it over the 100 GB limit. |
| `miles/0010-…ref-weight-backup-on-train-offload-disk.patch` | The reference model's weight backup goes to the train offload disk instead of pinned host memory. |
| `miles/0011-…ppo-actor-checkpoint-saves-its-real-weights.patch` | Under `--offload-train` the PPO actor's checkpoint saved stale weights; it now saves the real ones. `--rebuild-params-from-master-weight-on-load` repairs a checkpoint saved before the fix (fp32 params, which have no master copy, from the ref model). Before the fix, a PPO smoke's `iter_0000001` held zeros in 427 of 427 tensors. |
| `miles/0012-…drop-stale-optimizer-masters-after-checkpoint-load.patch` | After a checkpoint load, the CPU-offloaded optimizer's old GPU master copies are released at once instead of at the first step (~2 GB per rank, which OOMed relay jobs). |
| `miles/0013-…training-entropy-from-the-fused-log-prob-k….patch` | With the fused log-prob (0003) the full softmax is never built, so `--observe-training-entropy` logged `train/entropy_loss` = 0. `prl.algo.fused_logprob` now computes the exact entropy of the training forward (`PRL_OBSERVE_ENTROPY=1`, default on in `scripts/lib/env.sh`; detached, observed only), and this splits it into per-sample response chunks like the log-probs. |
| `miles/0014-…advantage-return-value-distributions-and-t….patch` | Miles logs only the means of advantages / returns / values. Adds their std / min / max over the loss-masked tokens and the critic's explained variance (verl's `vf_explained_var`), reduced over DP x CP, as `rollout_dist N`; a failing rank still joins the gather. |
| `miles/0015-…dumps-without-the-routing-arrays.patch` | `--dump-without-routed-experts`: `--dump-details` saved every sample's R3 routes ((tokens − 1) × layers × top-k int32, ~13 GB per update of a 35B-A3B 64 × 8 run, against the group's file quota); the flag leaves them out of the rollout and train dumps. Training is not affected. |
| `miles/0016-…async-loop-does-not-save-a-frozen-actor.patch` | `train_async.py` skips the actor's save during `--num-critic-only-steps`, as `train.py` does; before, Megatron raised `KeyError: 'master_param'` on an optimizer that had not stepped. |

Extensions that need no patch live in `prl/` instead (Miles' plug-in paths: `--custom-*-path`).

## How it is used

```
third_party/miles/ + patches/miles/*.patch ── scripts/setup/apply_patches.sh ─► build/miles
```

- `scripts/setup/apply_patches.sh` copies the vendored tree to `build/<name>/` and `git apply`s the patches in order. It writes a stamp (hash of upstream commit + patches + source tree) and is a no-op when nothing changed. A file lock prevents concurrent jobs from building at the same time.
- `scripts/lib/env.sh` runs it automatically before entering the container. All runtime paths use `build/`: PYTHONPATH, `cd build/miles && python3 train.py`, and checkpoint conversion.
- `build/` is generated. Never edit it, and never edit `third_party/`.

## Workflow

- **Change or add a patch:** edit in a scratch git checkout of the pinned commit, commit, then `git format-patch <commit> -o patches/<name>/`. Next launch rebuilds `build/` automatically, or run `bash scripts/setup/apply_patches.sh` directly.
- **Check the vendored trees are untouched:** `bash scripts/setup/check_third_party.sh` (clones upstream at the pinned commits and diffs).
- **Upgrade upstream:**
  1. Replace the vendored dir with the new commit.
  2. Update `patches/<name>/UPSTREAM_COMMIT`.
  3. Run `apply_patches.sh` and fix any patch that no longer applies.
  4. For Miles, also switch to the matching container tag (see `scripts/lib/env.sh`, `configs/sites/`).

