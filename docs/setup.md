# Setting up a cluster

Everything below is done once per cluster. A ready site needs the following:

- GPU nodes that run the Miles container, with Apptainer ≥ 1.4 or SingularityCE. Any GPU model: set `NUM_GPUS`,
  `PRL_GPU_MIN_MIB` and the memory settings in `configs/arms/common_ota_single_node.env` to match your nodes. Compute
  nodes need outbound internet to reach the sandboxes and any model API.
- An E2B-compatible sandbox service (anything the e2b SDK can talk to), its API key, and one template per task.
  Templates belong to an account, so a new account needs its own.
- A machine with Docker and sudo to build the task images.
- Optional: an OpenAI-compatible model API for transforms that ask a model.

Steps (launchers exist for Fujitsu PJM, `scripts/pjm/`, and PBS Pro, `scripts/pbs/`; another scheduler needs its own
`scripts/<scheduler>/{submit.sh,job.sh}`, about 30 lines each, copied from one of them):

| # | Step | Command |
|---|---|---|
| 1 | Site profile: scheduler and queue, paths (`PRL_EXT` for data, images, checkpoints and runs; `HF_HOME`; `MILES_IMAGE`), host Python, Apptainer or SingularityCE, the sandbox endpoint (`E2B_API_URL`, `E2B_DOMAIN`), the judge endpoint | copy [`configs/sites/example.env`](../configs/sites/example.env) to `configs/sites/<site>.env` (not tracked by git); select it with `PRL_SITE=<site>` |
| 2 | Miles container image (17.7 GB; `mksquashfs` needs ~74 GB RAM, so use a large CPU node) | `apptainer pull $PRL_EXT/sif/miles_dev-202610022330.sif docker://radixark/miles:dev-202610022330` |
| 3 | Patched upstream trees into `build/` (also run by every launch) and the check that `third_party/` is pristine | `bash scripts/setup/apply_patches.sh`; `bash scripts/setup/check_third_party.sh` |
| 4 | Harbor, the e2b SDK and litellm for the container, and the e2b version the templates were checked with (a node with internet) | `PRL_SITE=<site> bash scripts/setup/install_harbor_deps.sh` |
| 5 | Tasks (728 train + 70 TB-dev) as Harbor dirs and Miles JSONL, plus the model weights | `scripts/setup/prepare_otagent.sh` (usage in its header) |
| 6 | Megatron checkpoint of the base model (the training start and the KL reference) | `MODEL=qwen3.5-9B bash scripts/setup/convert_ckpt.sh` on one GPU |
| 7 | Sandboxes: a task snapshot for training and one for TB-dev (the task directories, with tmux added to their images), an image per task, pushed and registered as a template, the manifest in the snapshot, and the oracle / no-op results | `scripts/tools/make_task_snapshot.py`, then `scripts/tools/prebuild_task_images.py` on the Docker machine; the push, the registration and the checks: [`docs/sandboxes.md`](sandboxes.md) |
| 8 | Keys: private files (0600), never in the environment or the repo | sandboxes: `E2B_API_KEY_FILE` in the site profile; model API: `PRL_JUDGE_KEY_FILE` in the site profile (`configs/arms/ota_judge.env` lists the judge settings) |
| 9 | Sandbox checks | `python3 scripts/checks/sandbox_smoke.py` (the sandbox contract), `CHAIN=task_check` (oracle must score 1, no-op 0, on a template manifest) |
| 10 | Framework smoke: two updates on 8 tasks, then a CPU acceptance that writes the receipt every training job requires. Needed once per framework version, not per transform | see below |
| 11 | Trainer and failure checks (optional, after a trainer or launcher change): train-only replays of the smoke's batches (resume, data parallelism, longest sample, NaN) and drills (killed save, killed or frozen rank, SIGTERM) | `CHAIN=replay PRL_REPLAY_SOURCE=<smoke RUN_DIR>`, or `CHAIN=smoke_grpo+replay`; verdict in `$PRL_EXT/runs/replay_<job>/replay.json` |

The framework smoke for GRPO (PPO: `CHAIN=smoke_ppo ... PRL_RUN_ID=<id>`; the reroll and async smokes:
`ARM=ota_train_smoke_reroll`, `ARM=ota_train_smoke_async`):

```bash
PRL_SITE=<site> CHAIN=smoke_grpo TRANSFORM=atlas bash scripts/submit.sh   # ARM defaults to ota_train_smoke_transform
```

The smoke runs 8 tasks of the training snapshot (`configs/arms/ota_train_smoke.env`); its preflight reuses the
oracle / no-op result of those 8 on the snapshot's templates (`<snapshot>/contract`; a new snapshot gets one with
`CHAIN=task_check+smoke_grpo PRL_TEMPLATE_MANIFEST=$S/manifest.json PRL_TASK_ROOT=$S/tasks
PRL_TASK_PROMPTS=$S/smoke.jsonl PRL_TASK_CHECK_OUTPUT=$S/contract` for `S=$PRL_TRAIN_SNAPSHOT`,
see `scripts/chain/task_smoke.sh`). The receipt (`acceptance.json`) records the hashes of the framework sources
(not `transforms/`), the model, the context and the schedule; a training job takes the newest receipt that matches
(`scripts/checks/find_smoke.py`).
