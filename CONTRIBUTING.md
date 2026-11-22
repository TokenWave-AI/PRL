# Contributing

- Never edit `third_party/` or `build/`. Change upstream code only through `patches/`, one `git format-patch` file
  per change, listed in [`patches/README.md`](patches/README.md). Runtime code runs from `build/`.
- Settings are environment variables prefixed `PRL_`, documented in each module's docstring and forwarded by
  `scripts/run.sh`. Infrastructure failures raise; a reward of 0 means a real failure.
- Tests: `pip install -e '.[check,dev]'`, then `python -m pytest -q tests/unit`; the tests that need torch, Miles or
  Harbor skip without them (all of them run inside the Miles image, `CHAIN=unit`). Changing a framework source listed in `SOURCE_FILES` of
  `scripts/checks/grpo_acceptance.py` (the transform core and check, the rollout adapters, `scripts/run.sh`)
  invalidates the smoke receipts; run a new framework smoke.
- Framework smokes: `CHAIN=smoke_grpo TRANSFORM=atlas` (GRPO; the reroll and async smokes:
  `ARM=ota_train_smoke_reroll`, `ARM=ota_train_smoke_async`), `CHAIN=smoke_ppo TRANSFORM=atlas PRL_RUN_ID=<id>` (PPO, two
  jobs in one). The receipt (`acceptance.json`) records the hashes of the framework sources (not `transforms/`), the
  model, the context and the schedule; a training job takes the newest receipt that matches
  (`scripts/checks/find_smoke.py`).
- After a change to the trainer or the launchers: train-only replays of a smoke's batches (resume, data parallelism,
  the longest sample, NaN) and failure drills (killed save, killed or frozen rank, SIGTERM):
  `CHAIN=replay PRL_REPLAY_SOURCE=<smoke RUN_DIR>`, or `CHAIN=smoke_grpo+replay`; the verdict is in
  `$PRL_EXT/runs/replay_<job>/replay.json`.

## Repository layout

```
transforms/            your trajectory transforms, one file each (atlas.py, atlas_signed.py, rewrite.py, branch.py)
prl/
  transforms/          the framework: core.py (hook, checks, records), api.py (Step / Trajectory), check.py (pre-run check)
  rollout/             Miles entry points: Harbor generate/agent adapters, branching, session sample picker/postprocessor
  algo/                step_advantage.py (trainer hook), llm_client.py (model calls), labeler.py (ATLAS judge)
  sandbox/             task template manifest resolution
  datasets/            dataset converter (harbor/: Harbor task dirs -> Miles JSONL)
configs/
  sites/               example.env: the template of a site profile (PRL_SITE; your own profiles stay untracked)
  models/              model profiles (MODEL)
  datasets/            dataset presets (DATASET)
  arms/                experiment settings (ARM) on common_ota_single_node.env; ota_train_transform.env runs a
                       transform; ota_judge.env the judge settings
scripts/
  quickstart.sh        the quick start: download+convert+smoke+train_eval in one job, signed ATLAS by default
  submit.sh            submit on the site PRL_SITE names, through its scheduler's wrapper
  pjm/, pbs/, local/   submit.sh + job.sh per scheduler (Fujitsu PJM, PBS Pro, none: this machine)
  compat/              apptainer on SingularityCE; the image's Python as the host Python
  chain/               the chain steps: smoke, acceptance, full training, export, eval, task_check, replay
  lib/                 chain.sh (the steps), submit_chain.sh, env.sh, container_main.sh, relay.sh, guard.sh
  checks/              what a job runs to check itself: smoke preflight and acceptance, find_smoke.py (the receipt
                       a training job needs), the training watchdog, sandbox and task (oracle / nop) checks
  run_job.sh, run.sh   one Miles run: site + arm -> container
  setup/               one-off setup (build_sandboxes.sh, patches, deps, data, checkpoint conversion)
  tools/               maintenance (tbdev_score.py, prebuild_task_images.py, register_templates.py, prune_ckpts.sh,
                       torch_dist_to_hf.py)
  analysis/            training curves (plot_training.py, training_dynamics.py), not used by the pipeline
tests/
  unit/                pytest, no GPU or sandbox (CHAIN=unit runs it in the Miles image)
  fixtures/            the recorded trials of the transform pre-run check
third_party/           upstream Miles, byte-identical to its pinned commit (never edited)
patches/               every change to third_party/, applied into build/ (see patches/README.md)
assets/brand/          logos
docs/                  setup.md (setting up a site), sandboxes.md (task images and sandbox templates)
```
