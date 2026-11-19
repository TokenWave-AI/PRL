# Setting up a site

A site is a cluster, or a single machine, that runs PRL. Set it up once, in two steps:

| # | Step | Command |
|---|---|---|
| 1 | **Site profile**: fill in its first block (scheduler, a work directory, the sandbox service, the registry, the judge) and put the key files where it names them (chmod 600) | `cp configs/sites/example.env configs/sites/<site>.env` |
| 2 | **Sandboxes**: an image and a template per task, on a machine with Docker | `PRL_SITE=<site> bash scripts/setup/build_sandboxes.sh` ([`sandboxes.md`](sandboxes.md)) |

Then, from the repository root on the cluster:

```bash
PRL_SITE=<site> bash scripts/quickstart.sh
```

Its first job also does the rest, each part once: the container image (`radixark/miles:v0.1.2`), Harbor and its
dependencies, the tasks (728 training + 70 TB-dev), the weights and their Megatron checkpoint, and a framework smoke
(2 updates on 8 tasks). Later runs skip them.

## What a site needs

- One node of 4 GPUs with ≥ 90 GB each (e.g. H100 94 GB, H200, GB200), ≥ 768 GB of host RAM and node-local NVMe.
  Another GPU model: set `NUM_GPUS`, `PRL_GPU_MIN_MIB` and the memory settings in `configs/arms/common_ota_single_node.env`.
- Apptainer ≥ 1.4 or SingularityCE on the GPU nodes, and outbound internet (sandboxes, model API, Hugging Face).
- PBS Pro, Fujitsu PJM, or no scheduler (`PRL_SCHEDULER=local`: the job runs on the machine you start it on).
- An E2B-compatible sandbox service (anything the e2b SDK can talk to) and a container registry it can pull from.
- For step 2, a linux/amd64 machine with Docker (buildx) and sudo; it may be the GPU machine.
- For transforms that ask a model (e.g. signed ATLAS, the default): an OpenAI-compatible Chat Completions endpoint.

## Good to know

- **A quicker first try**: `PRL_SANDBOX_SMOKE_ONLY=1` in step 2 builds only the smoke's 8 tasks; the same quick
  start command then tries the transform on them (2 updates, about 30 min; the job asks for 2 h) instead of training.
  Step 2 run again without it builds the rest. The smoke splits half of its trials with the context summary on
  purpose; a transform that rewrites or rerolls skips those (a split trajectory cannot be rewritten).
- **Logs**: in `$PRL_EXT/logs/`; each run's `train.log` (`$PRL_EXT/runs/<run>/`) has a line per update,
  `step N: {'train/loss': ...}`; the quick start prints both places. Harmless at the start: SGLang's `post-warmup
  freeze_gc failed` with a `Connection refused` traceback, FlashInfer's `workspace preflight probe failed (...
  CUDA_ERROR_NOT_SUPPORTED: 801)` on GB200, and Apptainer's `signature not found` for the pulled (unsigned) image. Later
  a trial's `failed to send non-blocking keys ... no server running` (Harbor's tmux) is tried again by itself
  (`rollout/abort_retried_trials`), as is one that Harbor aborts with `Command ended without an end event` (a
  traceback, then Miles' unrelated `All samples truncated`; then `trial aborted, retrying it alone`), and one with `mismatch_refill rejected` (its tokens did not match the
  inference server's; `exhausted`: its group is replaced).
- `DRY_RUN=1` prints the scheduler command and submits nothing; `ELAPSE=hh:mm:ss` sets a job's time limit and
  `QUEUE=<queue>` its queue (PBS) or resource group (PJM), e.g. a short debug queue for the quicker first try, with
  the first job's steps run ahead (below). A job stopped by its time limit goes on when the same command is run
  again.
- `PRL_SITE=<site> CHAIN=task_check bash scripts/submit.sh` checks the smoke's 8 sandboxes alone (the official
  solution must score 1, a no-op agent 0); the first smoke does it by itself.
- The first job's steps, to run ahead (e.g. under a short time limit): `CHAIN=prepare` (image, Harbor, tasks),
  `download` (weights), `convert` (Megatron checkpoint), joined by `+`; measured on GB200: image pull ~10 min, the rest of
  `prepare` ~2 min, `convert` ~2 min; then the smoke reaches its first update in ~15-20 min.

## Site specifics

- Put the repository where the compute nodes can write: every job applies `patches/` to `third_party/` into `build/`
  there (`bash scripts/setup/check_third_party.sh` checks that `third_party/` is byte-identical to upstream).
- The containers see only `PRL_ROOT`, `PRL_EXT`, `HF_HOME`, `PRL_SCRATCH` and `PRL_EXTRA_BINDS`: keep every file a job
  reads, the key files too, under them (e.g. `$PRL_EXT/private/`).
- Step 2 on another machine than the cluster (no Docker there, or not linux/amd64): a clone of the repository there
  with the same `configs/sites/<site>.env` (git ignores it: copy it), the paths of this machine, then the snapshots
  copied to the cluster ([`sandboxes.md`](sandboxes.md)):

  ```bash
  PRL_SITE=<site> PRL_EXT=<dir> E2B_API_KEY_FILE=<sandbox key> PRL_REGISTRY_CREDENTIALS_FILE=<registry credentials> \
    bash scripts/setup/build_sandboxes.sh   # PRL_DOCKER_COMMAND="sudo docker --config <dir>": a Docker logged in elsewhere
  rsync -a <dir>/task_snapshots <cluster>:<the cluster's PRL_EXT>/
  ```

  The registry credentials are needed only there; the cluster needs the sandbox key and the judge key.
- SingularityCE, or login nodes unlike the compute nodes (another CPU, Python < 3.10): the commented settings under
  "Only for some sites" in `configs/sites/example.env`.
- No scheduler inside another scheduler's allocation: add `FOREGROUND=1`; that scheduler sets the time limit (`ELAPSE`
  and `AFTER` do not apply), e.g. `sbatch -N 1 --gpus 4 --wrap "PRL_SITE=<site> FOREGROUND=1 bash scripts/quickstart.sh"`,
  or `echo "cd $PWD && PRL_SITE=<site> FOREGROUND=1 bash scripts/quickstart.sh" | qsub -l select=1 -l walltime=168:00:00`.
- Another scheduler: its own `scripts/<scheduler>/{submit.sh,job.sh}`, about 30 lines each (copy `scripts/pbs/`).
