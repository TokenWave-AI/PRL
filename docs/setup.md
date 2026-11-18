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
  start command then tries the transform on them (2 updates, about 30 min) instead of training. Step 2 run again
  without it builds the rest.
- **Logs**: in `$PRL_EXT/logs/`; each run's `train.log` (`$PRL_EXT/runs/<run>/`) has a line per update,
  `step N: {'train/loss': ...}`. SGLang's `post-warmup freeze_gc failed` with a `Connection refused` traceback at the
  start is harmless.
- `DRY_RUN=1` prints the scheduler command and submits nothing; `ELAPSE=hh:mm:ss` sets a job's time limit. A job
  stopped by its time limit goes on when the same command is run again.
- `PRL_SITE=<site> CHAIN=task_check bash scripts/submit.sh` checks the smoke's 8 sandboxes alone (the official
  solution must score 1, a no-op agent 0); the first smoke does it by itself. `CHAIN=prepare` alone does the first
  job's preparation only (e.g. the image ahead of time).

## Site specifics

- Put the repository where the compute nodes can write: every job applies `patches/` to `third_party/` into `build/`
  there (`bash scripts/setup/check_third_party.sh` checks that `third_party/` is byte-identical to upstream).
- The containers see only `PRL_ROOT`, `PRL_EXT`, `HF_HOME`, `PRL_SCRATCH` and `PRL_EXTRA_BINDS`: keep every file a job
  reads, the key files too, under them (e.g. `$PRL_EXT/private/`).
- Step 2 on another machine than the cluster: set `PRL_EXT` to a directory there, then copy its `task_snapshots/` to
  the cluster's `PRL_EXT` ([`sandboxes.md`](sandboxes.md)).
- SingularityCE, or login nodes unlike the compute nodes (another CPU, Python < 3.10): the commented settings under
  "Only for some sites" in `configs/sites/example.env`.
- No scheduler inside another scheduler's allocation: add `FOREGROUND=1`, e.g.
  `sbatch -N 1 --gpus 4 --wrap "PRL_SITE=<site> FOREGROUND=1 bash scripts/quickstart.sh"`.
- Another scheduler: its own `scripts/<scheduler>/{submit.sh,job.sh}`, about 30 lines each (copy `scripts/pbs/`).
