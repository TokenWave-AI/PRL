# Setting up a site

A site is a cluster, or a single machine, that runs PRL. Set it up once; then `bash scripts/quickstart.sh` trains and
evaluates (README, Quick start ③).

What a site needs:

- GPUs: one node of 4 GPUs with ≥ 90 GB each (e.g. H100 94 GB, H200, GB200), ≥ 768 GB of host RAM and node-local
  NVMe. Another GPU model: set `NUM_GPUS`, `PRL_GPU_MIN_MIB` and the memory settings in
  `configs/arms/common_ota_single_node.env`.
- Apptainer ≥ 1.4 or SingularityCE on the GPU nodes, and outbound internet (sandboxes, model API, Hugging Face).
- A scheduler: PBS Pro (`scripts/pbs/`), Fujitsu PJM (`scripts/pjm/`), or none (`local`: the job runs on the machine
  you start it on). Another scheduler needs its own `scripts/<scheduler>/{submit.sh,job.sh}`, about 30 lines each.
- An E2B-compatible sandbox service (anything the e2b SDK can talk to) and its API key, and a container registry it
  can pull from.
- A linux/amd64 machine with Docker (buildx) and sudo to build the task images; it may be the GPU machine.
- For transforms that ask a model (e.g. signed ATLAS): an OpenAI-compatible Chat Completions endpoint.

| # | Step | Command |
|---|---|---|
| 1 | Site profile: scheduler, paths, sandbox endpoint, registry, judge. Keys and passwords only as files (chmod 600) | copy [`configs/sites/example.env`](../configs/sites/example.env) to `configs/sites/<site>.env` (not tracked by git) |
| 2 | The container image (when missing), Harbor and its dependencies for the container, the tasks (728 training + 70 TB-dev) | `PRL_SITE=<site> CHAIN=prepare bash scripts/submit.sh` |
| 3 | Sandboxes: an image and a template per task | on the Docker machine: `PRL_SITE=<site> bash scripts/setup/build_sandboxes.sh` ([`docs/sandboxes.md`](sandboxes.md)) |
| 4 | Train and evaluate | `PRL_SITE=<site> bash scripts/quickstart.sh` (README, Quick start ③) |

- Put the repository on the cluster where the compute nodes can write: every job applies `patches/` to `third_party/`
  into `build/` there (`bash scripts/setup/check_third_party.sh` checks that `third_party/` is byte-identical to upstream).
- The containers see only `PRL_ROOT`, `PRL_EXT`, `HF_HOME` (writable), `PRL_SCRATCH` and the directories in
  `PRL_EXTRA_BINDS`: keep every file a job reads, the key files too, under them (e.g. `$PRL_EXT/private/`).
- SingularityCE instead of Apptainer, or login nodes unlike the compute nodes (another CPU, Python < 3.10): the
  commented settings of `configs/sites/example.env` (`scripts/compat` on `PATH`, `PRL_SINGULARITY`,
  `PRL_SANDBOX_UNSQUASH=1`, `PRL_HOST_PYTHON`, `PRL_SUBMIT_FINDS_SMOKE=0`).
- No scheduler (`PRL_SCHEDULER=local`): a command starts the job in the background on this machine; inside another
  scheduler's allocation add `FOREGROUND=1`, e.g. `sbatch -N 1 --gpus 4 --wrap "PRL_SITE=<site> FOREGROUND=1 bash scripts/quickstart.sh"`.
- Step 3 on another machine than the cluster: copy `$PRL_EXT/task_snapshots` to the cluster's `PRL_EXT` afterwards.
- `PRL_SANDBOX_SMOKE_ONLY=1` builds only the 8 tasks of the framework smoke. The quick start then runs its framework
  smoke on them (2 plain GRPO updates) and stops before training, which needs every training task's template. To try a
  transform on these 8 tasks: `PRL_SITE=<site> CHAIN=download+convert+smoke_grpo TRANSFORM=atlas bash scripts/submit.sh`.
- `PRL_SITE=<site> CHAIN=task_check bash scripts/submit.sh` checks the smoke's 8 sandboxes before any training: the
  official solution must score 1, a no-op agent 0. The first smoke does this by itself when it was not done.
- The first run of step 4 also downloads the weights and converts them for Megatron, and runs the framework smoke
  (2 updates on 8 tasks); later runs of the same code skip these.
- `DRY_RUN=1` prints the scheduler command and submits nothing; `ELAPSE=hh:mm:ss` sets a job's time limit. Job logs are
  in `$PRL_EXT/logs/`; each run's `train.log` (in `$PRL_EXT/runs/<run>/`) has a line per update, `step N: {'train/loss': ...}`.
