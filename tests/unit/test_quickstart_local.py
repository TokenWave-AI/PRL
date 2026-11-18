"""The quick start's one command (scripts/quickstart.sh) as it is submitted, the scheduler-less launcher
(scripts/local/: variables, background start, one job per machine) and the oracle / no-op check a smoke runs first
when its snapshot has none (scripts/lib/chain.sh smoke_contract), with fakes for the scheduler, the GPUs and the
container. Bash only. Two throwaway site profiles (configs/sites/_test_qs_*.env, ignored by git)."""

from __future__ import annotations

import fcntl
import json
import socket
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMON = ("PRL_EXT=${PRL_EXT:-/nonexistent/ext}\nPRL_SCRATCH=${PRL_SCRATCH:-/tmp/scratch_test}\nHF_HOME=/nonexistent/hf\n"
          "MILES_IMAGE=/nonexistent/miles.sif\nE2B_API_URL=https://e2b.example\nE2B_DOMAIN=e2b.example\n"
          "E2B_API_KEY_FILE=$PRL_EXT/private/e2b_api_key\nPRL_HOST_PYTHON=python3\n"
          "PRL_JUDGE_API_BASE=${PRL_JUDGE_API_BASE:-https://judge.example/v1}\nPRL_JUDGE_MODEL=judge\n"
          "PRL_JUDGE_KEY_FILE=$PRL_EXT/private/judge_api_key\n")
SITES = {"_test_qs_pbs": "PRL_SCHEDULER=pbs\nPRL_PBS_QUEUE=gpu\n", "_test_qs_local": "PRL_SCHEDULER=local\n"}


@pytest.fixture(autouse=True, scope="module")
def _sites():
    paths = [ROOT / "configs" / "sites" / f"{name}.env" for name in SITES]
    for path, body in zip(paths, SITES.values()):
        path.write_text(body + COMMON)
    yield
    for path in paths:
        path.unlink(missing_ok=True)


def _sandboxes(tmp_path: Path, ready: int, tasks: int = 12) -> None:
    """The training snapshot build_sandboxes.sh leaves: its tasks, and templates for the first `ready` of them."""
    snapshot = tmp_path / "ext" / "task_snapshots" / "otagent728"
    (tmp_path / "ext" / "private").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ext" / "private" / "judge_api_key").write_text("k\n")
    for i in range(tasks):
        (snapshot / "tasks" / f"t{i}").mkdir(parents=True, exist_ok=True)
    rows = {f"t{i}": {"status": "ready" if i < ready else "failed"} for i in range(tasks)}
    (snapshot / "manifest.json").write_text(json.dumps({"tasks": rows}, indent=2))


@pytest.fixture
def built(tmp_path: Path) -> Path:
    _sandboxes(tmp_path, ready=12)
    return tmp_path


def _run(command: list[str], tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    fakes = tmp_path / "bin"
    fakes.mkdir(exist_ok=True)
    (fakes / "nvidia-smi").write_text("#!/bin/bash\necho 'NVIDIA GB200, 189471 MiB'\n")
    (fakes / "nvidia-smi").chmod(0o755)
    out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False, env={
        "PATH": f"{fakes}:/usr/bin:/bin", "HOME": "/tmp", "USER": "u", "PRL_EXT": str(tmp_path / "ext"),
        "PRL_SCRATCH": str(tmp_path / "scratch"), **env})
    out.stdout = out.stdout.replace("\\,", ",")  # the dry run prints the command %q-quoted: commas escaped
    return out


def test_quickstart_submits_signed_atlas_in_one_job(built: Path) -> None:
    tmp_path = built
    out = _run(["bash", "scripts/quickstart.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1")
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("qsub ")
    for part in ("PRL_CHAIN=prepare+download+convert+smoke+train_eval", "SMOKE_ARM=ota_train_smoke_noturncap",
                 "ARM=ota_train_atlas_signed", "ALGO=grpo", "PRL_RUN_ID=quickstart", "TRANSFORM=atlas_signed",
                 "PRL_TRANSFORM_ARGS_FILE=", "walltime=168:00:00"):
        assert part in out.stdout, part


def test_quickstart_follows_the_sandboxes_built(tmp_path: Path) -> None:
    (tmp_path / "ext" / "private").mkdir(parents=True)
    (tmp_path / "ext" / "private" / "judge_api_key").write_text("k\n")
    none = _run(["bash", "scripts/quickstart.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1")
    assert none.returncode != 0 and "no sandboxes yet" in none.stderr and "build_sandboxes.sh" in none.stderr
    _sandboxes(tmp_path, ready=8)  # PRL_SANDBOX_SMOKE_ONLY=1: the smoke's 8 only, so the transform's smoke on them
    tried = _run(["bash", "scripts/quickstart.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1")
    assert tried.returncode == 0, tried.stderr
    assert "trying atlas_signed on the smoke's 8" in tried.stderr
    for part in ("PRL_CHAIN=prepare+download+convert+smoke_grpo,", "ARM=ota_train_smoke_transform",
                 "TRANSFORM=atlas_signed", "PRL_TRANSFORM_ARGS_FILE=", "walltime=08:00:00"):
        assert part in tried.stdout, part
    assert "train_eval" not in tried.stdout
    rpp = _run(["bash", "scripts/quickstart.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1",
               ALGO="reinforce_plus_plus")
    assert rpp.returncode != 0 and "8 have a template" in rpp.stderr
    _sandboxes(tmp_path, ready=3)
    few = _run(["bash", "scripts/quickstart.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1")
    assert few.returncode != 0 and "3 of 12 training tasks" in few.stderr


def test_quickstart_wants_the_judge_of_a_transform_that_asks(built: Path) -> None:
    unset = _run(["bash", "scripts/quickstart.sh"], built, PRL_SITE="_test_qs_pbs", DRY_RUN="1",
                 PRL_JUDGE_API_BASE="https://<provider>/v1")
    assert unset.returncode != 0 and "atlas_signed asks a judge" in unset.stderr
    (built / "ext" / "private" / "judge_api_key").unlink()
    nokey = _run(["bash", "scripts/quickstart.sh"], built, PRL_SITE="_test_qs_pbs", DRY_RUN="1", TRANSFORM="rewrite")
    assert nokey.returncode != 0 and "rewrite asks a judge" in nokey.stderr
    # no model call: no judge needed
    plain = _run(["bash", "scripts/quickstart.sh"], built, PRL_SITE="_test_qs_pbs", DRY_RUN="1",
                 TRANSFORM="reroll_smoke", PRL_JUDGE_API_BASE="https://<provider>/v1")
    assert plain.returncode == 0, plain.stderr


def test_quickstart_runs_your_transform_on_the_transform_arm(built: Path) -> None:
    tmp_path = built
    out = _run(["bash", "scripts/quickstart.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1",
               TRANSFORM="atlas", PRL_RUN_ID="mine")
    assert out.returncode == 0, out.stderr
    assert "ARM=ota_train_transform," in out.stdout and "TRANSFORM=atlas," in out.stdout
    missing = _run(["bash", "scripts/quickstart.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1",
                   TRANSFORM="atlas+no_such_transform")
    assert missing.returncode != 0 and "no transforms/no_such_transform.py" in missing.stderr


def test_quickstart_ppo_needs_its_own_smoke(built: Path) -> None:
    tmp_path = built
    out = _run(["bash", "scripts/quickstart.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1",
               TRANSFORM="atlas", ALGO="ppo")
    assert out.returncode != 0 and "CHAIN=smoke_ppo first" in out.stderr
    signed = _run(["bash", "scripts/quickstart.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1", ALGO="ppo")
    assert signed.returncode != 0 and "signed ATLAS needs one advantage sign per trial" in signed.stderr


def test_local_submission_names_the_job_and_refuses_after(tmp_path: Path) -> None:
    out = _run(["bash", "scripts/submit.sh"], tmp_path, PRL_SITE="_test_qs_local", CHAIN="prepare", DRY_RUN="1")
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("bash scripts/local/job.sh local") and "PRL_CHAIN=prepare" in out.stdout
    after = _run(["bash", "scripts/submit.sh"], tmp_path, PRL_SITE="_test_qs_local", CHAIN="prepare", AFTER="1")
    assert after.returncode != 0 and "AFTER does not apply" in after.stderr


VARS = "PRL_SITE=_test_qs_local,PRL_CHAIN=no_such_step,MODEL=qwen3.5-9B"


def test_local_job_in_the_foreground_runs_the_chain(tmp_path: Path) -> None:
    out = _run(["bash", "scripts/local/job.sh", "local1", VARS], tmp_path, FOREGROUND="1")
    assert out.returncode != 0
    assert "chain no_such_step on" in out.stdout and "unknown step no_such_step" in out.stdout
    assert "unknown step no_such_step" in (tmp_path / "ext/logs/no_such_step_local1.log").read_text()


def test_local_job_detaches_into_the_background(tmp_path: Path) -> None:
    out = _run(["bash", "scripts/local/job.sh", "local2", VARS], tmp_path)
    assert out.returncode == 0 and "started in the background" in out.stdout, out.stderr
    log = tmp_path / "ext/logs/local_local2.out"
    for _ in range(100):
        if log.exists() and "unknown step no_such_step" in log.read_text():
            break
        time.sleep(0.1)
    assert "unknown step no_such_step" in log.read_text()


def test_a_second_local_job_on_the_machine_stops(tmp_path: Path) -> None:
    lock = tmp_path / f"ext/locks/local_{socket.gethostname().split('.')[0]}.lock"
    lock.parent.mkdir(parents=True)
    with lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        out = _run(["bash", "scripts/local/job.sh", "local3", VARS], tmp_path, FOREGROUND="1")
    assert out.returncode != 0 and "another local job runs" in out.stderr


def _contract(tmp_path: Path, status: int) -> subprocess.CompletedProcess:
    """smoke_contract of the GRPO smoke arm with a fake container that records the check's settings."""
    calls = tmp_path / "apptainer.log"
    fake = tmp_path / "bin" / "apptainer"
    fake.parent.mkdir(exist_ok=True)
    fake.write_text(f'#!/bin/bash\necho "$PRL_TASK_CHECK_OUTPUT $*" >> {calls}\n'
                    f'mkdir -p "$PRL_TASK_CHECK_OUTPUT" && touch "$PRL_TASK_CHECK_OUTPUT/summary.json"\nexit {status}\n')
    fake.chmod(0o755)
    script = ("set -euo pipefail; set -a; source configs/sites/" + "_test_qs_local.env; set +a; source scripts/lib/chain.sh; "
              "smoke_contract ota_train_smoke_noturncap")
    return _run(["bash", "-c", script], tmp_path, PRL_ROOT=str(ROOT), PRL_SITE="_test_qs_local", MODEL="qwen3.5-9B",
                JOB_ID="7")


def test_a_smoke_checks_its_tasks_once_when_the_snapshot_has_no_result(tmp_path: Path) -> None:
    snapshot = tmp_path / "ext/task_snapshots/otagent728"
    failed = _contract(tmp_path, 1)
    assert failed.returncode != 0 and not (snapshot / "contract").exists()
    out = _contract(tmp_path, 0)
    assert out.returncode == 0, out.stderr
    assert (snapshot / "contract/summary.json").exists() and not (snapshot / "contract.partial").exists()
    calls = (tmp_path / "apptainer.log").read_text().splitlines()
    assert len(calls) == 2 and calls[-1].startswith(f"{snapshot}/contract.partial ")
    assert f"PRL_TASK_PROMPTS={snapshot}/smoke.jsonl" in calls[-1]
    again = _contract(tmp_path, 1)  # the result is there: nothing runs
    assert again.returncode == 0 and len((tmp_path / "apptainer.log").read_text().splitlines()) == 2


def test_quickstart_scripts_are_plain_bash() -> None:
    for script in ("scripts/quickstart.sh", "scripts/local/submit.sh", "scripts/local/job.sh",
                   "scripts/setup/build_sandboxes.sh"):
        assert (ROOT / script).read_text().startswith("#!/bin/bash\n"), script


FAKE_E2B = '''
import os
class Template:
    def from_image(self, image, username, password):
        self.image = image
        return self
    @staticmethod
    def build(template, name, cpu_count, memory_mb, api_key, api_url, domain):
        if os.environ.get("FAIL_IMAGE") and os.environ["FAIL_IMAGE"] in template.image:
            raise RuntimeError("build failed")
        with open(os.environ["CALLS"], "a") as f:
            f.write(f"{template.image} {name} {cpu_count} {memory_mb} {api_key} {api_url} {domain}\\n")
        return type("Built", (), {"template_id": "tpl-" + template.image.rsplit("/", 1)[1], "build_id": "b1"})()
'''


def _register(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    (tmp_path / "e2b").mkdir(exist_ok=True)
    (tmp_path / "e2b" / "__init__.py").write_text(FAKE_E2B)
    docker = tmp_path / "docker"
    docker.write_text(f'#!/bin/bash\necho "$*" >> {tmp_path}/docker.log\n')
    docker.chmod(0o755)
    build = {"run_id": "20261118T000000-ABC", "tasks": {
        "task_1": {"status": "built", "image_tag": "harbor-task:h1", "context_hash": "h1"},
        "task_2": {"status": "built", "image_tag": "harbor-task:h2", "context_hash": "h2"},
        "task_3": {"status": "failed", "image_tag": "harbor-task:h3", "context_hash": "h3"}}}
    (tmp_path / "build.json").write_text(json.dumps(build))
    (tmp_path / "key").write_text("secret\n")
    (tmp_path / "creds.json").write_text('{"username": "u", "password": "p"}')
    return subprocess.run(["python3", str(ROOT / "scripts/tools/register_templates.py")], capture_output=True,
                          text=True, check=False, env={
        "PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path), "CALLS": str(tmp_path / "calls.log"),
        "PRL_BUILD_MANIFEST": str(tmp_path / "build.json"), "PRL_SNAPSHOT_MANIFEST": str(tmp_path / "manifest.json"),
        "PRL_REGISTRY": "registry.example/proj/", "PRL_REGISTRY_CREDENTIALS_FILE": str(tmp_path / "creds.json"),
        "E2B_API_URL": "https://e2b.example", "E2B_DOMAIN": "e2b.example", "E2B_API_KEY_FILE": str(tmp_path / "key"),
        "PRL_DOCKER_COMMAND": str(docker), "PRL_IMAGE_PRUNE": "1", "PRL_TEMPLATE_RETRY_WAIT": "0", **env})


def test_register_templates_pushes_registers_and_resumes(tmp_path: Path) -> None:
    out = _register(tmp_path)
    assert out.returncode == 0, out.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["api_url"] == "https://e2b.example" and manifest["builds"] == ["20261118T000000-ABC"]
    assert sorted(manifest["tasks"]) == ["task_1", "task_2"]  # a failed build is not registered
    row = manifest["tasks"]["task_1"]
    assert (row["status"], row["context_hash"], row["registry_image"]) == (
        "ready", "h1", "registry.example/proj/task_1:20261118t000000-abc")
    assert row["template_id"] == "tpl-task_1:20261118t000000-abc"
    docker = (tmp_path / "docker.log").read_text()
    assert "tag harbor-task:h1 registry.example/proj/task_1:20261118t000000-abc" in docker
    assert "push registry.example/proj/task_1:20261118t000000-abc" in docker and "rmi harbor-task:h1" in docker
    assert " 2 2048 secret https://e2b.example e2b.example" in (tmp_path / "calls.log").read_text()
    again = _register(tmp_path)
    assert again.returncode == 0 and "0 of 3 tasks to register" in again.stdout


def test_register_templates_reports_a_task_that_keeps_failing(tmp_path: Path) -> None:
    out = _register(tmp_path, FAIL_IMAGE="task_2")
    assert out.returncode == 1 and "FAILED task_2: RuntimeError: build failed" in out.stderr
    assert sorted(json.loads((tmp_path / "manifest.json").read_text())["tasks"]) == ["task_1"]
    other = _register(tmp_path, E2B_API_URL="https://other.example")
    assert other.returncode != 0 and "belongs to https://e2b.example" in other.stderr


def test_a_transform_smoke_chains_after_download_and_convert(tmp_path: Path) -> None:
    out = _run(["bash", "scripts/submit.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1",
               CHAIN="download+convert+smoke_grpo", TRANSFORM="atlas")
    assert out.returncode == 0, out.stderr
    assert "PRL_CHAIN=download+convert+smoke_grpo" in out.stdout and "ARM=ota_train_smoke_transform" in out.stdout
    both = _run(["bash", "scripts/submit.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1",
                CHAIN="smoke_grpo+train_eval", TRANSFORM="atlas", ARM="ota_train_transform", PRL_RUN_ID="x")
    assert both.returncode != 0 and "run alone" in both.stderr


def test_task_check_without_a_manifest_checks_the_smoke_tasks(tmp_path: Path) -> None:
    out = _run(["bash", "scripts/submit.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1", CHAIN="task_check")
    assert out.returncode == 0 and "PRL_TEMPLATE_MANIFEST" not in out.stdout, out.stderr
    half = _run(["bash", "scripts/submit.sh"], tmp_path, PRL_SITE="_test_qs_pbs", DRY_RUN="1", CHAIN="task_check",
                PRL_TASK_ROOT="/x")
    assert half.returncode != 0 and "task_check checks PRL_TEMPLATE_MANIFEST" in half.stderr
