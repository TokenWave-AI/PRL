"""Multi-node chain jobs (the 2-node MoE smoke): the arm, the node count a submission asks for under each scheduler,
and the head starting its worker nodes (scripts/lib/chain.sh start_workers) with a fake remote start. Bash only.
The submissions use two throwaway site profiles (configs/sites/_test_*.env, ignored by git), one per scheduler."""

from __future__ import annotations

import os
import socket
import stat
import subprocess
from pathlib import Path

import pytest
from test_ppo_config import ROOT, arm_env

MOE = "ota_train_smoke_moe"
SITES = {
    "_test_pjm": "PRL_SCHEDULER=pjm\nPRL_PJM_RSCGRP=gpu\nPRL_PJM_MULTINODE_HOURS=96\n",
    "_test_pbs": "PRL_SCHEDULER=pbs\nPRL_PBS_QUEUE=gpu\n",
}
COMMON = ("PRL_EXT=${PRL_EXT:-/nonexistent/ext}\nPRL_SCRATCH=${PRL_SCRATCH:-/tmp/scratch_test}\nHF_HOME=/nonexistent/hf\n"
          "MILES_IMAGE=/nonexistent/miles.sif\nPRL_NODE_FILE=${PRL_NODE_FILE:-}\nPRL_REMOTE_EXEC=${PRL_REMOTE_EXEC:-}\n"
          "PRL_NODE_IFACE=${PRL_NODE_IFACE:-ib0}\n")


@pytest.fixture(autouse=True, scope="module")
def _sites():
    paths = [ROOT / "configs" / "sites" / f"{name}.env" for name in SITES]
    for path, body in zip(paths, SITES.values()):
        path.write_text(body + COMMON)
    yield
    for path in paths:
        path.unlink(missing_ok=True)


def test_moe_arm_is_colocated_on_two_nodes() -> None:
    arm = arm_env(MOE)
    assert (arm["NNODES"], arm["NUM_GPUS"], arm["COLOCATE"], arm["PRL_ROLLOUT_ON_HEAD"]) == ("2", "4", "1", "0")
    extra = arm["EXTRA_ARGS"].split()
    for flag in ("--expert-model-parallel-size 8", "--mtp-num-layers 0", "--use-rollout-routing-replay",
                 "--pause-generation-mode retract", "--dump-without-routed-experts"):
        assert flag in arm["EXTRA_ARGS"], flag
    assert "--pin-rollout-manager-to-head" in extra and "flashinfer_cutlass" not in arm["EXTRA_ARGS"]  # H100, not GB200
    assert (arm["MAX_CONTEXT"], arm["PRL_FUSED_LOGPROB_CHUNK"]) == ("65536", "512")


def _submit(site: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "scripts/submit.sh"], cwd=ROOT, capture_output=True, text=True, check=False,
                          env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "DRY_RUN": "1", "PRL_SUBMIT_FINDS_SMOKE": "0",
                               "PRL_SITE": site, **env})


@pytest.mark.parametrize("env, nodes, elapse", [
    ({"CHAIN": "download+convert+smoke", "SMOKE_ARM": MOE, "MODEL": "qwen3.6-35B-A3B"}, 2, "08:00:00"),
    ({"CHAIN": "train_eval", "ARM": MOE, "MODEL": "qwen3.6-35B-A3B", "PRL_RUN_ID": "x"}, 2, "96:00:00"),
    ({"CHAIN": "smoke_grpo", "TRANSFORM": "reroll_smoke"}, 1, "08:00:00"),
    ({"CHAIN": "train_eval", "ARM": "ota_train_transform_reroll", "TRANSFORM": "x",
      "PRL_RUN_ID": "x"}, 1, "168:00:00"),
])
def test_pjm_asks_for_the_arms_nodes(env: dict, nodes: int, elapse: str) -> None:
    out = _submit("_test_pjm", **env)
    assert out.returncode == 0, out.stderr
    assert f"node={nodes}\\,elapse={elapse}" in out.stdout, out.stdout


def test_eval_hf_evaluates_an_exported_checkpoint_alone() -> None:
    """A re-evaluation without the training: the checkpoint reaches the job."""
    out = _submit("_test_pjm", CHAIN="eval_hf", MODEL_HF_DIR="/ext/ckpt/hf/run_iter_0000067", ELAPSE="12:00:00")
    assert out.returncode == 0, out.stderr
    for var in ("PRL_CHAIN=eval_hf", "MODEL=qwen3.5-9B", "MODEL_HF_DIR=/ext/ckpt/hf/run_iter_0000067",
                "EVAL_ARM=ota_eval_tbdev70_64k_summary_c16_noturncap"):
        assert var in out.stdout.split(" -x ")[1].split(" ")[0].split("\\,"), (var, out.stdout)
    assert "node=1\\,elapse=12:00:00" in out.stdout
    missing = _submit("_test_pjm", CHAIN="eval_hf")
    assert missing.returncode != 0 and "MODEL_HF_DIR" in missing.stderr


def test_pbs_refuses_a_multi_node_arm() -> None:
    out = _submit("_test_pbs", CHAIN="download+convert+smoke", SMOKE_ARM=MOE, MODEL="qwen3.6-35B-A3B")
    assert out.returncode != 0 and "one node per job" in out.stderr


def _start(tmp_path: Path, nodes: list[str]) -> tuple[subprocess.CompletedProcess, Path, Path]:
    (tmp_path / "nodes").write_text("\n".join(nodes + nodes[:1]) + "\n")  # PJM may list a node twice
    log = tmp_path / "remote.log"
    fake = tmp_path / "remote"
    fake.write_text(f'#!/bin/bash\necho "$@" >> {log}\n')
    fake.chmod(0o755)
    script = ("set -euo pipefail; set -a; source configs/sites/" + "_test_pjm.env; set +a; source scripts/lib/chain.sh; "
              f"start_workers {MOE}; wait")
    out = subprocess.run(["bash", "-c", script], cwd=ROOT, capture_output=True, text=True, check=False, env={
        "PATH": "/usr/bin:/bin:/usr/sbin", "HOME": "/tmp", "PRL_ROOT": str(ROOT), "PRL_SITE": "_test_pjm",
        "MODEL": "qwen3.6-35B-A3B", "JOB_ID": "42", "PRL_EXT": str(tmp_path / "ext"),
        "PRL_SCRATCH": str(tmp_path / "ssd"), "PRL_NODE_FILE": str(tmp_path / "nodes"), "PRL_REMOTE_EXEC": str(fake),
        "PRL_NODE_IFACE": "lo"})
    return out, log, tmp_path / "ext" / "runs" / "grpo_ota_train_smoke_42"


# start_workers runs on the host before the container starts, and reads the node's addresses with /usr/sbin/ip;
# the Miles image has no iproute2 (CHAIN=unit runs these tests inside it).
needs_host_ip = pytest.mark.skipif(not os.path.exists("/usr/sbin/ip"), reason="no /usr/sbin/ip (inside the Miles image)")


@needs_host_ip
@pytest.mark.parametrize("head, worker", [
    ("127.0.0.1", "172.16.17.16"),  # a node file of addresses; the test's head address is lo's
    (socket.gethostname().split(".")[0], "b0002"),  # a site listing host names
])
def test_the_head_prepares_then_starts_its_workers(tmp_path: Path, head: str, worker: str) -> None:
    out, log, run = _start(tmp_path, [head, worker])
    assert out.returncode == 0, out.stderr
    script = ROOT / "scripts" / "lib" / "chain_worker.sh"
    assert log.read_text().splitlines() == [f"{worker} bash {script} prepare {run}",
                                            f"{worker} bash {script} join {run} 127.0.0.1"]
    env = (run / "worker.env").read_text()
    assert stat.S_IMODE(os.stat(run / "worker.env").st_mode) == 0o600
    assert 'NNODES="2"' in env and 'PRL_FUSED_LOGPROB_CHUNK="512"' in env and f'PRL_SCRATCH="{tmp_path / "ssd"}"' in env
    assert "E2B_" not in env and "KEY" not in env and "HARBOR_" not in env  # the trials stay on the head


def test_a_job_with_other_nodes_than_the_arm_stops(tmp_path: Path) -> None:
    out, log, _ = _start(tmp_path, [socket.gethostname().split(".")[0]])
    assert out.returncode != 0 and "wants 2 nodes" in out.stderr and not log.exists()


@needs_host_ip
def test_a_head_missing_from_the_node_list_stops(tmp_path: Path) -> None:
    out, log, _ = _start(tmp_path, ["172.16.17.6", "172.16.17.16"])
    assert out.returncode != 0 and "is not in" in out.stderr and not log.exists()


def _pinned_sockets(**env: str) -> dict[str, str]:
    """The block of scripts/lib/container_main.sh that pins NCCL's and gloo's sockets, run on its own."""
    lines = (ROOT / "scripts" / "lib" / "container_main.sh").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('if [ "$NNODES" -gt 1 ] && [ -n "${PRL_NODE_IFACE'))
    block = "\n".join(lines[start:lines.index("fi", start) + 1])
    out = subprocess.run(["bash", "-c", block + '\necho "$NCCL_SOCKET_IFNAME|$GLOO_SOCKET_IFNAME"'], capture_output=True,
                         text=True, check=True, env={"PATH": "/usr/bin:/bin", **env})
    nccl, gloo = out.stdout.strip().split("|")
    return {"nccl": nccl, "gloo": gloo}


@pytest.mark.parametrize("env, want", [
    ({"NNODES": "2", "PRL_NODE_IFACE": "ib0"}, {"nccl": "ib0", "gloo": "ib0"}),  # multi-node over InfiniBand
    ({"NNODES": "1", "PRL_NODE_IFACE": "ib0"}, {"nccl": "", "gloo": ""}),  # one node: NCCL's own choice
    ({"NNODES": "2"}, {"nccl": "", "gloo": ""}),  # a site naming no interface: unchanged
    ({"NNODES": "2", "PRL_NODE_IFACE": "ib0", "NCCL_SOCKET_IFNAME": "ib1"}, {"nccl": "ib1", "gloo": "ib0"}),
])
def test_multi_node_pins_nccl_and_gloo_to_the_sites_interface(env: dict, want: dict) -> None:
    assert _pinned_sockets(**env) == want


def test_the_interface_reaches_the_head_container_and_the_workers() -> None:
    assert "RAY_HEAD_IP PRL_NODE_IFACE NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME" in (ROOT / "scripts" / "run.sh").read_text()
    worker = (ROOT / "scripts" / "lib" / "chain_worker.sh").read_text()
    assert "NCCL_SOCKET_IFNAME='${NCCL_SOCKET_IFNAME:-$PRL_NODE_IFACE}'" in worker
    assert "GLOO_SOCKET_IFNAME='${GLOO_SOCKET_IFNAME:-$PRL_NODE_IFACE}'" in worker


def _proxies(tmp_path, **env: str) -> list[str]:
    """The block of scripts/lib/container_main.sh that names an http:// proxy by its address, run on its own with a
    fake getent (proxy.site -> 10.0.0.9; anything else unknown)."""
    lines = (ROOT / "scripts" / "lib" / "container_main.sh").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("for v in https_proxy HTTPS_PROXY"))
    block = "\n".join(lines[start:lines.index("done", start) + 1])
    fake = tmp_path / "getent"
    fake.write_text('#!/bin/bash\n[ "$2" = proxy.site ] && echo "10.0.0.9      STREAM proxy.site"\n')
    fake.chmod(0o755)
    out = subprocess.run(["bash", "-c", "set -euo pipefail\n" + block + '\necho "${https_proxy:-}|${http_proxy:-}|${HTTPS_PROXY:-}"'],
                         capture_output=True, text=True, check=True, env={"PATH": f"{tmp_path}:/usr/bin:/bin", **env})
    return out.stdout.strip().split("|")


def test_an_http_proxy_is_named_by_its_address(tmp_path) -> None:
    assert _proxies(tmp_path, https_proxy="http://proxy.site:3128", http_proxy="http://u:p@proxy.site:3128/") == [
        "http://10.0.0.9:3128", "http://u:p@10.0.0.9:3128/", ""]
    # unknown names, addresses already, https:// proxies (TLS to the proxy needs its name) and no proxy stay
    assert _proxies(tmp_path, https_proxy="http://other.site:8080", HTTPS_PROXY="https://proxy.site:443") == [
        "http://other.site:8080", "", "https://proxy.site:443"]
    assert _proxies(tmp_path, https_proxy="http://172.23.1.9:3128") == ["http://172.23.1.9:3128", "", ""]
    assert _proxies(tmp_path) == ["", "", ""]
