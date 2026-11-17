"""Validate a remote E2B provider before running Harbor model trials.

Run explicitly with the Harbor dependencies on PYTHONPATH. This creates short-lived
sandboxes, runs SDK and real Harbor/TmuxSession contracts, and deletes its sandboxes.
It does not run a model or grade benchmark tasks. Exit nonzero on any failed contract;
write configuration, package versions, timings and failures as JSON for the report.

Configuration:
    E2B_API_KEY_FILE: required private file containing the plain API key.
    E2B_API_URL / E2B_DOMAIN: required explicit cloud endpoint and sandbox domain.
    PRL_SANDBOX_SMOKE_TEMPLATE: existing template, default code-interpreter-v1.
    PRL_SANDBOX_SMOKE_OUTPUT: required output JSON path.
    PRL_SANDBOX_SMOKE_HARBOR: 1 (default) also checks Harbor and Terminus-2 tmux.
    PRL_SANDBOX_SMOKE_LIFETIME: sandbox expiry in seconds, default 600.

The Harbor checks use an existing cloud template and a synthetic /tmp workdir. They cannot establish task-image parity.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import platform
import socket
import tempfile
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from e2b import AsyncSandbox, TimeoutException
from e2b.sandbox.commands.command_handle import CommandExitException


async def main() -> bool:
    key_path = Path(os.environ["E2B_API_KEY_FILE"])
    if key_path.stat().st_mode & 0o077:
        raise ValueError("API key file must not be readable by group or others")
    key = key_path.read_text().strip()
    if not key or "\n" in key or key.startswith("export "):
        raise ValueError("API key file must contain one plain key")
    api_url = os.environ["E2B_API_URL"]
    domain = os.environ["E2B_DOMAIN"]
    if not api_url.startswith("https://"):
        raise ValueError("Cloud smoke requires an explicit HTTPS API URL")
    os.environ["E2B_API_KEY"] = key
    # A leftover local-backend URL would redirect envd requests away from the service.
    os.environ.pop("E2B_SANDBOX_URL", None)
    template = os.getenv("PRL_SANDBOX_SMOKE_TEMPLATE", "code-interpreter-v1")
    lifetime = int(os.getenv("PRL_SANDBOX_SMOKE_LIFETIME", "600"))
    output = Path(os.environ["PRL_SANDBOX_SMOKE_OUTPUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "job_id": os.getenv("JOB_ID"),
        "python": platform.python_version(),
        "config": {"api_url": api_url, "domain": domain, "template": template, "lifetime_sec": lifetime},
        "versions": {},
        "checks": [],
        "sandbox_ids": [],
    }
    for package in ("e2b", "harbor", "pyqwest", "connectrpc", "httpx", "h2"):
        try:
            result["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["versions"][package] = None

    def save() -> None:
        output.write_text(json.dumps(result, indent=2) + "\n")

    async def check(name: str, operation: Callable[[], Awaitable[Any]]) -> Any:
        start = time.monotonic()
        row: dict[str, Any] = {"name": name}
        try:
            value = await operation()
            row["ok"] = True
            return value
        except Exception as exc:
            row.update(ok=False, error_type=type(exc).__name__, error=str(exc).replace(key, "[REDACTED]")[:2000])
            raise
        finally:
            row["seconds"] = round(time.monotonic() - start, 3)
            result["checks"].append(row)
            save()
            print(json.dumps(row), flush=True)

    async def create() -> AsyncSandbox:
        sandbox = await AsyncSandbox.create(
            template=template, timeout=lifetime, api_key=key, api_url=api_url,
            domain=domain, request_timeout=30, metadata={"purpose": "environment-contract"},
        )
        result["sandbox_ids"].append(sandbox.sandbox_id)
        save()
        return sandbox

    async def delete(sandbox: AsyncSandbox) -> None:
        await sandbox.kill()
        # Some services' envd /health returns HTTP 500 after deletion, whereas the e2b SDK's
        # is_running expects 502. Verify control-plane absence instead.
        for _ in range(5):
            pages = AsyncSandbox.list(api_key=key, api_url=api_url, domain=domain, request_timeout=30)
            present = False
            while pages.has_next:
                present |= any(item.sandbox_id == sandbox.sandbox_id for item in await pages.next_items())
            if not present:
                return
            await asyncio.sleep(1)
        raise AssertionError("Sandbox still listed after kill")

    async def sdk_contract() -> None:
        sandbox = await check("sdk_create", create)
        try:
            async def commands() -> None:
                r = await sandbox.commands.run("printf out; printf err >&2", timeout=20)
                assert (r.stdout, r.stderr, r.exit_code) == ("out", "err", 0)
                try:
                    await sandbox.commands.run("exit 7", timeout=20)
                except CommandExitException as exc:
                    assert exc.exit_code == 7
                else:
                    raise AssertionError("Nonzero command exit was lost")

            async def files_and_isolation() -> None:
                await sandbox.files.write("/tmp/contract.bin", b"\x00\xffcontract\n")
                assert bytes(await sandbox.files.read("/tmp/contract.bin", format="bytes")) == b"\x00\xffcontract\n"
                other = await create()
                try:
                    assert not await other.files.exists("/tmp/contract.bin")
                finally:
                    await check("isolation_sandbox_delete", lambda: delete(other))

            async def timeout() -> None:
                try:
                    await sandbox.commands.run("sleep 5", timeout=1)
                except TimeoutException:
                    pass
                else:
                    raise AssertionError("Long command did not raise TimeoutException")
                assert (await sandbox.commands.run("printf alive", timeout=10)).stdout == "alive"

            await check("sdk_commands_and_exit_code", commands)
            await check("sdk_binary_files_and_isolation", files_and_isolation)
            await check("sdk_command_timeout_and_recovery", timeout)
            await check("sdk_renew", lambda: sandbox.set_timeout(lifetime))
        finally:
            await check("sdk_delete", lambda: delete(sandbox))

    async def harbor_contract() -> None:
        from harbor.agents.terminus_2.tmux_session import TmuxSession
        from harbor.environments.e2b import E2BEnvironment
        from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
        from harbor.models.trial.paths import TrialPaths

        with tempfile.TemporaryDirectory(prefix="sandbox-contract-") as directory:
            root = Path(directory)
            env_dir = root / "environment"
            env_dir.mkdir()
            (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /tmp\n")
            paths = TrialPaths(trial_dir=root / "trial")
            paths.mkdir()
            env = E2BEnvironment(
                environment_dir=env_dir, environment_name="sandbox-contract", session_id="smoke",
                trial_paths=paths, task_env_config=EnvironmentConfig(),
                network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
                prebuilt_template_id=template, sandbox_timeout_secs=lifetime, renew_interval=0,
            )
            try:
                await check("harbor_start", lambda: env.start(force_build=False))
                sandbox = env._sandbox
                assert sandbox is not None
                result["sandbox_ids"].append(sandbox.sandbox_id)

                async def roundtrip() -> None:
                    r = await env.exec("pwd; printf $PRL_MARKER", env={"PRL_MARKER": "marker"})
                    assert (r.return_code, r.stdout) == (0, "/tmp\nmarker")
                    r = await env.exec("printf error >&2; exit 3")
                    assert (r.return_code, r.stderr) == (3, "error")
                    source = root / "upload"
                    (source / "sub").mkdir(parents=True)
                    (source / "sub" / "payload").write_bytes(b"\x00\xffroundtrip")
                    await env.upload_dir(source, "/tmp/contract-upload")
                    await env.download_dir("/tmp/contract-upload", root / "download")
                    assert (root / "download/sub/payload").read_bytes() == b"\x00\xffroundtrip"

                async def tmux() -> None:
                    await env.exec("mkdir -p /logs/agent")
                    session = TmuxSession(
                        session_name="terminus-2", environment=env,
                        logging_path=PurePosixPath("/logs/agent/terminus_2.pane"),
                        local_asciinema_recording_path=None, remote_asciinema_recording_path=None,
                    )
                    await session.start()
                    await session.send_keys(["export PRL_PERSIST=42", "Enter"], block=True, max_timeout_sec=30)
                    await session.send_keys(["echo persist-$PRL_PERSIST", "Enter"], block=True, max_timeout_sec=30)
                    assert await session.is_session_alive()
                    assert "persist-42" in await session.capture_pane()
                    await session.stop()

                await check("harbor_exec_and_directory_roundtrip", roundtrip)
                await check("harbor_terminus2_tmux_persistence", tmux)
            finally:
                # Harbor.stop logs deletion errors; explicitly verify deletion to avoid a false pass.
                sandbox = env._sandbox
                await env.stop(delete=True)
                if sandbox is not None:
                    await check("harbor_delete_verified", lambda: delete(sandbox))

    try:
        await sdk_contract()
        if os.getenv("PRL_SANDBOX_SMOKE_HARBOR", "1") == "1":
            await harbor_contract()
        result["passed"] = True
    except Exception as exc:  # noqa: BLE001 -- Persist failures and return a nonzero process status.
        result.update(passed=False, failure_type=type(exc).__name__, failure=str(exc).replace(key, "[REDACTED]")[:2000])
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return result["passed"]


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main()) else 1)
