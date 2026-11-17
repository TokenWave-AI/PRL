"""prl/rollout/harbor_fixes.py against a real sandbox (no model).

Uses one TB-dev task without tmux in its image (broken-python by default), as an eval trial would:
  1. Harbor's TmuxSession.start installs tmux (with the retry wrapper) and the session works.
  2. A command whose stream is dropped mid-run (handle.disconnect()) is waited for over a new stream to its pid:
     full exit code, output after the drop, run once.
Env: E2B_API_KEY_FILE, E2B_API_URL, E2B_DOMAIN; PRL_TEMPLATE_MANIFEST, PRL_TASK_ROOT; PRL_CHECK_TASK.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path, PurePosixPath

import prl.rollout.harbor_fixes as fx
from prl.sandbox.template_manifest import resolve_template


async def main() -> None:
    from harbor.agents.terminus_2.tmux_session import TmuxSession
    from harbor.environments.e2b import E2BEnvironment
    from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
    from harbor.models.trial.paths import TrialPaths

    os.environ.setdefault("E2B_API_KEY", Path(os.environ["E2B_API_KEY_FILE"]).read_text().strip())
    task = Path(os.environ["PRL_TASK_ROOT"]) / os.getenv("PRL_CHECK_TASK", "broken-python")
    template = resolve_template(os.environ["PRL_TEMPLATE_MANIFEST"], task, os.environ["E2B_API_URL"])
    results: list[dict] = []

    async def check(name, fn):
        start = time.time()
        try:
            detail = await fn()
            results.append({"name": name, "ok": True, "detail": detail, "seconds": round(time.time() - start, 1)})
        except Exception as e:  # noqa: BLE001 -- each check records its failure; the others still run
            results.append({"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"[:400]})
        print(json.dumps(results[-1]), flush=True)

    with tempfile.TemporaryDirectory(prefix="sandbox-fixes-") as directory:
        paths = TrialPaths(trial_dir=Path(directory) / "trial")
        paths.mkdir()
        env = E2BEnvironment(
            environment_dir=task / "environment", environment_name=task.name, session_id="fixes",
            trial_paths=paths, task_env_config=EnvironmentConfig(),
            network_policy=NetworkPolicy(network_mode=NetworkMode.PUBLIC),
            prebuilt_template_id=template, sandbox_timeout_secs=900, renew_interval=0,
        )
        await env.start(force_build=False)
        try:
            session = TmuxSession(
                session_name="terminus-2", environment=env, logging_path=PurePosixPath("/logs/agent/terminus_2.pane"),
                local_asciinema_recording_path=None, remote_asciinema_recording_path=None,
            )

            async def tmux_installed():
                assert (await env.exec("tmux -V", user="root")).return_code != 0, "image already has tmux"
                await env.exec("mkdir -p /logs/agent")
                await session.start()
                await session.send_keys(["echo shell-alive", "Enter"], block=True, max_timeout_sec=30)
                assert "shell-alive" in await session.capture_pane()
                return (await env.exec("tmux -V")).stdout.strip()

            async def dropped_stream():
                handle = await env._dispatch_command(
                    "echo before; sleep 8; echo after >&2; echo once >> /tmp/exec-runs; exit 7",
                    cwd=None, env=None, timeout_sec=60, user="root",
                )
                await asyncio.sleep(2)
                await handle.disconnect()  # what a gateway drop leaves: the process runs on, the stream is gone
                result = await fx._reconnect(env, handle, 60, Exception("Command ended without an end event"))
                runs = (await env.exec("wc -l < /tmp/exec-runs")).stdout.strip()
                assert (result.return_code, "after" in result.stderr, runs) == (7, True, "1"), (result, runs)
                return {"return_code": result.return_code, "stderr": result.stderr.strip(), "runs": runs}

            await check("tmux_install", tmux_installed)
            await check("dropped_stream_reconnect", dropped_stream)
        finally:
            await env.stop(delete=True)
    assert all(r["ok"] for r in results), results


if __name__ == "__main__":
    asyncio.run(main())
