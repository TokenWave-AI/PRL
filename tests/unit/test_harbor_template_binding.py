"""Per-task template selection must not leak across trials or hide cloud errors."""
from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
import sys
import types

import pytest

from prl.sandbox.template_manifest import context_hash, context_inventory


@pytest.fixture
def wrapper(monkeypatch, tmp_path):
    @dataclasses.dataclass
    class Binding:
        kwargs: object

    stub = types.ModuleType("harbor_agent_function")
    stub.HARNESS_BINDINGS = {k: Binding(lambda *a: {"llm_call_kwargs": {}}) for k in ("terminus", "terminus-2")}

    def config(metadata, *_):
        return types.SimpleNamespace(
            task=types.SimpleNamespace(path=tmp_path / metadata["instance_id"]),
            environment=types.SimpleNamespace(kwargs=json.loads(__import__("os").environ["HARBOR_ENV_KWARGS"])),
            agent=types.SimpleNamespace(name="terminus-2", import_path=None, kwargs={}),
        )

    stub.build_trial_config = config
    stub.trial_result_to_metadata = lambda result: {"reward": 0.0}

    async def run(*args, **kwargs):
        return {"reward": 0.0, "exit_status": "AgentError", "trial_dir": "/saved/trial"}

    stub.run = run
    monkeypatch.setitem(sys.modules, "harbor_agent_function", stub)
    name = "prl.rollout.harbor_agent"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


def test_per_task_binding_does_not_change_shared_options(wrapper, monkeypatch, tmp_path):
    rows = {}
    for task in ("a", "b"):
        env = tmp_path / task / "environment"
        env.mkdir(parents=True)
        (env / "Dockerfile").write_text("FROM ubuntu:24.04\n")
        rows[task] = {"status": "ready", "template_id": task, "context_hash": context_hash(context_inventory(env))}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"api_url": "https://region.example", "tasks": rows}))
    options = {"prl_template_manifest": str(manifest), "prl_max_concurrent_trials": 2}
    monkeypatch.setenv("HARBOR_ENV_KWARGS", json.dumps(options))
    monkeypatch.setenv("E2B_API_URL", "https://region.example")
    configs = [wrapper._trial_config({"instance_id": task}, "url", {}) for task in ("a", "b", "a")]
    assert [c.environment.kwargs for c in configs] == [{"prebuilt_template_id": t} for t in ("a", "b", "a")]


@pytest.mark.parametrize("evaluation", [False, True])
def test_training_adapter_selects_subclass_and_limits_probe_to_train(wrapper, monkeypatch, evaluation):
    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"prl_force_summary_once": true}')
    config = wrapper._trial_config(
        {"instance_id": "a", "prl_training_contract": True, "prl_evaluation": evaluation}, "url", {},
    )
    assert config.agent.name is None
    assert config.agent.import_path == "prl.rollout.training_terminus:TrainingTerminus2"
    assert config.agent.kwargs["force_summary_once"] is (not evaluation)
    assert config.environment.kwargs == {}


def test_cloud_agent_error_is_not_a_valid_zero_reward(wrapper, monkeypatch):
    monkeypatch.setenv("HARBOR_ENV_KWARGS", json.dumps({"prl_template_manifest": "/manifest"}))
    with pytest.raises(RuntimeError, match="/saved/trial"):
        asyncio.run(wrapper.run())


def test_verifier_timeout_is_infrastructure_not_policy_failure(wrapper, monkeypatch):
    monkeypatch.setenv("HARBOR_ENV_KWARGS", json.dumps({"prl_template_manifest": "/manifest"}))
    result = types.SimpleNamespace(
        exception_info=types.SimpleNamespace(exception_type="VerifierTimeoutError"), trial_uri="/saved/trial",
    )
    with pytest.raises(RuntimeError, match="VerifierTimeoutError"):
        wrapper._trial_metadata(result)
    result.exception_info.exception_type = "AgentTimeoutError"
    assert wrapper._trial_metadata(result) == {"reward": 0.0}
