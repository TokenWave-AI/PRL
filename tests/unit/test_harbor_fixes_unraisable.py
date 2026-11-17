"""harbor_fixes (4): an unawaited litellm coroutine is logged; any other unawaited coroutine reaches Miles' hook."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

harbor_fixes = pytest.importorskip("prl.rollout.harbor_fixes")


def test_only_litellm_coroutines_are_tolerated():
    seen = []
    hook = harbor_fixes._tolerate_litellm(seen.append)
    litellm = SimpleNamespace(exc_value=RuntimeWarning("coroutine 'OpenAIChatCompletion.acompletion' was never awaited"))
    other = SimpleNamespace(exc_value=RuntimeWarning("coroutine 'Trainer.step' was never awaited"))
    hook(litellm)
    hook(other)
    assert seen == [other]
    assert harbor_fixes._tolerate_litellm(hook) is hook  # wrapping twice is a no-op


def test_stack_dumps_go_to_the_run_dir(tmp_path, monkeypatch):
    import faulthandler
    import time

    monkeypatch.setattr(harbor_fixes, "_stack_dump_file", None)
    path = harbor_fixes.start_stack_dumps(str(tmp_path), every=0.05)
    try:
        assert path is not None and path.startswith(str(tmp_path / "stack_dumps"))
        assert harbor_fixes.start_stack_dumps(str(tmp_path)) is None  # once per process
        for _ in range(100):
            if "test_stack_dumps_go_to_the_run_dir" in Path(path).read_text():
                break
            time.sleep(0.05)
        text = Path(path).read_text()
        assert "test_stack_dumps_go_to_the_run_dir" in text and "--- thread" in text
    finally:
        faulthandler.disable()
