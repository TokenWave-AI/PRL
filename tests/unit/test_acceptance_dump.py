"""The smoke acceptance reads Miles' rollout dumps (traces/rollout_data/<n>.pt, torch.save zip archives) without
building objects from them (scripts/checks/grpo_acceptance.py PlainDataUnpickler); with R3 (MoE models) a sample
also carries rollout_routed_experts, a numpy array, kept as an OpaqueArray with its shape."""

from __future__ import annotations

import io
import pickle
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "checks"))
from grpo_acceptance import OpaqueArray, PlainDataUnpickler, load_samples

np = pytest.importorskip("numpy")


def _dump(tmp_path: Path, samples: list[dict]) -> Path:
    """A rollout dump as torch.save writes it: a zip archive with <name>/data.pkl."""
    (tmp_path / "traces" / "rollout_data").mkdir(parents=True)
    with zipfile.ZipFile(tmp_path / "traces" / "rollout_data" / "0.pt", "w") as archive:
        archive.writestr("0/data.pkl", pickle.dumps({"samples": samples}, protocol=2))
    return tmp_path


def test_routed_experts_are_read_as_their_shape(tmp_path):
    experts = np.zeros((4, 40, 8), dtype=np.int32)  # tokens - 1 rows x layers x top-k
    run = _dump(tmp_path, [{"tokens": [1, 2, 3, 4, 5], "rollout_routed_experts": experts, "loss_mask": [1]}])
    sample, = load_samples(run, 0)
    assert isinstance(sample["rollout_routed_experts"], OpaqueArray)
    assert sample["rollout_routed_experts"].shape == (4, 40, 8)
    assert sample["tokens"] == [1, 2, 3, 4, 5]


def test_other_objects_are_refused():
    class Other:
        pass
    globals()["Other"] = Other  # picklable by reference
    Other.__qualname__ = "Other"
    with pytest.raises(ValueError, match="Unexpected object"):
        PlainDataUnpickler(io.BytesIO(pickle.dumps({"x": Other()}))).load()
