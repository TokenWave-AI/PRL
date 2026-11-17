"""Make the recorded trials of the transform pre-run check (prl/transforms/check.py) from training batches.

    python scripts/tools/make_transform_fixture.py OUT.json.gz TOKENIZER.json BATCH.pt:TRIAL[,TRIAL...] [...]

BATCH.pt is a run's traces/rollout_data/<update>.pt (the samples Miles trained on; read without torch or Miles: the
batch is plain pickled data). TRIAL is a trial directory's name. Each trial is stored as text: per leaf its node id,
prompt and response, the response as alternating trained (model output) and untrained (tool output, template) spans,
so the check can re-tokenize it with any model's tokenizer. Rollout log probs and other per-token data are dropped.
The file records each batch's SHA256.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import pickle
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


class _Stub:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        pass


class _Unpickler(pickle.Unpickler):
    # Miles / numpy classes are not needed for the fields read here
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (_Stub,), {})

    def persistent_load(self, pid):
        return None


def _runs(mask: list[int]) -> list[tuple[int, int, int]]:
    runs, start = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            runs.append((mask[start], start, i))
            start = i
    return runs


def main() -> None:
    from tokenizers import Tokenizer

    out, tokenizer, *specs = sys.argv[1:]
    tok = Tokenizer.from_file(tokenizer)

    def decode(ids):
        return tok.decode(list(ids), skip_special_tokens=False)

    sources, trials = [], []
    for spec in specs:
        path, _, names = spec.partition(":")
        raw = Path(path).read_bytes()
        archive = zipfile.ZipFile(io.BytesIO(raw))
        batch = _Unpickler(io.BytesIO(archive.read(next(n for n in archive.namelist() if n.endswith("data.pkl"))))).load()
        wanted = names.split(",")
        sources.append({"file": Path(path).name, "sha256": hashlib.sha256(raw).hexdigest(), "trials": wanted})
        for name in wanted:
            leaves = [s for s in batch["samples"] if Path(s["metadata"]["trial_dir"]).name == name]
            assert leaves, f"{name} not in {path}"
            stored = []
            for s in sorted(leaves, key=lambda s: s["metadata"]["leaf"]["node_id"]):
                prompt_len = len(s["tokens"]) - s["response_length"]
                response = s["tokens"][prompt_len:]
                stored.append({"node_id": s["metadata"]["leaf"]["node_id"], "prompt": decode(s["tokens"][:prompt_len]),
                               "response": [{"text": decode(response[a:b]), "trained": int(t)}
                                            for t, a, b in _runs(list(s["loss_mask"]))]})
            trials.append({"name": name, "task": leaves[0]["metadata"]["instance_id"],
                           "reward": float(leaves[0]["reward"]), "rollout_id": leaves[0]["rollout_id"],
                           "samples": stored})
    data = {"made": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tokenizer": Path(tokenizer).name,
            "sources": sources, "trials": trials}
    Path(out).write_bytes(gzip.compress(json.dumps(data, ensure_ascii=False).encode(), mtime=0))
    print(f"{out}: {len(trials)} trials, {Path(out).stat().st_size} bytes")


if __name__ == "__main__":
    main()
