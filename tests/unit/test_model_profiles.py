"""Model profiles (configs/models/*.env) against the vendored Miles tree. scripts/lib/container_main.sh turns a
profile into --hf-checkpoint, the Megatron model args, --ref-load, --tito-model and the SGLang parsers; Miles does
not derive the tool-call parser from the TITO tokenizer, so a mismatch would only show up as broken tool calls.
Bash + ast only (no torch, no model files)."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MILES = ROOT / "third_party" / "miles"
PROFILES = sorted((ROOT / "configs" / "models").glob("*.env"))
KEYS = ("MODEL_HF_REPO", "MODEL_ARGS_NAME", "MODEL_TORCH_DIST", "TITO_MODEL", "SGLANG_TOOL_CALL_PARSER",
        "SGLANG_REASONING_PARSER", "MODEL_MAX_CONTEXT", "MODEL_HF_DIR")


def load(profile: Path, hf_home: Path, **env: str) -> dict[str, str]:
    echo = "; ".join(f'echo "{k}=${{{k}}}"' for k in KEYS)
    out = subprocess.run(["bash", "-c", f'set -euo pipefail; set -a; source "{profile}"; set +a; {echo}'],
                         env={"PATH": "/usr/bin:/bin", "HF_HOME": str(hf_home), "PRL_EXT": "/ext", **env},
                         capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines())


def tito_parsers() -> dict[str, tuple[str | None, str | None]]:
    """--tito-model value -> (tool_call_parser, reasoning_parser) of the class Miles resolves it to."""
    src = (MILES / "miles/utils/chat_template_utils/tito_tokenizer.py").read_text()
    classes: dict[str, tuple[list[str], dict[str, object]]] = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef):
            attrs = {t.id: n.value.value for n in node.body if isinstance(n, ast.Assign)
                     and isinstance(n.value, ast.Constant) for t in n.targets if isinstance(t, ast.Name)}
            classes[node.name] = ([b.id for b in node.bases if isinstance(b, ast.Name)], attrs)

    def attr(cls: str, key: str):
        bases, attrs = classes[cls]
        if key in attrs:
            return attrs[key]
        return next((v for b in bases if b in classes and (v := attr(b, key)) is not None), None)

    members = classes["TITOTokenizerType"][1]
    resolved = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.match_case) and isinstance(node.pattern, ast.MatchValue):
            value = node.pattern.value
            ret = node.body[0]
            if (isinstance(value, ast.Attribute) and isinstance(ret, ast.Return) and isinstance(ret.value, ast.Name)
                    and value.attr in members and ret.value.id in classes):
                resolved[members[value.attr]] = (attr(ret.value.id, "tool_call_parser"),
                                                 attr(ret.value.id, "reasoning_parser"))
    return resolved


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.stem)
def test_profile_matches_miles(profile: Path, tmp_path: Path) -> None:
    p = load(profile, tmp_path)
    assert all(p[k] for k in KEYS if k != "MODEL_HF_DIR"), p
    assert (MILES / "scripts" / "models" / f"{p['MODEL_ARGS_NAME']}.py").is_file()
    assert p["MODEL_TORCH_DIST"].startswith("/ext/ckpt/miles/")
    assert int(p["MODEL_MAX_CONTEXT"]) >= 32768
    tool, reasoning = tito_parsers()[p["TITO_MODEL"]]
    assert p["SGLANG_TOOL_CALL_PARSER"] == tool
    assert p["SGLANG_REASONING_PARSER"] == (reasoning or p["SGLANG_REASONING_PARSER"])


def test_known_tito_pairs() -> None:
    # The two families in use; guards the AST walk above against silently resolving nothing.
    pairs = tito_parsers()
    assert pairs["qwen35"] == ("qwen3_coder", "qwen3") and pairs["qwen3"] == ("qwen25", "qwen3")


def test_default_profiles_for_model_size() -> None:
    # MODEL unset -> qwen3.5-$MODEL_SIZE (scripts/lib/env.sh, container_main.sh; default 9B)
    for size in ("9B",):
        assert (ROOT / "configs" / "models" / f"qwen3.5-{size}.env").is_file()


def test_hf_dir_resolution(tmp_path: Path) -> None:
    profile = ROOT / "configs" / "models" / "qwen3-8B.env"
    assert load(profile, tmp_path)["MODEL_HF_DIR"] == ""  # not downloaded yet: empty, not a set -e failure
    snap = tmp_path / "hub" / "models--Qwen--Qwen3-8B" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    assert load(profile, tmp_path)["MODEL_HF_DIR"] == str(snap)
    assert load(profile, tmp_path, MODEL_HF_DIR="/export")["MODEL_HF_DIR"] == "/export"
    q35 = ROOT / "configs" / "models" / "qwen3.5-9B.env"
    assert load(q35, tmp_path, QWEN35_9B="/q35")["MODEL_HF_DIR"] == "/q35"  # qsub -v QWEN35_9B=<export> still works
