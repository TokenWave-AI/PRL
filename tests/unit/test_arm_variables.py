"""Every variable an arm sets is read somewhere: arms are plain bash, so a misspelled name (PRL_LENGTH_LAMBDA, see
scripts/run.sh) would otherwise be ignored without a word."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=", re.MULTILINE)
# Where the variables are read: our code, our Miles patches (PRL_ROLLOUT_ON_HEAD) and the Miles Harbor recipe
# (HARBOR_* / AGENT_*, through scripts/run.sh)
READERS = ("scripts", "prl", "transforms", "configs", "patches",
           "build/miles/examples/experimental/harbor", "build/miles/examples/swe-agent-harbor-docker")
# Read under a built name: prl/algo/llm_client.py reads f"{prefix}KEY_FILE"
BUILT = {"PRL_JUDGE_KEY_FILE"}
SUFFIXES = (".sh", ".py", ".env", ".inner", ".pbs", ".patch", "")


def test_every_variable_an_arm_sets_is_read():
    texts = []
    for top in READERS:
        for path in (ROOT / top).rglob("*"):
            if path.is_file() and path.suffix in SUFFIXES and "__pycache__" not in path.parts:
                text = path.read_text(errors="replace")
                if path.suffix == ".env":  # an assignment is not a read
                    text = ASSIGN.sub("", text)
                texts.append(text)
    corpus = "\n".join(texts)
    assigned = {name for arm in (ROOT / "configs/arms").glob("*.env") for name in ASSIGN.findall(arm.read_text())}
    unread = sorted(name for name in assigned - BUILT if not re.search(rf"(?<![A-Z0-9_]){name}(?![A-Z0-9_])", corpus))
    assert not unread, f"set by an arm, read nowhere (misspelled?): {unread}"
