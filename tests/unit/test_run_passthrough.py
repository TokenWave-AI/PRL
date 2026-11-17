"""scripts/run.sh forwards knobs into the Miles container, which runs with --cleanenv: a PRL_* variable that our
code reads but run.sh does not forward keeps its default silently (as PRL_LENGTH_LAMBDA once did)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_every_prl_variable_our_code_reads_reaches_the_container() -> None:
    run = (ROOT / "scripts" / "run.sh").read_text()
    [prefixes] = re.findall(r"grep -E '\^\(([^)]*)\)'", run)
    forwarded = tuple(prefixes.split("|"))
    read = set()
    for path in [*(ROOT / "prl").rglob("*.py"), *(ROOT / "transforms").rglob("*.py")]:
        read |= set(re.findall(r"""(?:getenv|environ\.get|environ\[)\(?\s*["'](PRL_\w+)""", path.read_text()))
    assert read, "no PRL_ variable found: the pattern is out of date"
    missing = sorted(v for v in read if not v.startswith(forwarded) and f"{v}=" not in run)
    assert not missing, f"read in the container but not forwarded by scripts/run.sh: {missing}"
