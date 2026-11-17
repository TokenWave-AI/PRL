"""Re-encoding a rewritten turn (prl/transforms/api.py Trajectory rebuilds the samples with these): only the
changed text is tokenized again, the policy's own tokens around it are kept, and per-call metadata (weight versions)
moves with the tokens. Also the chat-template markers a turn is parsed by.
"""

from __future__ import annotations

import copy
import dataclasses
import difflib
import os
from typing import Any

# The model turn's and the environment message's structure (Qwen chat template + Terminus-2), as decoded with special
# tokens: "<thinking></think>\n\n{json}<|im_end|>" and "\n<|im_start|>user\n<message><|im_end|>\n<|im_start|>assistant\n<think>\n"
THINK_END = "</think>"


USER_START = "<|im_start|>user\n"


TURN_END = "<|im_end|>"


def _splice(tokenizer: Any, ids: list[int], old: str, new: str) -> tuple[list[int], list[int]]:
    """Token ids for ``new`` that keep the policy's own tokens (``ids``, the text ``old``) for the text the two share
    at both ends, and for each new token the index of the policy token it is (-1: encoded anew). Re-encoding a whole
    turn would change its tokenization in ~3.5% of turns (the policy sometimes samples a non-canonical one; grpo32
    updates 0-19), i.e. touch text nobody edited. A turn is spliced on each side of its "</think>" (one special
    token): a reasoning edit and a new action keep the policy's tokens between them; between two edits on the same
    side, text is encoded anew."""
    if old == new:
        return list(ids), list(range(len(ids)))
    marks = [p for p, t in enumerate(ids) if tokenizer.decode([t]) == THINK_END]
    if len(marks) == 1 and old.count(THINK_END) == 1 and new.count(THINK_END) == 1:
        p, cut, at = marks[0] + 1, old.index(THINK_END) + len(THINK_END), new.index(THINK_END) + len(THINK_END)
        if tokenizer.decode(ids[:p]) == old[:cut]:
            first, o1 = _splice_part(tokenizer, ids[:p], old[:cut], new[:at])
            second, o2 = _splice_part(tokenizer, ids[p:], old[cut:], new[at:])
            if tokenizer.decode(first + second) == new:
                return first + second, o1 + [o if o < 0 else o + p for o in o2]
    return _splice_part(tokenizer, ids, old, new)


SPLICE_MIN_MATCH = 12  # characters: a shorter run the edit shares with the old text is a chance match inside new text


def _token_spans(tokenizer: Any, ids: list[int], text: str) -> list[tuple[int, int, int, int]] | None:
    """(first token, end token, start char, end char) of each token in ``text``, a run of tokens that only decodes
    together (the bytes of one character) as one; None unless the tokens decode to ``text`` piece by piece."""
    units, pos, k = [], 0, 0
    while k < len(ids):
        j = k + 1
        piece = tokenizer.decode(ids[k:j])
        while "�" in piece and j < len(ids) and j - k < 8:
            j += 1
            piece = tokenizer.decode(ids[k:j])
        if not text.startswith(piece, pos):
            return None
        units.append((k, j, pos, pos + len(piece)))
        pos, k = pos + len(piece), j
    return units if pos == len(text) else None


def _splice_diff(tokenizer: Any, ids: list[int], old: str, new: str) -> tuple[list[int], list[int]] | None:
    """The policy's own tokens for every stretch of ``old`` that ``new`` keeps (at least SPLICE_MIN_MATCH characters),
    new text encoded anew (origin -1) in between; None when that does not decode to ``new`` (the caller falls back).
    Only what the transform wrote, and a policy token that straddles an edit, count as new (with rewrite+mask, one
    splice from the first edit to the last left ~10-15% of the policy's own reasoning out of the loss)."""
    units = _token_spans(tokenizer, ids, old)
    if units is None:
        return None
    # a shared head or tail is anchored, however short; a run in between must be long enough not to be chance
    blocks = [b for b in difflib.SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks()
              if b.size and (b.size >= SPLICE_MIN_MATCH or (b.a == b.b == 0)
                             or (b.a + b.size == len(old) and b.b + b.size == len(new)))]
    kept, bi = [], 0  # (start in new, end in new, first token, end token)
    for k, j, s, e in units:
        while bi < len(blocks) and blocks[bi].a + blocks[bi].size <= s:
            bi += 1
        if bi < len(blocks) and blocks[bi].a <= s and e <= blocks[bi].a + blocks[bi].size:
            kept.append((blocks[bi].b + s - blocks[bi].a, blocks[bi].b + e - blocks[bi].a, k, j))
    out, origin, cur = [], [], 0
    for ns, ne, k, j in kept:
        if ns > cur:
            middle = _encode(tokenizer, new[cur:ns])
            out, origin = out + middle, origin + [-1] * len(middle)
        out, origin, cur = out + list(ids[k:j]), origin + list(range(k, j)), ne
    if cur < len(new):
        middle = _encode(tokenizer, new[cur:])
        out, origin = out + middle, origin + [-1] * len(middle)
    return (out, origin) if tokenizer.decode(out) == new else None


def _splice_part(tokenizer: Any, ids: list[int], old: str, new: str) -> tuple[list[int], list[int]]:
    if old == new:
        return list(ids), list(range(len(ids)))
    if (spliced := _splice_diff(tokenizer, ids, old, new)) is not None:
        return spliced
    pre = len(os.path.commonprefix([old, new]))
    suf = len(os.path.commonprefix([old[pre:][::-1], new[pre:][::-1]]))

    def last(ok: Any, lo: int, hi: int) -> int:  # the largest p in [lo, hi] with ok(p), given ok(lo)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if ok(mid) else (lo, mid - 1)
        return lo

    def head_fits(p: int) -> bool:
        text = tokenizer.decode(ids[:p])
        return len(text) <= pre and old.startswith(text)

    def tail_fits(n: int) -> bool:  # the last n tokens
        text = tokenizer.decode(ids[len(ids) - n:]) if n else ""
        return len(text) <= suf and old.endswith(text)

    p = last(head_fits, 0, len(ids))
    t = last(tail_fits, 0, len(ids) - p)
    head, tail = tokenizer.decode(ids[:p]), tokenizer.decode(ids[len(ids) - t:]) if t else ""
    middle = _encode(tokenizer, new[len(head):len(new) - len(tail)])
    out = ids[:p] + middle + ids[len(ids) - t:]
    if tokenizer.decode(out) != new:  # a seam inside a multi-byte character: encode it whole
        out = _encode(tokenizer, new)
        return out, [-1] * len(out)
    return out, list(range(p)) + [-1] * len(middle) + list(range(len(ids) - t, len(ids)))


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Token ids of text that contains the chat template's special tokens (they map to their ids, as when decoded)."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:  # a tokenizer without that keyword (the pre-run check's wrapper, tests)
        ids = tokenizer.encode(text)
    return list(getattr(ids, "ids", ids))


def _field(span: Any, key: str) -> Any:
    return span[key] if isinstance(span, dict) else getattr(span, key)


def _moved(call: Any, start: int, end: int, shift: int, reencoded: bool) -> Any:
    """A turn's weight-version record (Miles WeightVersionsPerCall, or its dict form) at the turn's new position: one
    span over re-encoded text (its first version), else every span shifted."""
    call = copy.deepcopy(call)
    spans = call.spans if hasattr(call, "spans") else call
    if not spans:
        return call
    if reencoded:
        first = spans[0]
        version = _field(first, "version")
        span = ({"version": version, "abs_start": start, "abs_end": end} if isinstance(first, dict)
                else type(first)(version=version, abs_start=start, abs_end=end))
        if hasattr(call, "spans"):
            call.spans = [span]
        else:
            call[:] = [span]
        return call
    if hasattr(call, "spans"):  # Miles' WeightVersionSpan is a frozen dataclass: new spans, not assignments
        call.spans = [dataclasses.replace(s, abs_start=s.abs_start + shift, abs_end=s.abs_end + shift) for s in spans]
        return call
    for span in spans:
        span["abs_start"] += shift
        span["abs_end"] += shift
    return call


def _runs(mask: list[int]) -> list[tuple[int, int, int]]:
    runs, start = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            runs.append((mask[start], start, i))
            start = i
    return runs
