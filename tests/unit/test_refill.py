"""prl.rollout.refill: after the first draw of an update, a refill takes one prompt group, not a whole new batch."""

from __future__ import annotations

import asyncio
import importlib
import sys
import types


class _Source:
    def __init__(self) -> None:
        self.asked: list[int] = []
        self.added: list = []

    def get_samples(self, n: int) -> list:
        self.asked.append(n)
        return [[object()] for _ in range(n)]

    def add_samples(self, samples: list) -> None:
        self.added.extend(samples)


class _HarborRolloutFn:
    """Stands in for generate.RolloutFn: draws like Miles' driver when 2 groups of a batch of 4 are dropped."""

    def __init__(self, source: _Source) -> None:
        self.data_source = source

    async def _call_train(self, input: object) -> list:
        groups = self.data_source.get_samples(4)
        for _ in range(2):
            groups += self.data_source.get_samples(4)
        self.data_source.add_samples([])
        return groups


def _refill(monkeypatch):
    monkeypatch.setitem(sys.modules, "generate", types.SimpleNamespace(RolloutFn=_HarborRolloutFn))
    monkeypatch.delitem(sys.modules, "prl.rollout.refill", raising=False)
    return importlib.import_module("prl.rollout.refill")


def test_a_refill_draws_one_group_per_dropped_group(monkeypatch):
    source = _Source()
    fn = _refill(monkeypatch).RolloutFn(source)
    groups = asyncio.run(fn._call_train(None))
    assert source.asked == [4, 1, 1]
    assert len(groups) == 6
    assert fn.data_source is source


def test_each_update_starts_with_a_full_draw(monkeypatch):
    source = _Source()
    fn = _refill(monkeypatch).RolloutFn(source)
    asyncio.run(fn._call_train(None))
    asyncio.run(fn._call_train(None))
    assert source.asked == [4, 1, 1, 4, 1, 1]
