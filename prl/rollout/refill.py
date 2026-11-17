"""Training rollout function: a dropped prompt group is refilled with one prompt, not a whole new batch.

Where: ``--rollout-function-path prl.rollout.refill.RolloutFn`` (scripts/lib/container_main.sh), for synchronous
training; ``--fully-async`` brings its own rollout function. It is the Harbor recipe's ``generate.RolloutFn`` (agent
metrics; examples/swe-agent-harbor-docker/generate.py) with one change to how prompts are drawn. Eval is unchanged.

Why: Miles' training rollout (``inference_rollout_train.generate_rollout_async``) calls
``data_source(--over-sampling-batch-size)`` whenever fewer groups are in flight than the batch still needs. A group
dropped before the batch (an infrastructure-aborted trial: ``check_no_aborted``) therefore draws a whole new batch of
prompts (64 in ota_train_full), and once the batch is full the groups still running are aborted and their prompts are
not put back (that needs ``--partial-rollout``, which cannot resume a Harbor trial). One run drew 800 prompts in 14
updates and trained 448 of them. Here only the first draw of an update takes the over-sampling batch; each refill draws
one group. With ``--over-sampling-batch-size`` equal to the batch (our arms), the groups in flight plus the kept ones
then always equal the batch, so the only prompts an epoch skips are those of the dropped groups themselves.
"""

from __future__ import annotations

from typing import Any

from generate import RolloutFn as _HarborRolloutFn


class _OneGroupRefill:
    """The data source as one update sees it: the first draw as asked, every later one a single group."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self._drawn = False

    def get_samples(self, num_samples: int) -> list:
        n = 1 if self._drawn else num_samples
        self._drawn = True
        return self._source.get_samples(n)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)


class RolloutFn(_HarborRolloutFn):
    async def _call_train(self, input: Any) -> Any:
        source = self.data_source
        self.data_source = _OneGroupRefill(source)
        try:
            return await super()._call_train(input)
        finally:
            self.data_source = source
