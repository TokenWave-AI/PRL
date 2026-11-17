"""Signed ATLAS: PRL_TRANSFORM=atlas_signed. A judge model grades the policy's own steps
good, bad or neutral (prl/algo/labeler.py ``grade``, configured by PRL_JUDGE_*), and the steps get positive
or negative weights on the trial's advantage. No text is changed, so every trained token is the policy's own.

The weights keep the sign rules of the outcome reward: in GRPO a passed trial's advantage is >= 0 and a failed one's
<= 0, so a bad step is always pushed down and a good step is never pushed down:

    mode      passed trial (A >= 0)                     failed trial (A <= 0)
    mild      bad -1, every other step 1               every step 1 (plain GRPO; not graded)
    strict    good 1, bad -1, neutral left out         bad 1, good and neutral left out

Parameters (the defaults give the table above):

    bad_weight=b         a passed trial's bad step gets -b instead of -1 (both modes)
    failed_good_weight=g mild also grades failed trials: their good steps get g (0: a correct step of a failed trial
                         is not pushed down, as in strict), bad and neutral steps 1 (pushed down as in GRPO);
                         None (default): failed trials are not graded
    quote_bad=True       token level (mild): the judge also quotes each bad step's mistake verbatim
                         (labeler.grade_quoted), and in a passed trial only the quoted tokens get -b; the rest of the
                         step keeps 1 (step.weight_text, every occurrence: a wrong command often also stands in the
                         reasoning and the plan). A quote that does not occur in the step (also after JSON-escaping
                         it: the judge may quote a command's "\n" as a newline) falls back to -b on the whole step;
                         record "quoted" / "quote_fallback" count the two (offline: 28 of 30 bad steps found, the
                         quote about 4% of the step's tokens)

``mild`` is plain GRPO plus a penalty on the mistakes of passed trials; ``strict`` trains only the graded steps (neutral
steps leave the loss, as in ATLAS). A strict trial with no step left to train gets weight 0 on every step (no push,
its tokens stay in the per-token normalization) instead of being left out as a whole. A passed trial in a group that
passed every time has advantage 0, so its bad steps are not pushed either. For GRPO and REINFORCE++ (one advantage
sign per trial), not PPO (the critic sets each token's sign). A trial the judge could not grade is skipped: it trains
as plain GRPO. The judge's request and answer are in ``<trial_dir>/atlas_labels.json``; the grades also in
``traj.record["atlas_signed"]`` (rollout/atlas_signed/* metrics, prl/rollout/rollout_metrics.py).
"""

from __future__ import annotations

import json

from prl.algo import labeler
from prl.transforms import Trajectory

MODES = ("mild", "strict")


async def atlas_signed(traj: Trajectory, mode: str = "mild", bad_weight: float = 1.0,
                       failed_good_weight: float | None = None, quote_bad: bool = False) -> Trajectory:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not bad_weight > 0:
        raise ValueError(f"bad_weight must be > 0 (a bad step of a passed trial gets -bad_weight), got {bad_weight}")
    if failed_good_weight is not None and (mode != "mild" or not 0 <= failed_good_weight <= 1):
        raise ValueError(f"failed_good_weight is for mode mild, in [0, 1] (0: not pushed down), got {failed_good_weight}")
    if quote_bad and mode != "mild":
        raise ValueError("quote_bad is for mode mild")
    passed = (traj.reward or 0) > 0
    record = traj.record["atlas_signed"] = {"mode": mode, "steps": len(traj.steps), "passed": passed}
    if mode == "mild" and not passed and failed_good_weight is None:
        record["graded"] = False  # a failed trial trains as in GRPO: no judge call
        return traj
    grades = await (labeler.grade_quoted(traj) if quote_bad else labeler.grade(traj))
    record["graded"] = grades is not None
    if grades is None:
        traj.skip("the judge gave no usable answer (atlas_labels.json)")
        return traj
    good, bad, quotes = grades if quote_bad else (*grades, {})
    record.update(good=sorted(n + 1 for n in good), bad=sorted(n + 1 for n in bad))
    if mode == "mild":
        if quote_bad:
            record.update(quoted=0, quote_fallback=0)
        for step in traj.steps:
            if passed and step.index in bad and quote_bad:
                text = _locate(step.action, quotes.get(step.index))
                if text is None:
                    step.weight = -bad_weight
                    record["quote_fallback"] += 1
                else:
                    step.weight_text(text, -bad_weight, every=True)
                    record["quoted"] += 1
            elif passed and step.index in bad:
                step.weight = -bad_weight
            elif not passed and step.index in good and failed_good_weight != 1:
                step.weight = failed_good_weight  # A <= 0: weight 0 keeps a correct step from being pushed down
        return traj
    weights = {n: (1.0 if n in good else -bad_weight if n in bad else 0.0) if passed else (1.0 if n in bad else 0.0)
               for n in range(len(traj.steps))}
    if not any(weights.values()):
        for step in traj.steps:
            step.weight = 0.0
        return traj
    for step in traj.steps:
        if weights[step.index]:
            step.weight = weights[step.index]
        else:
            step.mask()
    return traj


def _locate(action: str, quote: str | None) -> str | None:
    """The quote as it occurs in the action: verbatim, stripped, or JSON-escaped (commands sit in the action JSON, where a
    newline is written \\n); None if none of them does."""
    if not quote:
        return None
    def escape(t: str) -> str:
        return json.dumps(t, ensure_ascii=False)[1:-1]
    # the quote as given first: stripping first would cut a command's trailing newline off before it is escaped
    for text in (quote, escape(quote), quote.strip(), escape(quote.strip())):
        if text and text in action:
            return text
    return None
