"""Hint reroll: TRANSFORM=hint_reroll,
with the reroll arm (configs/arms/ota_train_transform_reroll.env: the group-reward post-process). GRPO.

For a failed trial, the run's model (the judge profile, PRL_JUDGE_*) names the first step where the agent went
wrong and writes one short hint for it: what to check again, never the command, a value or the answer (it gives
away as little as possible). The policy then plays the trial again from that step ``n`` times (``step.reroll``): the
steps before it replayed in a fresh sandbox, the hint at the end of the message the policy gets before the step (part of
its prompt, never trained; prl/rollout/reroll_agent.py), the task's tests scoring each branch. Credit ``verified``
(prl/algo/added_rewards.py): when a branch scores more than the trial, the trial's steps before the fork are not
blamed (advantage 0), and from the fork on the trial and its branches are compared with each other; when none does, the
trial trains as in GRPO.

The judge sees the task, the trajectory (clipped as for ATLAS, prl/algo/labeler.py ``render``) and the end of the
trial's test output (``verifier/test-stdout.txt``; ``test_chars``, 0: none), not the task's tests or solution. An
answer is refused, and the judge asked again with the reason, when its step is out of range (step 1 would be a new
trial) or its hint is longer than ``max_hint_words`` or reads like a command (backquotes, ``$``, ``|``, ``;``, ``&&``,
redirections) or spans several lines.

Params: ``n`` (branches, default 2), ``prompt`` (transforms/rewrite_prompts/<prompt>.md, default hint_v2: where to look
again, not what is wrong; hint_v3 stricter: only the part to look at, no detail inside it; hint_v1 also names the
mistake), ``max_hint_words`` (default 15; with hint_v1 pass 25, the length its prompt was written for),
``test_chars`` (default 1500), ``credit`` (default verified; subgroup or member to compare). On 16 failed trials of one of
our runs both prompts picked the same fork steps. Passed trials, trials split by a summary and trials in which the judge
finds no step to do better are left as they are. The decision is in ``traj.record["hint_reroll"]``, the judge's calls in ``<trial_dir>/transform_calls.json``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from prl.algo import labeler
from prl.transforms import Trajectory, ask
from prl.transforms.api import CREDITS

PROMPTS = Path(__file__).resolve().parent / "rewrite_prompts"
# A hint names what to check; these mark a command or code instead
COMMAND_MARKS = ("`", "$", "|", ";", "&&", ">", "<")


async def hint_reroll(traj: Trajectory, n: int = 2, prompt: str = "hint_v2", max_hint_words: int = 15,
                      test_chars: int = 1500, credit: str = "verified") -> Trajectory:
    if credit not in CREDITS:
        raise ValueError(f"credit must be one of {CREDITS}, got {credit!r}")
    if isinstance(max_hint_words, bool) or not isinstance(max_hint_words, int) or max_hint_words < 1:
        raise ValueError(f"max_hint_words must be a positive integer, got {max_hint_words!r}")
    record = traj.record["hint_reroll"] = {"prompt": prompt}
    if traj.reward is None or traj.reward > 0:
        record["why"] = f"reward {traj.reward}: only failed trials branch"
        return traj
    if traj.split:
        record["why"] = "split by a summary"
        return traj
    if len(traj.steps) < 2:
        record["why"] = "fewer than 2 steps: nothing to replay before a fork"
        return traj
    request = build_prompt(traj, instructions(prompt, max_hint_words), _test_output(traj, test_chars))
    answer = await ask(request, check=lambda content: parse_answer(content, len(traj.steps), max_hint_words))
    record.update(step=answer["step"], hint=answer["hint"], prompt_chars=len(request))
    if answer["step"] is None:
        record["why"] = "the judge found no step to do better"
        return traj
    traj.steps[answer["step"] - 1].reroll(n, credit, hint=f"Hint: {answer['hint']}")
    return traj


def instructions(version: str, max_words: int) -> str:
    """``rewrite_prompts/<version>.md`` with {max_words} filled in."""
    file = PROMPTS / f"{version}.md"
    if not file.is_file():
        raise ValueError(f"no prompt {version!r}; available: {sorted(f.stem for f in PROMPTS.glob('hint*.md'))}")
    return file.read_text().replace("{max_words}", str(max_words))


def _test_output(traj: Trajectory, chars: int) -> str | None:
    """The end of the trial's test output, or None (none asked, no trial directory, an empty or missing file)."""
    if chars <= 0 or not traj.trial_dir:
        return None
    try:
        text = (Path(traj.trial_dir) / "verifier" / "test-stdout.txt").read_text(errors="replace").strip()
    except OSError:
        return None
    return text[-chars:] if text else None


def build_prompt(traj: Trajectory, rules: str, tests: str | None) -> str:
    """The trajectory as the ATLAS judge sees it (shrunk to fit), the end of the test output, then the instructions."""
    scale = 1.0
    while True:
        task, steps = labeler.render(traj, scale)
        parts = ["## Task given to the agent (its full first prompt)", task, "", "## Trajectory",
                 ("Each step is one model turn: its output (reasoning, tool calls), then what the agent received next "
                  "(terminal output). Long texts are clipped."), ""]
        for k, step in enumerate(steps, 1):
            parts += [f"### Step {k}", "[model]", step.action, "[received]", step.observation or "(nothing)", ""]
        parts += ["## Outcome", "The task's tests failed after the last step.", ""]
        if tests:
            parts += ["## End of the test output (for you only)", tests, ""]
        text = "\n".join([*parts, rules])
        if len(text) <= labeler.TRAJECTORY_CHARS or scale < 0.05:
            return text
        scale *= 0.7


def parse_answer(content: str, n_steps: int, max_words: int) -> dict[str, Any]:
    """{"step": 2..n_steps, "hint": one short line} or {"step": None, "hint": None}, from the answer's last JSON object;
    raises ValueError (the judge is asked again with the reason) for anything else."""
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)
    for answer in reversed(labeler.json_objects(content)):
        if "step" not in answer:
            continue
        step, hint = answer["step"], answer.get("hint")
        if step is None:
            return {"step": None, "hint": None}
        if isinstance(step, str) and step.strip().isdigit():
            step = int(step)
        if isinstance(step, bool) or not isinstance(step, int) or not 2 <= step <= n_steps:
            raise ValueError(f"step must be a step number from 2 to {n_steps} (from step 1 the agent would just start "
                             f"over), or null; got {step!r}")
        if not isinstance(hint, str) or not hint.strip():
            raise ValueError("hint must be one short sentence")
        if "\n" in hint.strip():
            raise ValueError("hint must be a single sentence on one line")
        hint = " ".join(hint.split())
        if len(hint.split()) > max_words:
            raise ValueError(f"the hint has {len(hint.split())} words, at most {max_words}: say less")
        marks = [m for m in COMMAND_MARKS if m in hint]
        if marks:
            raise ValueError(f"the hint contains {' '.join(marks)}: name what to check, do not write a command or code")
        return {"step": step, "hint": hint}
    raise ValueError('no JSON object with "step" and "hint" in the answer')
