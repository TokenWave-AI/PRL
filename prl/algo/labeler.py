"""ATLAS critical-step labels for one training trial, from a judge model behind an OpenAI-compatible API.

A step is one agent turn = one trained run of ``loss_mask`` (after Miles' ownership masks and the summary masking,
training_samples.py): ``prl.transforms.core.Trajectory`` builds them, so a label maps onto the tokens exactly.
``render`` clips each step for the judge: the model's output (thinking clipped) and the context it received next
(tool output, clipped); ``label`` asks the judge which steps are critical, ``grade`` which are good and which are
bad (GRADE_INSTRUCTIONS). Used by the ``atlas`` and ``atlas_signed`` transforms (transforms/atlas.py,
transforms/atlas_signed.py), which run in the async generate adapter (harbor_training.py): the session postprocessor
runs on the session server's event loop and must not wait on the network.

Configuration: the judge profile of prl/algo/llm_client.py (PRL_JUDGE_API_BASE, _KEY_FILE, _MODEL,
_EXTRA_BODY, _CONCURRENCY, _TIMEOUT, _ATTEMPTS, _MAX_TOKENS), and
    PRL_JUDGE_PROMPT     ``paper`` (default): ATLAS' selection prompt (ATLaS, Chen et al., ACL 2025, arXiv
                               2503.02197, appendix D.1) with the RL changes marked in PAPER_INSTRUCTIONS;
                               ``short``: an earlier, shorter prompt of our own (not the paper's)
A trial whose labeling fails (HTTP error, unparsable answer, no step marked) is skipped by the atlas transform; the
label file says why. Each trial's request, answer and labels go to ``<trial_dir>/atlas_labels.json``, with the model
name the API answered as and why the answer ended.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prl.algo import llm_client

if TYPE_CHECKING:
    from prl.transforms.core import Trajectory

logger = logging.getLogger(__name__)

TASK_CHARS = 12000
THINK_CHARS = 2000
ACTION_CHARS = 3000
OBS_CHARS = 2000
# Whole trajectory text; per-step budgets shrink to fit (the judge's context, and its latency)
TRAJECTORY_CHARS = 240000

# short: an earlier, shorter prompt of our own
SYSTEM = "You review trajectories of an AI agent that solves tasks in a Linux terminal and identify its critical steps."

INSTRUCTIONS = """## Instructions
Identify the critical steps of this trajectory: the steps that most determine its outcome and are most worth learning
from (in a failed run: the steps most responsible for the failure). Typical critical steps:
- forming or revising the plan from what has been observed;
- a key decision, such as the command or approach that solves (or breaks) the task;
- interpreting an important observation or error and recovering from it;
- verifying the result and deciding to finish.
Routine steps are not critical: exploration that turned up nothing decisive, repeated or redundant commands, waiting,
cosmetic changes. Select the smallest set that captures the decisive actions: usually a third of the steps or fewer,
and at least one.

Answer with only a JSON object:
{"critical_steps": [<step numbers>], "reasons": {"<step number>": "<a few words>"}}"""

# The paper's text, verbatim but for: "conversation" for its "expert conversation" (the policy's own attempt), the
# sentence on failed attempts (RL trajectories fail too; the outcome is shown above), the step cap filled in as
# max(1, floor(30% of the steps)), and a JSON answer with the paper's three parts (to parse it).
PAPER_RATIO = 0.3
PAPER_INSTRUCTIONS = """A critical step is defined as a key action or decision that, if performed correctly, significantly increases the
likelihood of successfully completing the task. It represents a turning point in the process that influences the
outcome of subsequent actions. More specifically, critical steps include:
- Plan Creation: Steps where the LLM agent formulates sub-goals by analyzing previous observations and considering the
final objective, breaking down the larger goal into manageable tasks that guide the agent's actions towards the overall
outcome.
- Critical Observation: Steps where the LLM agent identifies and analyzes key information from the environment, which
help agent understand the objective or state and refine its strategy and decision-making towards more effective
outcomes.
- Critical Action: Steps where the LLM agent takes decisive and impactful actions based on prior observations,
significantly advancing the process toward the final objectives. These actions are crucial in shaping the direction of
the agent's strategy and are often pivotal moments that determine progress or failure, ensuring that the agent remains
on track to achieve the desired outcome.
- Self Correction: Steps where the LLM agent carefully recalls and assesses its previous actions or decisions,
especially after encountering failure or suboptimal outcomes. During this process, the agent reflects on what went
wrong, identifies areas for improvement, and adjusts its approach to enhance future performance, which helps the agent
refine its decision-making and better align with the overall objective.

Task Description: Your task is:
1. Induce a high-level plan or strategy based on the conversation, summarizing the key steps needed to successfully
complete the task.
2. Based on this high-level plan, identify the most critical action steps in the conversation. A maximum of {cap} steps
may be chosen from the conversation.
3. Provide a detailed explanation for choosing these critical steps, specifying which category (e.g., key observation,
planning, recall, pivotal action) they belong to and why mastering these steps ensures the success of the task.
If the conversation failed (see its outcome), the critical steps are the ones that most determined the failure.

Answer Format (only a JSON object):
{{"plan": "<the high-level plan: the strategy and key steps for task completion>", "critical_steps": [<step numbers>],
"reasons": {{"<step number>": "<category, and why the step is critical>"}}}}"""


# Signed ATLAS: the judge marks good and bad steps, and the transform turns them into positive and
# negative step weights. Both are capped as the paper caps critical ones: uncapped, gpt-6.1-sol (low effort) marked
# 62% of the steps of passed fixture trials bad (72% of all), which would push a passed trial down on balance.
# A bad step needs a nameable mistake. Short reasons: they are only for debugging.
GRADE_SYSTEM = "You review trajectories of an AI agent that solves tasks in a Linux terminal and grade its steps."
GRADE_INSTRUCTIONS = """## Instructions
Grade the steps of this trajectory. A step is one model turn: its reasoning and its commands.
- good: a step that clearly moves the task toward success: a correct key decision, command, reading of an output, or
  check of the result.
- bad: a step with a mistake you can name: a claim the terminal output contradicts, a wrong or failing command, a
  needless repeat of an earlier command, a wrong decision about what to do next, declaring the task done when it is
  not, or going on after it is done.
- every other step is neutral: leave it out.
Grade each step on its own, whatever the outcome: a passed run can contain bad steps and a failed run good ones. Mark
at most {cap} steps good and at most {cap} steps bad: the ones that mattered most. Keep each reason to a few words.

Answer with only a JSON object:
{{"good": [<step numbers>], "bad": [<step numbers>], "reasons": {{"<step number>": "<a few words>"}}}}"""

# Token-level signed ATLAS: the same grading, plus a verbatim quote of each bad step's mistake, so
# only those tokens are pushed down (transforms/atlas_signed.py quote_bad; core.Step.weight_text).
GRADE_QUOTE_INSTRUCTIONS = GRADE_INSTRUCTIONS.replace("""Answer with only a JSON object:
{{"good": [<step numbers>], "bad": [<step numbers>], "reasons": {{"<step number>": "<a few words>"}}}}""",
"""For each bad step, quote its mistake: copy, character for character, the shortest piece of that step's [model] text
that contains the mistake (a command, a wrong claim, or the sentence with the wrong decision), at most {quote_chars}
characters, from a part that is shown (not across an omission mark).

Answer with only a JSON object:
{{"good": [<step numbers>], "bad": [<step numbers>], "quotes": {{"<bad step number>": "<exact text>"}},
"reasons": {{"<step number>": "<a few words>"}}}}""")
QUOTE_CHARS = 300


def step_cap(n_steps: int) -> int:
    return max(1, int(n_steps * PAPER_RATIO))


@dataclass
class Rendered:
    """One step as the judge sees it (clipped)."""

    action: str
    observation: str


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n[... {len(text) - limit} characters omitted ...]\n{text[-tail:]}"


def _render_action(text: str, think_chars: int, action_chars: int) -> str:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not match:
        return clip(text.strip(), action_chars)
    rest = (text[: match.start()] + text[match.end():]).strip()
    return f"<think>\n{clip(match.group(1).strip(), think_chars)}\n</think>\n{clip(rest, action_chars)}"


def render(traj: Trajectory, scale: float = 1.0) -> tuple[str, list[Rendered]]:
    """The task prompt and the trial's steps, clipped (per-step budgets times ``scale``)."""
    steps = [Rendered(_render_action(s.action, int(THINK_CHARS * scale), int(ACTION_CHARS * scale)),
                      clip(s.observation.strip(), int(OBS_CHARS * scale))) for s in traj.steps]
    return clip(traj.task, TASK_CHARS), steps


def build_prompt(task: str, steps: list[Rendered], reward: float | None, style: str = "paper") -> str:
    parts = ["## Task given to the agent (its full first prompt)", task, "",
             "## Trajectory",
             ("Each step is one model turn: its output (reasoning, tool calls), then what the agent received next "
              "(terminal output). Long texts are clipped."), ""]
    for n, step in enumerate(steps, 1):
        parts += [f"### Step {n}", "[model]", step.action, "[received]", step.observation or "(nothing)", ""]
    if reward is not None:
        parts += ["## Outcome", f"The task's tests {'passed' if reward > 0 else 'failed'} after the last step.", ""]
    capped = {"paper": PAPER_INSTRUCTIONS, "grade": GRADE_INSTRUCTIONS, "grade_quote": GRADE_QUOTE_INSTRUCTIONS}
    parts.append(capped[style].format(cap=step_cap(len(steps)), quote_chars=QUOTE_CHARS) if style in capped
                 else INSTRUCTIONS)
    return "\n".join(parts)



def json_objects(content: str) -> list[dict[str, Any]]:
    """Every JSON object in a model's answer, in order (text around and between them, e.g. "step {3} looked off",
    is skipped): the parsers take the last one that fits."""
    decoder, found, i = json.JSONDecoder(), [], content.find("{")
    while i >= 0:
        try:
            value, end = decoder.raw_decode(content, i)
        except json.JSONDecodeError:
            i = content.find("{", i + 1)
            continue
        if isinstance(value, dict):
            found.append(value)
        i = content.find("{", end)
    return found

def parse_answer(content: str, n_steps: int, cap: int | None = None) -> tuple[list[int], dict[str, str]]:
    """Critical step numbers (1-based, in range, sorted) and reasons, from the judge's last JSON object; more than
    ``cap`` steps is an invalid answer (retried)."""
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)
    for answer in reversed(json_objects(content)):
        steps = sorted({int(x) for x in answer.get("critical_steps", []) if 1 <= int(x) <= n_steps})
        if not steps:
            raise ValueError("no critical step in range")
        if cap is not None and len(steps) > cap:
            raise ValueError(f"{len(steps)} critical steps, more than the cap of {cap}")
        return steps, {str(k): str(v) for k, v in (answer.get("reasons") or {}).items()}
    raise ValueError("no JSON object in the answer")


def parse_grades(content: str, n_steps: int, cap: int) -> tuple[set[int], set[int], dict[str, str]]:
    """Good and bad step numbers (1-based, in range) and reasons, from the judge's last JSON object; a step both good and
    bad, or more than ``cap`` good or bad steps, is an invalid answer (retried). No good or bad step is a valid answer."""
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)
    for answer in reversed(json_objects(content)):
        if not isinstance(answer, dict) or not {"good", "bad"} <= answer.keys():
            continue
        good, bad = ({int(x) for x in answer[key] if 1 <= int(x) <= n_steps} for key in ("good", "bad"))
        if good & bad:
            raise ValueError(f"steps {sorted(good & bad)} are both good and bad")
        for name, steps in (("good", good), ("bad", bad)):
            if len(steps) > cap:
                raise ValueError(f"{len(steps)} {name} steps, more than the cap of {cap}")
        return good, bad, {str(k): str(v) for k, v in (answer.get("reasons") or {}).items()}
    raise ValueError("no JSON object with good and bad in the answer")


def prompt_style() -> str:
    style = os.getenv("PRL_JUDGE_PROMPT", "paper")
    if style not in ("paper", "short"):
        raise ValueError(f"PRL_JUDGE_PROMPT must be paper or short, got {style!r}")
    return style


def probe() -> None:
    """Launcher check: the judge's prompt style, key file, base URL and model (llm_client.probe, one short request)."""
    prompt_style()
    llm_client.probe()


async def _judge(traj: Trajectory, style: str, parse: Callable[[str, int], Any],
                 summary: Callable[[Any], dict[str, Any]], feedback: bool = False) -> Any | None:
    """``parse(answer, n_steps)`` of the judge's answer on the trial (None: no usable answer); the request, the answer
    and ``summary(parsed)`` go to ``<trial_dir>/atlas_labels.json``. ``feedback``: a rejected answer goes back to the
    judge with the reason (llm_client.ask)."""
    cfg = llm_client.config()
    reward = traj.reward
    scale = 1.0
    while True:
        task, steps = render(traj, scale)
        prompt = build_prompt(task, steps, reward, style)
        if len(prompt) <= TRAJECTORY_CHARS or scale < 0.05:
            break
        scale *= 0.7
    record: dict[str, Any] = {"model": cfg["model"], "prompt_style": style, "extra_body": cfg["extra_body"],
                              "n_steps": len(steps), "reward": reward, "scale": scale, "prompt_chars": len(prompt),
                              "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()}
    answer = None
    if steps:
        grade_system = [{"role": "system", "content": GRADE_SYSTEM}]
        system = {"paper": [], "grade": grade_system, "grade_quote": grade_system}.get(  # the paper has none
            style, [{"role": "system", "content": SYSTEM}])
        answer = await llm_client.ask([*system, {"role": "user", "content": prompt}],
                                      lambda content: parse(content, len(steps)), record=record, feedback=feedback)
        if answer is not None:
            record.update(summary(answer))
    record["fallback"] = answer is None
    if traj.trial_dir:
        try:
            (Path(traj.trial_dir) / "atlas_labels.json").write_text(json.dumps({**record, "prompt": prompt}) + "\n")
        except OSError:
            logger.exception("[prl] atlas: could not write labels to %s", traj.trial_dir)
    return answer


async def label(traj: Trajectory) -> set[int] | None:
    """The 0-based indices of the trial's critical steps (None: labeling failed, keep all)."""
    style = prompt_style()
    answer = await _judge(
        traj, style, lambda content, n: parse_answer(content, n, step_cap(n) if style == "paper" else None),
        lambda a: {"critical_steps": a[0], "reasons": a[1]})
    return None if answer is None else {n - 1 for n in answer[0]}


async def grade(traj: Trajectory) -> tuple[set[int], set[int]] | None:
    """The 0-based indices of the trial's good and bad steps (None: grading failed)."""
    answer = await _judge(traj, "grade", lambda content, n: parse_grades(content, n, step_cap(n)),
                          lambda a: {"good_steps": sorted(a[0]), "bad_steps": sorted(a[1]), "reasons": a[2]},
                          feedback=True)  # over a cap, the judge is told so (asked again plainly, it repeats itself)
    return None if answer is None else ({n - 1 for n in answer[0]}, {n - 1 for n in answer[1]})


def parse_quotes(content: str) -> dict[int, str]:
    """The quotes of the judge's last JSON object with good and bad (1-based step number -> text); a missing or
    malformed quote is left out (the transform falls back to the whole step)."""
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)
    for answer in reversed(json_objects(content)):
        if isinstance(answer, dict) and {"good", "bad"} <= answer.keys():
            quotes = answer.get("quotes") if isinstance(answer.get("quotes"), dict) else {}
            out = {}
            for k, v in quotes.items():
                try:
                    out[int(k)] = v
                except (TypeError, ValueError):
                    continue
            return {k: v for k, v in out.items() if isinstance(v, str) and v.strip()}
    return {}


async def grade_quoted(traj: Trajectory) -> tuple[set[int], set[int], dict[int, str]] | None:
    """``grade`` with a verbatim quote of each bad step's mistake: (good, bad, {bad step: quote}), 0-based (None:
    grading failed). A quote is the judge's text as given; whether it occurs in the step is the caller's check."""
    answer = await _judge(traj, "grade_quote",
                          lambda content, n: (*parse_grades(content, n, step_cap(n)), parse_quotes(content)),
                          lambda a: {"good_steps": sorted(a[0]), "bad_steps": sorted(a[1]), "reasons": a[2],
                                     "quotes": {str(k): v for k, v in a[3].items()}},
                          feedback=True)
    if answer is None:
        return None
    good, bad, _, quotes = answer
    return {n - 1 for n in good}, {n - 1 for n in bad}, {n - 1: q for n, q in quotes.items() if n in bad}
