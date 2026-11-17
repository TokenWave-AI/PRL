"""Rewrite successful trials into cleaner examples: PRL_TRANSFORM=rewrite.

A teacher model (the judge profile of prl/algo/llm_client.py: the model PRL_JUDGE_MODEL names)
reads a trial that passed its tests and proposes small edits that make it a more standard solution: delete turns that
repeat a check or explore without effect, fix wrong reasoning with minimal edits, replace a detour by the action that
was evident from what had been observed. Where an edit changes what the terminal would have shown, the teacher plays
the terminal and writes the new output. It also judges whether the edited trial would still pass; the reward stays 1,
and a plan it does not expect to pass is not used. Edits are small on purpose: the trial should stay the policy's own.

Failed trials, trials split by a summary, and trials without an editable turn pass unchanged (status ok). A plan is
applied to a copy first and must keep to the budget (at most ``max_step_share`` of the turns deleted or given a new
action; at most ``max_char_share`` of the edited trial's model text written by the teacher; with ``max_change_share``,
at most that share of the agent's text changed, written and removed alike, a deleted turn counting whole; with
``max_removed_share``, at most that share removed; with ``ops``, only those operations), leave the keystrokes and
terminal outputs of the last ``protect_last`` turns alone (the completion protocol; their reasoning and the analysis /
plan text of their action may change), give actions Harbor's Terminus-2 parser accepts, and change terminal outputs
only at or after its first edit of the model's text; otherwise the teacher is asked again with the reason (the
profile's attempts), then the trial is skipped. The criteria come from ``rewrite_prompts/<prompt>.md``; text before
its first "## " heading replaces the request's opening paragraph (v5: correct mistakes only). The framework
saves every request and answer (``<trial_dir>/transform_calls.json``); a summary of the edits goes to
``traj.record["rewrite"]``.

``mask_new=True`` (rewrite+mask): the teacher's text stays in the context but out of the loss
(``Trajectory.mask_new_tokens``), so the policy is trained only on its own words, in the context the teacher corrected.
The teacher may then write more; what the policy loses is the positive signal of its own removed text, which
``max_removed_share`` bounds (removed or replaced own text is what unbalances GRPO).

``successes="keep"`` with ``failures="fix"`` (PPO fix-only): successful trials train as the policy wrote them and
only failed ones are repaired, like rejection sampling that rescues its rejects with the smallest edits;
``fix_max_change_share`` bounds a repair as ``max_change_share`` bounds a cleanup (prompt ``fix_v2``).

``added="reroll"``: the teacher picks the turn from which the policy plays a failed trial again (``step.reroll()``,
prl/transforms/api.py), and may correct the turns before it with the smallest edits. The corrections reach the replay
only; the trial trains as sampled, and the reroll's own verifier reward makes it one more member of the group.
``reroll_prompt``: the criteria (``reroll_v1``, a default to start from: when to reroll is the user's choice);
``reroll_successes``: passing trials too. The answer is {"edits": [...], "reroll": <step> | null, "reason": ...}; the
edits are delete / edit_thinking / set_command on turns before the reroll step (the replay's terminal output is real),
within ``fix_max_change_share`` if set. ``reroll_n`` branches from that step (default 1), trained as ``reroll_credit``
says (``member``, default, or ``subgroup``; prl/transforms/api.py ``Step.reroll``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from prl.algo.labeler import clip
from prl.transforms import Trajectory, ask

TASK_CHARS = 12000
TESTS_CHARS = 12000  # per file of the task's tests/ (fix mode)
PROMPT_CHARS = 200000  # a longer trial is left unchanged (the teacher needs the reasoning verbatim)
MODEL_OPS = ("delete", "edit_thinking", "set_command")
TERMINAL_PREFIX = "New Terminal Output:\n\n"  # Terminus-2's message after a turn; its finishing messages differ
REROLL_OPS = MODEL_OPS  # before a reroll: the replay runs the commands, so no terminal output is the teacher's

PROMPTS = Path(__file__).with_name("rewrite_prompts")  # the rewrite criteria, one file per version
# The answer format check_plan expects; the same for every version of the criteria
ANSWER = """## Answer: only a JSON object
{"edits": [
  {"step": <n>, "op": "delete", "why": "..."},
  {"step": <n>, "op": "edit_thinking", "old": "<exact text from that step's reasoning, occurring once>", "new": "...", "why": "..."},
  {"step": <n>, "op": "set_command", "command": {...}, "why": "..."},
  {"step": <n>, "op": "set_terminal", "terminal": "<the full terminal output after step n>", "why": "..."}],
 "passes": true,
 "reason": "<why the edited trajectory still solves the task>"}"""
REROLL_ANSWER = """## Answer: only a JSON object
{"edits": [
  {"step": <n>, "op": "delete", "why": "..."},
  {"step": <n>, "op": "edit_thinking", "old": "<exact text from that step's reasoning, occurring once>", "new": "...", "why": "..."},
  {"step": <n>, "op": "set_command", "command": {...}, "why": "..."}],
 "reroll": <the step the agent writes again on its own, after every edited step; or null>,
 "reason": "<what went wrong, and why the agent can succeed from that step>"}"""


def criteria(version: str) -> str:
    """``rewrite_prompts/<version>.md``: what to fix and what to leave, with {budget}, {n}, {protected} and {change_share}
    filled in by build_prompt (v1: the prompt of the first pilot, verbatim; v2: a numbered checklist; v4: after the
    prompt loop; v5: correct mistakes only, after a run with 21% of the reasoning changed diverged)."""
    file = PROMPTS / f"{version}.md"
    if not file.is_file():
        raise ValueError(f"no rewrite prompt {version!r}; available: {sorted(f.stem for f in PROMPTS.glob('*.md'))}")
    return file.read_text()


def build_prompt(traj: Trajectory, budget: int, protect_last: int, obs_chars: int, version: str = "v2",
                 tests: str | None = None, change_share: float | None = None, removed_share: float | None = None,
                 answer: str = ANSWER) -> str:
    """The teacher's request. With ``tests`` (a failed trial to fix), the task's tests are shown and the goal is a
    trajectory that passes."""
    n = len(traj.steps)
    protected = ", ".join(str(i + 1) for i in range(max(0, n - protect_last), n))
    rules = criteria(version).replace("{budget}", str(budget)).replace("{n}", str(n)).replace("{protected}", protected)
    rules = rules.replace("{change_share}", f"{change_share:.0%}" if change_share is not None else "a small part")
    rules = rules.replace("{removed_share}", f"{removed_share:.0%}" if removed_share is not None else "a small part")
    own_intro, _, rest = rules.partition("\n## ")
    if tests is None and not rules.startswith("## ") and rest:  # the version's own opening paragraph
        intro, rules = [own_intro.strip(), ""], "## " + rest
    elif tests is None:
        intro = [("You review a trajectory of an AI agent that solved a task in a Linux terminal: it passed the task's "
                  "tests. Turn it into a cleaner example of how to solve the task, with as few changes as possible; it "
                  "will be used to train the agent. You also play the terminal: where your edits change what the "
                  "terminal would have shown, you write what it would show instead."), ""]
    else:
        intro = [("You review a trajectory of an AI agent that tried to solve a task in a Linux terminal and failed the "
                  "task's tests. Turn it into a trajectory that passes, changing as little as possible; it will be used "
                  "to train the agent. You also play the terminal: where your edits change what the terminal would have "
                  "shown, you write what it would show instead."), "",
                 "## The task's tests (the agent never saw them)", tests, ""]
    parts = [*intro, "## Task (the agent's first prompt)", clip(traj.task, TASK_CHARS), "", "## Trajectory",
             "Each step: the agent's reasoning, its action, and what the terminal showed next (long outputs clipped).",
             ""]
    for step in traj.steps:
        command = step.command
        parts += [f"### Step {step.index + 1}", "[reasoning]", step.thinking or "",
                  "[action]", json.dumps(command, indent=2, ensure_ascii=False) if command is not None else step.action,
                  "[terminal]", clip(step.terminal, obs_chars) if step.terminal is not None else "(end of trajectory)", ""]
    parts.append(rules + answer)
    return "\n".join(parts)


def _parser() -> Any:
    try:
        from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
    except ImportError:  # outside the Miles container without Harbor: the structural check below only
        return None
    return TerminusJSONPlainParser()


def apply_plan(traj: Trajectory, edits: list[dict[str, Any]]) -> None:
    for edit in edits:
        step = traj.steps[edit["step"] - 1]
        if edit["op"] == "delete":
            step.delete()
        elif edit["op"] == "edit_thinking":
            step.edit_thinking(edit["old"], edit["new"])
        elif edit["op"] == "set_command":
            step.set_command(edit["command"])
        else:
            step.set_terminal(edit["terminal"])


def check_plan(traj: Trajectory, answer: dict[str, Any], budget: int, protect_last: int, max_char_share: float,
               fix: bool = False, max_change_share: float | None = None, ops: tuple[str, ...] | None = None,
               max_removed_share: float | None = None, execute: bool = False) -> None:
    """Raise ValueError unless the plan keeps to the rules; it is applied to a copy of the trial's steps. ``fix`` (a
    failed trial): the finishing steps' terminal outputs may change too, their keystrokes still may not."""
    edits = answer.get("edits")
    if not isinstance(edits, list) or not isinstance(answer.get("passes"), bool):
        raise ValueError("the answer needs an 'edits' list and a boolean 'passes'")  # noqa: TRY004 -- ask again
    if edits and not answer["passes"] and not execute:  # execute: the verifier decides, not the teacher
        raise ValueError("edits on a trajectory the teacher does not expect to pass")
    n = len(traj.steps)
    for edit in edits:
        if not (isinstance(edit, dict) and edit.get("op") in (*MODEL_OPS, "set_terminal")
                and isinstance(edit.get("step"), int) and 1 <= edit["step"] <= n):
            raise ValueError(f"malformed edit {str(edit)[:200]}")
        if ops is not None and edit["op"] not in ops:
            raise ValueError(f"step {edit['step']}: {edit['op']} is not allowed here, only {', '.join(ops)}")
        if edit["step"] > n - protect_last and edit["op"] != "edit_thinking" and not (fix and edit["op"] == "set_terminal") \
                and not (edit["op"] == "set_command"
                         and _same_commands(traj.steps[edit["step"] - 1].command, edit.get("command"))):
            raise ValueError(f"step {edit['step']} finishes the task: only its reasoning, and the analysis and plan of "
                             "its action, may change")
    model_steps = {e["step"] for e in edits if e["op"] in MODEL_OPS}
    structural = {e["step"] for e in edits if e["op"] in ("delete", "set_command") and e["step"] <= n - protect_last}
    if len(structural) > budget:  # minimal reasoning fixes are bounded by max_char_share instead
        raise ValueError(f"{len(structural)} steps deleted or given a new action, more than {budget}")
    if any(e["op"] == "set_terminal" and (not model_steps or e["step"] < min(model_steps)) for e in edits):
        raise ValueError("a terminal output changed before any edit of the agent's turns")
    steps = copy.deepcopy(traj.steps)
    scratch = Trajectory(traj.task, traj.reward, steps, None, [], None)
    written = [0] * n  # per step: characters the teacher wrote into its action
    removed = [0] * n  # ...and of the agent's that it removed
    for edit in edits:
        before = steps[edit["step"] - 1].action
        apply_plan(scratch, [edit])  # wrong 'old' text, a step without JSON or terminal: ValueError
        written[edit["step"] - 1] += _written(before, steps[edit["step"] - 1].action)
        removed[edit["step"] - 1] += _written(steps[edit["step"] - 1].action, before)
    kept = [i for i in range(n) if not steps[i].deleted]
    total = sum(len(steps[i].action) for i in kept)
    if sum(written[i] for i in kept) > max_char_share * total:  # deleting is bounded by the step budget instead
        raise ValueError(f"{sum(written[i] for i in kept)} of {total} characters of the agent's text written by the "
                         f"teacher, more than {max_char_share:.0%}")
    if max_removed_share is not None:  # the agent's own text the plan takes away (rewrite+mask: its positive signal)
        gone = sum(len(traj.steps[i].action) if steps[i].deleted else removed[i] for i in range(n))
        original = sum(len(st.action) for st in traj.steps)
        if gone > max_removed_share * original:
            raise ValueError(f"{gone} of {original} characters of the agent's own text removed or replaced, more than "
                             f"{max_removed_share:.0%}: keep the agent's correct text, change only what is wrong")
    if max_change_share is not None:  # deleting counts too (a run diverged with 21% of the reasoning changed)
        changed, original = change_chars(traj, steps, written, removed)
        if changed > max_change_share * original:
            raise ValueError(f"{changed} of {original} characters of the agent's text changed (written, removed or in "
                             f"a deleted step), more than {max_change_share:.0%}: correct only what is wrong, with the "
                             "smallest edits")
    parser = _parser()
    for e in edits:
        if e["op"] != "set_command":
            continue
        command = e["command"]
        if not isinstance(command, dict) or not isinstance(command.get("commands"), list):
            raise ValueError(f"step {e['step']}: the action needs a 'commands' list")  # noqa: TRY004
        if command.get("task_complete") and e["step"] <= n - protect_last:
            raise ValueError(f"step {e['step']}: only the last steps complete the task")
        if parser is not None and (error := parser.parse_response(json.dumps(command)).error):
            raise ValueError(f"step {e['step']}: Terminus-2 cannot parse the action: {error}")


def change_chars(traj: Trajectory, steps: list[Any], written: list[int], removed: list[int]) -> tuple[int, int]:
    """(characters of the agent's text changed by a plan, characters of its text): written and removed per kept step,
    the whole action of a deleted step."""
    changed = sum(len(traj.steps[i].action) if steps[i].deleted else written[i] + removed[i] for i in range(len(steps)))
    return changed, sum(len(st.action) for st in traj.steps)


def _same_commands(old: dict[str, Any] | None, new: Any) -> bool:
    """A finishing step's new action keeps its keystrokes and completion flag (only analysis and plan text changed)."""
    return isinstance(old, dict) and isinstance(new, dict) and old.get("commands") == new.get("commands") \
        and bool(old.get("task_complete")) == bool(new.get("task_complete"))


def _written(old: str, new: str) -> int:
    """Characters of ``new`` between its common head and tail with ``old``: the text one edit wrote."""
    head = len(os.path.commonprefix([old, new]))
    tail = len(os.path.commonprefix([old[head:][::-1], new[head:][::-1]]))
    return len(new) - head - tail


async def _execute(twin: Trajectory, edits: list[dict[str, Any]], trial_dir: str | None) -> tuple[float | None, dict[str, Any]]:
    """Run the rewritten trial; its reward and the real terminal outputs from the first edit on. (None, why) if it could not."""
    from prl.rollout.replay import ReplayFailed, replay

    live = [s for s in twin.steps if not s.deleted and (s.command or {}).get("commands") is not None]
    try:
        reward, outputs = await replay(str(trial_dir), [s.command["commands"] for s in live])
    except ReplayFailed as exc:
        return None, {"failed": str(exc)}
    first = min(e["step"] for e in edits if e["op"] in MODEL_OPS) - 1
    replaced = 0
    for step, out in zip(live, outputs):  # after an agent timeout, the turns that ran
        terminal = step.terminal
        if step.index >= first and terminal is not None and terminal.startswith(TERMINAL_PREFIX):
            step.set_terminal(TERMINAL_PREFIX + out["output"])
            replaced += 1
    return reward, {"reward": reward, "terminals": replaced, "timed_out": sum(o["timed_out"] for o in outputs)}


def _correct(traj: Trajectory, edits: list[dict[str, Any]], reward: float) -> int | None:
    """The first turn the plan gives other text, as a demonstration; its number, or None."""
    for number in sorted({e["step"] for e in edits if e["op"] in ("edit_thinking", "set_command")}):
        steps = copy.deepcopy(traj.steps)
        scratch = Trajectory(traj.task, traj.reward, steps, None, [], None)
        apply_plan(scratch, [e for e in edits if e["step"] == number and e["op"] != "set_terminal"])
        if traj.demonstrate(traj.steps[number - 1], steps[number - 1].action, reward):
            return number
    return None


def task_tests(traj: Trajectory) -> str | None:
    """The files of the task's tests/ directory as text, found through the Harbor trial's config.json (task.path)."""
    try:
        task = Path(json.loads((Path(traj.trial_dir or "") / "config.json").read_text())["task"]["path"])
    except (OSError, KeyError, TypeError, ValueError):
        return None
    parts = []
    for file in sorted(p for p in (task / "tests").glob("*") if p.is_file()):
        try:
            parts.append(f"### {file.name}\n{clip(file.read_text(), TESTS_CHARS)}")
        except (OSError, UnicodeDecodeError):
            continue
    return "\n\n".join(parts) or None


def check_reroll(traj: Trajectory, answer: dict[str, Any], max_change_share: float | None) -> None:
    """Raise ValueError unless the answer names a reroll step (or null) after every edit, and its edits keep to the
    plan rules (check_plan; ``passes`` does not apply: the reroll's verifier decides)."""
    n = len(traj.steps)
    reroll = answer.get("reroll", "missing")
    if reroll is not None and not (isinstance(reroll, int) and not isinstance(reroll, bool) and 1 <= reroll <= n):
        raise ValueError(f"'reroll' must be a step number from 1 to {n}, or null; got {reroll!r}")
    edits = answer.get("edits")
    if reroll is None:
        if edits:
            raise ValueError("edits without a reroll step: with \"reroll\": null, give no edits")
        return
    check_plan(traj, {**answer, "passes": False}, n, 0, 1.0, False, max_change_share, REROLL_OPS, execute=True)
    late = sorted({e["step"] for e in edits if e["step"] >= reroll})
    if late:
        raise ValueError(f"steps {late} edited at or after the reroll step {reroll}: the agent writes those itself")


async def teacher_choice(traj: Trajectory, version: str = "reroll_v1", obs_chars: int = 3000,
                         change_share: float | None = None) -> int | None:
    """The teacher's reroll step (0-based) with its corrections before it applied to ``traj``, or None (the teacher
    chose none, or the trial is too long for it); the decision in ``traj.record["rewrite"]``. Used by ``added="reroll"``
    and by the branch template's ``teacher`` selection (transforms/branch.py)."""
    request = build_prompt(traj, len(traj.steps), 0, obs_chars, version, None, change_share, None, REROLL_ANSWER)
    if len(request) > PROMPT_CHARS:
        traj.record["rewrite"] = {"edited": False, "why": f"too long for the teacher ({len(request)} characters)"}
        return None

    def accept(content: str) -> dict[str, Any]:  # raises to have the teacher asked again
        answer = parse_answer(content)
        check_reroll(traj, answer, change_share)
        return answer

    answer = await ask(request, check=accept)
    if answer["reroll"] is not None:
        apply_plan(traj, answer["edits"])
    traj.record["rewrite"] = {"mode": "reroll", "edited": bool(answer["edits"]), "reroll": answer["reroll"],
                              "prompt": version, "prompt_sha": hashlib.sha256(criteria(version).encode()).hexdigest()[:12],
                              "steps": len(traj.steps), "max_change_share": change_share,
                              "edits": [{k: e[k] for k in ("step", "op", "why") if k in e} for e in answer["edits"]],
                              "reason": answer.get("reason")}
    return None if answer["reroll"] is None else answer["reroll"] - 1


async def _reroll(traj: Trajectory, version: str, obs_chars: int, change_share: float | None, n: int,
                  credit: str) -> Trajectory:
    """``added="reroll"``: the teacher's reroll step and corrections before it (module docstring)."""
    k = await teacher_choice(traj, version, obs_chars, change_share)
    if k is not None:
        traj.steps[k].reroll(n, credit)
    return traj


def parse_answer(content: str) -> dict[str, Any]:
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)
    start, end = content.find("{"), content.rfind("}") + 1
    if not 0 <= start < end:
        raise ValueError("no JSON object in the answer")
    answer = json.loads(content[start:end])
    if not isinstance(answer, dict):
        raise ValueError("the answer is not a JSON object")  # noqa: TRY004 -- a malformed answer: ask again
    return answer


async def rewrite(traj: Trajectory, prompt: str = "v2", max_step_share: float = 0.3, max_char_share: float = 0.3,
                  protect_last: int = 2, obs_chars: int = 3000, failures: str = "keep",
                  fix_prompt: str = "fix_v1", max_change_share: float | None = None, ops: str | None = None,
                  mask_new: bool = False, max_removed_share: float | None = None, successes: str = "rewrite",
                  fix_max_change_share: float | None = None, apply_edits: bool = True, clean: bool = True,
                  share: float = 1.0, added: str | None = None, added_reward: float = 1.0,
                  execute: bool = False, reroll_prompt: str = "reroll_v1", reroll_successes: bool = False,
                  reroll_n: int = 1, reroll_credit: str = "member") -> Trajectory:
    """``failures="fix"`` (for PPO): a failed trial goes to the teacher too, with the task's tests in view and the
    ``fix_prompt`` criteria, without step or text budgets (unless ``fix_max_change_share``); if the teacher expects
    the rewrite to pass, the trial's reward becomes 1. The criteria forbid using the tests in the agent's reasoning or
    actions. ``successes="keep"`` (or ``clean=False``): a successful trial is not sent to the teacher.
    ``apply_edits=False``: the plan is not applied; the steps it deletes or gives other commands leave the loss, so the
    same selection trains in the policy's own context. ``share``: the part of the trials sent to the teacher (by a
    hash of the trial).
    ``added`` keeps the trial as sampled and trains the teacher's work besides it, as one more member of the prompt's
    GRPO group with ``added_reward``: "correct", the teacher's tokens in the first turn it changes, after the trial's
    own text up to there; "twin", the whole rewritten trial.
    ``execute`` (with "twin"): the rewritten trial's commands run in a fresh sandbox, the task's verifier gives its reward,
    pass or fail, and from the first edit on the terminal outputs are the real ones, none the teacher's.
    ``added="reroll"``: the teacher picks a turn to play again from (module docstring)."""
    if added not in (None, "correct", "twin", "reroll"):
        raise ValueError(f"added must be None, 'correct', 'twin' or 'reroll', got {added!r}")
    if failures not in ("keep", "fix"):
        raise ValueError(f"failures must be 'keep' or 'fix', got {failures!r}")
    if successes not in ("rewrite", "keep"):
        raise ValueError(f"successes must be 'rewrite' or 'keep', got {successes!r}")
    n = len(traj.steps)
    if added == "reroll":
        why = ("no reward" if traj.reward is None else "successful trial (reroll_successes off)"
               if traj.reward >= 1.0 and not reroll_successes else "split by a summary" if len(traj.samples) > 1
               else f"outside the teacher's share {share}" if share < 1.0 and int(
                   hashlib.sha256(str(traj.trial_dir).encode()).hexdigest()[:8], 16) / 2**32 >= share else None)
        if why:
            traj.record["rewrite"] = {"mode": "reroll", "edited": False, "why": why}
            return traj
        return await _reroll(traj, reroll_prompt, obs_chars, fix_max_change_share, reroll_n, reroll_credit)
    fix = traj.reward is not None and traj.reward < 1.0 and failures == "fix"
    if traj.reward is None or (traj.reward < 1.0 and not fix):
        traj.record["rewrite"] = {"edited": False, "why": "failed trial (left to mask+rewrite)"}
        return traj
    if not fix and (successes == "keep" or not clean):
        traj.record["rewrite"] = {"edited": False, "why": "successful trial kept as the policy wrote it"}
        return traj
    if share < 1.0 and int(hashlib.sha256(str(traj.trial_dir).encode()).hexdigest()[:8], 16) / 2**32 >= share:
        traj.record["rewrite"] = {"edited": False, "why": f"outside the teacher's share {share}"}
        return traj
    if len(traj.samples) > 1 or n - protect_last < 1:
        traj.record["rewrite"] = {"edited": False, "why": "split by a summary" if len(traj.samples) > 1
                                  else f"no editable step ({n} steps, the last {protect_last} protected)"}
        return traj
    tests = task_tests(traj) if fix else None
    if fix and tests is None:
        traj.record["rewrite"] = {"edited": False, "mode": "fix", "why": "the task's tests were not found"}
        return traj
    version = fix_prompt if fix else prompt
    budget, char_share = (n, 1.0) if fix else (max(1, int(max_step_share * n)), max_char_share)
    change_share = fix_max_change_share if fix else max_change_share
    allowed = None if fix or not ops else tuple(o.strip() for o in ops.split(","))  # e.g. "delete,edit_thinking" (v6)
    if allowed and not set(allowed) <= {*MODEL_OPS, "set_terminal"}:
        raise ValueError(f"unknown ops {ops!r}; choose from {', '.join((*MODEL_OPS, 'set_terminal'))}")
    removed_share = None if fix else max_removed_share
    request = build_prompt(traj, budget, protect_last, obs_chars, version, tests, change_share, removed_share)
    if len(request) > PROMPT_CHARS:
        traj.record["rewrite"] = {"edited": False, "why": f"too long for the teacher ({len(request)} characters)"}
        return traj

    def accept(content: str) -> dict[str, Any]:  # raises to have the teacher asked again
        answer = parse_answer(content)
        check_plan(traj, answer, budget, protect_last, char_share, fix, change_share, allowed, removed_share, execute)
        return answer

    answer = await ask(request, check=accept)  # no valid plan after the attempts: the trial is skipped
    extra = added is not None
    corrected = None
    if added == "correct":
        corrected = _correct(traj, answer["edits"], added_reward)
    executed = None
    if added == "twin" and answer["edits"]:
        twin = Trajectory.of(copy.deepcopy(traj.samples), traj.tokenizer)
        apply_plan(twin, [e for e in answer["edits"] if not (execute and e["op"] == "set_terminal")])
        reward = added_reward
        if execute:
            reward, executed = await _execute(twin, answer["edits"], traj.trial_dir)
        if reward is not None:
            twin._apply()
            traj._new_tokens += twin._new_tokens
            traj.add_sample(twin.samples[0], reward)
    if apply_edits and not extra:
        apply_plan(traj, answer["edits"])
        traj.mask_new_tokens = mask_new and bool(answer["edits"])
    elif not extra:
        for edit in answer["edits"]:
            step = traj.steps[edit["step"] - 1]
            if edit["op"] == "delete" or (edit["op"] == "set_command" and not _same_commands(step.command, edit["command"])):
                step.mask()
    fixed = fix and bool(answer["edits"]) and answer["passes"] and not extra
    if fixed:
        traj.reward = 1.0  # the teacher's verdict; the original reward stays in the transform record
    traj.record["rewrite"] = {"edited": bool(answer["edits"]), "mode": "fix" if fix else "clean", "fixed": fixed,
                              "prompt": version, "prompt_sha": hashlib.sha256(criteria(version).encode()).hexdigest()[:12],
                              "steps": n, "budget": budget, "max_change_share": change_share, "ops": ops,
                              "mask_new": mask_new, "max_removed_share": removed_share,
                              "applied": apply_edits and not extra, "corrected_step": corrected, "executed": executed,
                              "edits": [{k: e[k] for k in ("step", "op", "why") if k in e} for e in answer["edits"]],
                              "passes": answer["passes"], "reason": answer.get("reason")}
    return traj
