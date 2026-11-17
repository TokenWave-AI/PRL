"""The step API of a trajectory transform: what traj is and what a transform may do to it (the framework that runs
it: prl/transforms/core.py).

What a transform is: a file ``transforms/<name>.py`` at the repo root defining ``<name>(traj, **params) -> traj |
None``, sync or async; ``<name>`` is how runs select it (e.g. transforms/atlas.py). It maps {trajectory, reward} to
{trajectory, reward}: ``traj`` (``Trajectory``) is a text view of the trial: ``task`` (its first prompt), ``reward``
(writable), ``steps`` (one per model turn, in order), ``trial_dir`` and ``record`` (a dict saved with the trial). A
step's ``action`` / ``observation`` are the model's output and the environment's message after it, as text;
``thinking``, ``command`` (the action JSON) and ``terminal`` (the message without its chat-template wrapper) are their
parts. The function may
  - ``step.mask()``: the step's tokens leave the loss (and its per-token normalization);
  - ``step.weight = w``: the step's advantage is scaled by ``w`` (``prl/algo/step_advantage.py``); a negative
    ``w`` reverses the push (e.g. a mistake in a successful trial is pushed down: transforms/atlas_signed.py);
  - ``step.weight_text(text, w[, every])``: token level: the tokens of ``text`` (a piece of the step's action as
    generated, occurring once in it, or with ``every`` each occurrence) get ``w`` instead of the step's weight (``train_metadata["token_weight_spans"]``: response
    token ranges [start, end, w]; a token only partly inside ``text`` counts as inside). Not in a call that also
    rewrites, and a later rewrite of such a trial is refused (the ranges would move);
  - rewrite: ``step.edit_thinking(old, new)`` (a minimal edit), ``step.set_thinking``, ``step.set_command``,
    ``step.set_terminal`` (e.g. a simulated terminal output), ``step.delete()`` (the turn and the message after it),
    ``traj.reward = r``. The framework re-tokenizes only the changed text and keeps the policy's own tokens around it
    (``_splice``). Rewritten text was not sampled by the policy, so it needs --use-tis and --use-rollout-logprobs off.
    A changed reward is normalized with the rest of the prompt's group (GRPO): turning a group's failures into
    successes can leave the group without signal. Not for trials split by a summary (several leaves); the last turn
    cannot be deleted;
  - ``step.reroll(n=1, credit="member")``: the policy generates again from this turn on, ``n`` times (branches;
    ``Trajectory._reroll_request``). After the chain, for each branch a fresh sandbox of the task runs the turns before
    it, as they now stand (a rewrite of them reaches the branches only: the trial itself trains as generated, with its
    masks and weights), and the policy continues in a new session (prl/rollout/harbor_training.py,
    prl/rollout/reroll_agent.py). A branch's turns before ``step`` are context; it trains with its own verifier
    reward (prl/algo/added_rewards.py), as ``credit`` says: ``member``, one more member of the prompt's group;
    ``subgroup``, against its siblings only (the other branches of this step and the trial from this step on, whose
    tokens from here on then take that advantage; GRPO only). Several steps of one trial may branch (one transform of
    the chain asks for all of them); not at or after a rewritten turn, not for trials split by a summary;
  - ``traj.skip(reason)``: leave the trial as it is;
  - ``await ask(prompt, check)``: the run's model (see ``ask``); every call is saved with the trial, and no acceptable
    answer leaves the trial as it is (skipped). A function changes nothing unless it returns normally, so a
    multi-step pipeline is all or nothing;
  - advanced: change ``traj.samples`` (the Miles samples) directly; then not the step API in the same call, and every
    sample must stay consistent (tokens, response_length, loss_mask, rollout_log_probs, weight_versions).
A sync function runs in a worker thread, so a blocking call (an HTTP request) does not stall the other trials.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from prl.transforms.tokens import THINK_END, TURN_END, USER_START, _moved, _runs, _splice

# Branches per rerolled step: each is a whole new trial (sandbox, replay, generation)
MAX_BRANCHES = 16


CREDITS = ("member", "subgroup")  # how a branch trains (prl/algo/added_rewards.py)


UNREBUILT = ("rollout_sampling_mask", "teacher_log_probs", "opd_reverse_kl", "rollout_routed_experts",
             "rollout_indexer_topk")


@dataclass
class Step:
    """One model turn: a maximal run of trained tokens (loss_mask 1) in one sample.

    ``action`` / ``observation`` are its text as it now stands, decoded with special tokens: the model's output and the
    untrained span after it (the environment's message in its chat-template wrapper; "" after the last step).
    ``thinking``, ``command`` and ``terminal`` are their parts; the edit methods change the text, and the framework
    re-tokenizes only what changed."""

    index: int  # 0-based, in commit order over the whole trial
    action: str
    observation: str
    weight: float = 1.0
    masked: bool = False
    deleted: bool = False
    rerolls: int = 0  # branches asked for from this step (reroll)
    credit: str = "member"  # how they train (reroll)
    _sample: int = field(default=0, repr=False)  # samples[_sample].loss_mask[_start:_end] (response-relative)
    _start: int = field(default=0, repr=False)
    _end: int = field(default=0, repr=False)
    _original: tuple[str, str] = field(default=("", ""), repr=False)  # (action, observation) as generated
    _spans: list[tuple[int, int, float]] = field(default_factory=list, repr=False)  # (char start, char end, weight)

    def mask(self) -> None:
        self.masked = True

    def weight_text(self, text: str, w: float, every: bool = False) -> None:
        """Token-level weight: the tokens of ``text``, which must occur exactly once in the action as generated (or, with
        ``every``, at least once: then every occurrence), get ``w`` instead of the step's weight (e.g. only the wrong
        command of a bad step is pushed down, also where the reasoning repeats it)."""
        if (self.action, self.observation) != self._original:
            raise ValueError(f"step {self.index}: weight_text on a rewritten step")
        found = self.action.count(text) if text else 0
        if not (found == 1 or (every and found > 1)):
            raise ValueError(f"step {self.index}: {text[:80]!r} must occur {'at least' if every else 'exactly'} once "
                             f"in the action, found {found}")
        if not (isinstance(w, (int, float)) and math.isfinite(w)):
            raise ValueError(f"step {self.index}: weight must be a finite number, got {w!r}")
        start = self.action.find(text)
        while start >= 0:  # non-overlapping occurrences, as str.count counts them
            self._spans.append((start, start + len(text), float(w)))
            start = self.action.find(text, start + len(text))

    def delete(self) -> None:
        """Remove this turn and the environment message after it."""
        self.deleted = True

    def reroll(self, n: int = 1, credit: str = "member") -> None:
        """The policy generates again from this turn on, ``n`` times, after the turns before it as they now stand;
        ``credit``: ``member`` or ``subgroup`` (module docstring). A second call on the step replaces the first."""
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_BRANCHES:
            raise ValueError(f"step {self.index}: reroll n must be an integer in 1..{MAX_BRANCHES}, got {n!r}")
        if credit not in CREDITS:
            raise ValueError(f"step {self.index}: reroll credit must be one of {CREDITS}, got {credit!r}")
        self.rerolls, self.credit = n, credit

    @property
    def rerolled(self) -> bool:
        return self.rerolls > 0

    @property
    def thinking(self) -> str | None:
        i = self.action.find(THINK_END)
        return None if i < 0 else self.action[:i]

    def _json_span(self) -> tuple[int, int] | None:
        i = self.action.find(THINK_END)
        start, end = self.action.find("{", 0 if i < 0 else i + len(THINK_END)), self.action.rfind("}") + 1
        return (start, end) if 0 <= start < end else None

    @property
    def command(self) -> dict[str, Any] | None:
        """The action JSON (Terminus-2: analysis, plan, commands, task_complete); None if it does not parse."""
        span = self._json_span()
        try:
            value = None if span is None else json.loads(self.action[span[0]:span[1]])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def _terminal_span(self) -> tuple[int, int] | None:
        i, j = self.observation.find(USER_START), self.observation.rfind(TURN_END)
        return (i + len(USER_START), j) if 0 <= i and i + len(USER_START) <= j else None

    @property
    def terminal(self) -> str | None:
        """The environment's message after this turn, without the chat-template wrapper; None after the last turn."""
        span = self._terminal_span()
        return None if span is None else self.observation[span[0]:span[1]]

    def edit_thinking(self, old: str, new: str) -> None:
        """A minimal edit: ``old``, which must occur exactly once in the thinking, becomes ``new``."""
        thinking = self.thinking
        if thinking is None or thinking.count(old) != 1:
            raise ValueError(f"step {self.index}: {old[:80]!r} must occur once in the thinking, "
                             f"found {0 if thinking is None else thinking.count(old)}")
        self.action = thinking.replace(old, new, 1) + self.action[len(thinking):]

    def set_thinking(self, text: str) -> None:
        thinking = self.thinking
        if thinking is None:
            raise ValueError(f"step {self.index} has no thinking")
        self.action = text + self.action[len(thinking):]

    def set_command(self, command: dict[str, Any] | str) -> None:
        """Replace the action JSON; a dict is written as the model writes it (indent 2)."""
        span = self._json_span()
        if span is None:
            raise ValueError(f"step {self.index} has no action JSON")
        body = command if isinstance(command, str) else json.dumps(command, indent=2, ensure_ascii=False)
        self.action = self.action[:span[0]] + body + self.action[span[1]:]

    def set_terminal(self, text: str) -> None:
        """Replace the environment's message after this turn (e.g. a simulated terminal output)."""
        span = self._terminal_span()
        if span is None:
            raise ValueError(f"step {self.index} has no environment message after it")
        self.observation = self.observation[:span[0]] + text + self.observation[span[1]:]

    def _state(self) -> tuple[Any, ...]:
        return (self.weight, self.masked, self.deleted, self.action, self.observation, tuple(self._spans),
                self.rerolls, self.credit)

    def _rewritten(self) -> bool:
        return self.deleted or (self.action, self.observation) != self._original


@dataclass
class Trajectory:
    task: str
    reward: float | None
    steps: list[Step]
    trial_dir: str | None
    samples: list[Any] = field(repr=False)
    tokenizer: Any = field(repr=False)
    record: dict[str, Any] = field(default_factory=dict)
    skipped: str | None = None
    _initial: list[tuple[Any, ...]] = field(default_factory=list, repr=False)
    _initial_reward: float | None = field(default=None, repr=False)
    _new_tokens: int = field(default=0, repr=False)  # trained tokens the policy did not sample (set by _render)
    # True: text a transform wrote stays in the context but out of the loss, so only the policy's own tokens are trained
    # (rewrite+mask; set by the transform). It splits a turn's trained run, so no transform with step weights follows.
    mask_new_tokens: bool = False
    _extras: list[Any] = field(default_factory=list, repr=False)  # added to the trial's samples after _apply
    _reroll: dict[str, Any] | None = field(default=None, repr=False)  # set by _apply when steps are rerolled

    def skip(self, reason: str) -> None:
        self.skipped = reason

    def add_sample(self, sample: Any, reward: float) -> None:
        """Train ``sample`` besides the trial's own, with this reward (prl/algo/added_rewards.py): under GRPO one
        more member of the prompt's group, under PPO against the critic's values; the critic does not learn from it
        (step_advantage). ``sample.reward`` stays the trial's, so the trial's reward metric does not count it."""
        sample.train_metadata = {**(getattr(sample, "train_metadata", None) or {}), "group_reward": float(reward)}
        self._extras.append(sample)

    def demonstrate(self, step: Step, action: str, reward: float) -> bool:
        """A correction at a state the policy visited: a new sample with the trial's text up to ``step`` and then
        ``action``, ending at the last token the policy did not sample; only those tokens train. False if the action
        writes none."""
        s = self.samples[step._sample]
        prompt = len(s.tokens) - s.response_length
        response = list(s.tokens[prompt:])
        piece, origin = _splice(self.tokenizer, response[step._start:step._end], step._original[0], action)
        new = [i for i, o in enumerate(origin) if o < 0]
        if not new:
            return False
        piece, origin = piece[:new[-1] + 1], origin[:new[-1] + 1]
        out = copy.copy(s)
        out.metadata = dict(s.metadata or {})
        out.tokens = list(s.tokens[:prompt]) + response[:step._start] + piece
        out.response_length = step._start + len(piece)
        out.loss_mask = [0] * step._start + [int(o < 0) for o in origin]
        if getattr(s, "rollout_log_probs", None) is not None:  # as _render: 0 for text the policy did not sample
            out.rollout_log_probs = list(s.rollout_log_probs[:step._start]) + [0.0] * len(piece)
        calls = list(getattr(s, "weight_versions", None) or [])
        if calls:
            k = sum(1 for st in self.steps if st._sample == step._sample and st.index < step.index)
            out.weight_versions = calls[:k] + [_moved(calls[k], prompt + step._start, prompt + out.response_length, 0, True)]
        if hasattr(out, "response"):
            out.response = self.tokenizer.decode(out.tokens[prompt:])
        self._new_tokens += len(new)
        self.add_sample(out, reward)
        return True

    @classmethod
    def of(cls, samples: list[Any], tokenizer: Any) -> Trajectory:
        """The trial's steps in commit order (leaves sorted by their session node id, as committed)."""
        order = sorted(range(len(samples)), key=lambda i: (samples[i].metadata or {}).get("leaf", {}).get("node_id", i))
        first = samples[order[0]]
        steps: list[Step] = []
        for i in order:
            s = samples[i]
            response = s.tokens[len(s.tokens) - s.response_length:]
            runs = _runs(list(s.loss_mask))
            # Weights an earlier transform of the chain set, one per trained run (step_advantage's segmentation)
            weights = iter((getattr(s, "train_metadata", None) or {}).get("step_weights") or [])
            for k, (trained, start, end) in enumerate(runs):
                if not trained:
                    continue
                following = runs[k + 1] if k + 1 < len(runs) else None
                action = tokenizer.decode(response[start:end])
                observation = tokenizer.decode(response[following[1]:following[2]]) if following else ""
                steps.append(Step(len(steps), action, observation, weight=float(next(weights, 1.0)), _sample=i,
                                  _start=start, _end=end, _original=(action, observation)))
        reward = next((s.metadata["reward"] for s in samples if (s.metadata or {}).get("reward") is not None), None)
        reward = None if reward is None else float(reward)
        trial_dir = next((s.metadata["trial_dir"] for s in samples if (s.metadata or {}).get("trial_dir")), None)
        return cls(tokenizer.decode(first.tokens[: len(first.tokens) - first.response_length]), reward, steps,
                   trial_dir, samples, tokenizer, _initial=[s._state() for s in steps], _initial_reward=reward)

    def _edited(self) -> bool:
        return [s._state() for s in self.steps] != self._initial or self.reward != self._initial_reward

    def _apply(self) -> None:
        """The step decisions into the samples: rewritten turns re-tokenized (``_render``); masked runs zeroed; with any
        weight other than 1, every sample gets ``train_metadata["step_weights"]``, one weight per run still trained
        (step_advantage's segmentation); a changed reward on every sample."""
        for step in self.steps:
            if not (isinstance(step.weight, (int, float)) and math.isfinite(step.weight)):
                raise ValueError(f"step {step.index}: weight must be a finite number, got {step.weight!r}")
        rerolled = [s for s in self.steps if s.rerolled]
        if rerolled:
            self._reroll = self._reroll_request(rerolled)
        rewritten = {s._sample for s in self.steps if s._rewritten()}
        spanned = [s for s in self.steps if s._spans]
        if rewritten and spanned:
            raise ValueError("weight_text and rewriting in one call: the token ranges would move")
        if rewritten:
            if any((getattr(self.samples[i], "train_metadata", None) or {}).get("token_weight_spans") for i in rewritten):
                raise ValueError("rewriting a trial an earlier transform gave token weights (weight_text)")
            self._render(rewritten)
        live = [s for s in self.steps if not s.deleted]
        weighted = any(s.weight != 1.0 for s in live if not s.masked)
        by_sample: dict[int, list[Step]] = {}
        for step in live:
            by_sample.setdefault(step._sample, []).append(step)
            if step.masked:
                self.samples[step._sample].loss_mask[step._start:step._end] = [0] * (step._end - step._start)
        if weighted:
            for i, steps in by_sample.items():
                sample = self.samples[i]
                sample.train_metadata = {**(sample.train_metadata or {}),
                                         "step_weights": [float(s.weight) for s in steps if not s.masked]}
        for step in spanned:
            if step.masked or step.deleted:
                raise ValueError(f"step {step.index}: weight_text on a masked or deleted step")
            sample = self.samples[step._sample]
            ranges = [[*self._token_range(step, a, b), w] for a, b, w in step._spans]
            old = (sample.train_metadata or {}).get("token_weight_spans") or []
            sample.train_metadata = {**(sample.train_metadata or {}), "token_weight_spans": [*old, *ranges]}
        if self.reward != self._initial_reward:
            if not (isinstance(self.reward, (int, float)) and math.isfinite(self.reward)):
                raise ValueError(f"reward must be a finite number, got {self.reward!r}")
            for sample in self.samples:  # Miles normalizes sample.reward within the prompt's group (GRPO)
                if isinstance(getattr(sample, "reward", None), dict):
                    raise TypeError("a dict reward (--reward-key) cannot be rewritten")
                sample.reward = float(self.reward)
                sample.metadata["reward"] = float(self.reward)

    def _reroll_request(self, rerolled: list[Step]) -> dict[str, Any]:
        """What the reroll agent replays, per rerolled step (``anchors``, in step order): the first user message and,
        for each live turn before that step, its thinking and reply as messages (the chat template renders them back
        to the policy's context) and the message that followed it as generated (the agent compares the executed one
        with it); with the step's ``n`` and ``credit``, and ``token_start``, the step's first response token in the
        trial's sample (a subgroup's advantage starts there). The turns' rewrites go into this request only; the
        trial's own text is restored, so it trains as generated."""
        if len(self.samples) > 1:
            raise ValueError("reroll: not for a trial split by a summary")
        rewritten = [s.index for s in self.steps if s._rewritten()]
        start = self.task.find(USER_START)
        end = self.task.find(TURN_END, start)
        if start < 0 or end < 0:
            raise ValueError("reroll: no user message in the prompt")
        instruction = self.task[start + len(USER_START):end].strip()
        sample = self.samples[0]
        prompt = len(sample.tokens) - sample.response_length
        turns: list[dict[str, Any]] = []
        for s in self.steps[:max(at.index for at in rerolled)]:  # the replay of every anchor, built once
            if s.deleted:
                continue
            i = s.action.find(THINK_END)
            if i < 0 or not s.action.endswith(TURN_END):
                raise ValueError(f"step {s.index}: no {THINK_END} or {TURN_END}; it cannot be replayed as a message")
            turns.append({"step": s.index, "reasoning": s.action[:i].strip(),
                          "content": s.action[i + len(THINK_END):-len(TURN_END)].strip(), "message": s.terminal})
        anchors = []
        for at in sorted(rerolled, key=lambda s: s.index):
            if at.deleted:
                raise ValueError(f"step {at.index}: a deleted step cannot be rerolled")
            late = [k for k in rewritten if k >= at.index]
            if late:
                raise ValueError(f"steps {late}: rewritten at or after the rerolled step {at.index}; the policy writes "
                                 "those")
            anchors.append({
                "step": at.index, "n": at.rerolls, "credit": at.credit, "instruction": instruction,
                "turns": [t for t in turns if t["step"] < at.index], "trial_dir": self.trial_dir,
                # The trial's tokens through the last replayed reply: the agent's context count until the server gives one
                "context_tokens": prompt + (self.steps[at.index - 1]._end if at.index else 0),
                "token_start": at._start, "rewritten_steps": [k for k in rewritten if k < at.index]})
        for s in self.steps[:anchors[-1]["step"]]:
            s.action, s.observation = s._original
            s.deleted = False
        return {"anchors": anchors}

    def _token_range(self, step: Step, a: int, b: int) -> tuple[int, int]:
        """The response token range [start, end) that covers characters [a, b) of the step's action. The action is the
        decode of the step's tokens, so the decoded prefix length grows with the token count: binary search on it."""
        s = self.samples[step._sample]
        tokens = s.tokens[len(s.tokens) - s.response_length:][step._start:step._end]
        def length(k: int) -> int:
            return len(self.tokenizer.decode(tokens[:k]))

        def first(pred: Callable[[int], bool]) -> int:  # the smallest k in [0, n] with pred(k), n if none
            lo, hi = 0, len(tokens)
            while lo < hi:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if pred(mid) else (mid + 1, hi)
            return lo
        k0 = max(0, first(lambda k: length(k) > a) - 1)  # the token whose text holds character a
        k1 = first(lambda k: length(k) >= b)
        text = step.action[a:b]
        if k0 >= k1 or text not in self.tokenizer.decode(tokens[k0:k1]):
            raise ValueError(f"step {step.index}: could not map {text[:80]!r} onto its tokens")
        return step._start + k0, step._start + k1

    def _render(self, which: set[int]) -> None:
        """Re-tokenize the rewritten samples. Unchanged turns and messages keep their tokens and rollout log probs;
        changed text is encoded anew (log probs 0: the policy did not sample it); a deleted turn goes with the message
        after it. Weight-version spans follow their turns."""
        if len(self.samples) > 1:
            raise ValueError("rewriting a trial split by a summary (several leaves) is not supported")
        if self.steps[-1].deleted:
            raise ValueError("the last turn (the task's completion) cannot be deleted")
        for i in which:
            s = self.samples[i]
            # Per-token fields this renderer does not rebuild (None in our runs: no OPD, R3, sampling masks)
            other = [f for f in UNREBUILT if getattr(s, f, None) is not None]
            if other:
                raise ValueError(f"cannot rewrite a sample with {other}")
            prompt = len(s.tokens) - s.response_length
            response = s.tokens[prompt:]
            log_probs = getattr(s, "rollout_log_probs", None)
            calls = list(getattr(s, "weight_versions", None) or [])
            steps = [st for st in self.steps if st._sample == i]
            if calls and len(calls) != len(steps):
                raise ValueError(f"{len(calls)} weight-version calls for {len(steps)} turns")
            ids: list[int] = []
            mask: list[int] = []
            lps: list[float] = []
            new_calls = []
            owner: Step | None = None  # the turn whose environment message the next untrained run is
            k = 0
            for trained, a, b in _runs(list(s.loss_mask)):
                if trained:
                    st = owner = steps[k]
                    call = calls[k] if calls else None
                    k += 1
                    if st.deleted:
                        continue
                    piece, origin = _splice(self.tokenizer, list(response[a:b]), st._original[0], st.action)
                    st._start, st._end = len(ids), len(ids) + len(piece)
                    if not st.masked and not self.mask_new_tokens:
                        self._new_tokens += sum(1 for o in origin if o < 0)
                    if call is not None:
                        new_calls.append(_moved(call, prompt + st._start, prompt + st._end, st._start - a,
                                                origin != list(range(b - a))))
                else:
                    if owner is not None and owner.deleted:
                        continue
                    old, now = (owner._original[1], owner.observation) if owner is not None else ("", "")
                    piece, origin = _splice(self.tokenizer, list(response[a:b]), old, now)
                ids += piece
                mask += [int(trained and (o >= 0 or not self.mask_new_tokens)) for o in origin]
                if log_probs is not None:  # 0 for text the policy did not sample (TIS is off for rewrites)
                    lps += [float(log_probs[a + o]) if o >= 0 else 0.0 for o in origin]
            s.tokens = list(s.tokens[:prompt]) + ids
            s.response_length = len(ids)
            s.loss_mask = mask
            if log_probs is not None:
                s.rollout_log_probs = lps
            if calls:
                s.weight_versions = new_calls
            if hasattr(s, "response"):
                s.response = self.tokenizer.decode(ids)
