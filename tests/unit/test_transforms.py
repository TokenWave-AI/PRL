"""prl/transforms + transforms/atlas.py: selection by name, the trajectory view, writing step decisions back,
checks, fallbacks, ATLAS."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from prl.algo import llm_client
from prl.transforms import core
from prl.transforms.core import Trajectory
from transforms import atlas as atlas_module
from transforms import atlas_signed as atlas_signed_module


class CharTokenizer:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _sample(prompt, parts, node_id=0, reward=1.0, trial_dir=None):
    """parts: (text, trained) spans of the response, in order."""
    response = [ord(c) for text, _ in parts for c in text]
    return SimpleNamespace(tokens=[ord(c) for c in prompt] + response, response_length=len(response),
                           loss_mask=[t for text, t in parts for _ in text], reward=reward, rollout_id=7,
                           rollout_log_probs=[0.0] * len(response), train_metadata=None,
                           metadata={"leaf": {"node_id": node_id}, "reward": reward, "trial_dir": trial_dir})


def _trial(trial_dir=None):
    """Steps A, B in the leaf committed first (node 4), C in the later one (node 9); listed in the other order."""
    return [_sample("P2", [("C", 1), ("o3", 0)], node_id=9, trial_dir=trial_dir),
            _sample("TASK", [("A", 1), ("o1", 0), ("BB", 1), ("o2", 0)], node_id=4, trial_dir=trial_dir)]


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for key in ("PRL_TRANSFORM", "PRL_TRANSFORM_ARGS", "PRL_TRANSFORM_MAX_FAILURES",
                "PRL_TRANSFORM_TIMEOUT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(core, "_failures", 0)


@pytest.fixture
def use(monkeypatch):
    """Select a test function as if it were transforms/<name>.py."""
    fns = {}
    real = core._resolve
    monkeypatch.setattr(core, "_resolve", lambda name: fns[name] if name in fns else real(name))

    def select(fn, args=None, name="t"):
        fns[name] = fn
        monkeypatch.setenv("PRL_TRANSFORM", name)
        if args is not None:
            monkeypatch.setenv("PRL_TRANSFORM_ARGS", json.dumps(args))
    return select


def _run(samples, args=None):
    return asyncio.run(core.run(samples, CharTokenizer(), args))


def test_no_transform_by_default(monkeypatch):
    samples = _trial()
    assert core.configured() == [] and _run(samples) is samples
    monkeypatch.setenv("PRL_TRANSFORM", "none")
    assert core.configured() == [] and samples[0].train_metadata is None


def test_steps_in_commit_order():
    traj = Trajectory.of(_trial("/t"), CharTokenizer())
    assert (traj.task, traj.reward, traj.trial_dir) == ("TASK", 1.0, "/t")
    assert [(s.index, s.action, s.observation, s._sample) for s in traj.steps] == [
        (0, "A", "o1", 1), (1, "BB", "o2", 1), (2, "C", "o3", 0)]


def test_mask_and_weights_are_written_back(use, tmp_path):
    def fn(traj):
        traj.steps[1].weight = 0.5
        traj.steps[2].mask()
        traj.record["note"] = "x"

    use(fn)
    kept = _run(_trial(str(tmp_path)))
    assert [s.loss_mask for s in kept] == [[1, 0, 0, 1, 1, 0, 0]]  # the leaf with C has no trained token left
    assert kept[0].train_metadata == {"step_weights": [1.0, 0.5]}
    record = json.loads((tmp_path / "transform.json").read_text())
    assert kept[0].metadata["prl_transform"] == record
    assert record["chain"][0]["status"] == "ok" and record["chain"][0]["record"] == {"note": "x"}
    assert (record["trained_tokens_before"], record["trained_tokens_after"], record["tokens_changed"]) == (4, 3, False)
    assert record["dropped_samples"] == 1
    entry = record["chain"][0]
    assert (entry["weighted_steps"], entry["masked_steps"]) == ({"1": 0.5}, [2]) and "token_weighted_steps" not in entry


def test_the_record_names_only_what_this_function_changed(monkeypatch):
    weights = {"first": (lambda traj: setattr(traj.steps[0], "weight", -1.0)),
               "same": (lambda traj: setattr(traj.steps[0], "weight", -1.0)),
               "span": (lambda traj: traj.steps[1].weight_text("BB", 0.0))}
    monkeypatch.setattr(core, "_resolve", lambda name: weights[name])
    monkeypatch.setenv("PRL_TRANSFORM", "first,same,span")
    chain = _run(_trial())[0].metadata["prl_transform"]["chain"]
    # the second sets the weight the first left: no change of its own
    assert [(e.get("weighted_steps"), e.get("token_weighted_steps")) for e in chain] == [
        ({"0": -1.0}, None), (None, None), (None, [1])]


def test_weights_follow_the_runs_left_after_masking(use):
    def fn(traj):
        traj.steps[0].mask()
        traj.steps[1].weight = 0.3

    use(fn)
    kept = _run(_trial())
    # step_advantage segments the final loss_mask: the masked run is gone, so each weight list covers what remains
    assert [s.train_metadata.get("step_weights") for s in kept] == [[1.0], [0.3]]


def test_selection_params_and_chain(use, monkeypatch, tmp_path):
    def scale(traj, w=1.0):
        traj.steps[0].weight = w

    use(scale, {"w": 0.2})
    assert core.configured()[0][2] == {"w": 0.2}
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"nope": 1}')
    with pytest.raises(ValueError, match="do not fit"):
        core.configured()
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", "[1]")
    with pytest.raises(ValueError, match="JSON object"):
        core.configured()
    monkeypatch.delenv("PRL_TRANSFORM_ARGS")
    monkeypatch.setenv("PRL_TRANSFORM", "nope")
    with pytest.raises(ValueError, match="available: .*atlas"):  # the error lists transforms/
        core.configured()
    # By name from the transforms package (here a temporary one); a chain takes its params keyed by name
    package = tmp_path / "my_transforms"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "halve.py").write_text("def halve(traj, w=0.5):\n    traj.steps[-1].weight = w\n")
    (package / "nofn.py").write_text("x = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(core, "PACKAGE", "my_transforms")
    monkeypatch.setenv("PRL_TRANSFORM", "t, halve")
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"t": {"w": 0.2}, "halve": {"w": 0.4}}')
    kept = _run(_trial())
    assert [s.train_metadata["step_weights"] for s in kept] == [[0.4], [0.2, 1.0]]
    monkeypatch.setenv("PRL_TRANSFORM", "t+halve")  # submit.sh's form: pjsub -x splits on commas
    assert [name for name, _, _ in core.configured()] == ["t", "halve"]
    assert core.available() == ["halve", "nofn"]
    for name, error in (("nofn", "must define a function nofn"), ("a.b", "file name")):
        monkeypatch.setenv("PRL_TRANSFORM", name)
        monkeypatch.delenv("PRL_TRANSFORM_ARGS", raising=False)
        with pytest.raises(ValueError, match=error):
            core.configured()


def test_sync_functions_run_in_a_thread(use):
    seen = []
    use(lambda traj: seen.append(threading.get_ident()))
    _run(_trial())
    assert seen and seen[0] != threading.get_ident()


@pytest.mark.parametrize("bad", ["raise", "length", "mask_all", "other", "weight"])
def test_failures_keep_the_trial(use, bad, tmp_path):
    def fn(traj):
        if bad == "raise":
            raise RuntimeError("boom")
        if bad == "length":
            traj.samples[0].loss_mask.append(1)
        if bad == "mask_all":
            for step in traj.steps:
                step.mask()
        if bad == "weight":
            traj.steps[0].weight = float("nan")
        return "other" if bad == "other" else traj

    use(fn)
    kept = _run(_trial(str(tmp_path)))
    assert [s.loss_mask for s in kept] == [s.loss_mask for s in _trial()]
    assert all(s.train_metadata == {} for s in kept)
    entry = json.loads((tmp_path / "transform.json").read_text())["chain"][0]
    assert entry["status"] == "error" and entry["reason"]


def test_skip_keeps_the_trial(use):
    def fn(traj):
        traj.steps[0].mask()
        traj.skip("no label")

    use(fn)
    kept = _run(_trial())
    assert [s.loss_mask for s in kept] == [s.loss_mask for s in _trial()]
    assert kept[0].metadata["prl_transform"]["chain"][0] | {"seconds": 0} == {
        "transform": "t", "params": {}, "record": {}, "status": "skipped", "reason": "no label", "seconds": 0}


def _rewrite(new_response, mask=None, prompt=None, reward=None):
    def fn(traj):
        s = traj.samples[1]
        head = s.tokens[: len(s.tokens) - s.response_length] if prompt is None else [ord(c) for c in prompt]
        s.tokens = head + [ord(c) for c in new_response]
        s.response_length = len(new_response)
        s.loss_mask = mask or [1] * len(new_response)
        s.rollout_log_probs = [0.0] * len(new_response)
        if reward is not None:
            s.reward = reward
    return fn


def test_rewrite_tokens(use):
    use(_rewrite("XYZ"))
    kept = _run(_trial(), SimpleNamespace(use_tis=False, use_rollout_logprobs=False))
    assert kept[1].tokens == [ord(c) for c in "TASKXYZ"] and kept[1].metadata["prl_transform"]["tokens_changed"]


@pytest.mark.parametrize("case", ["tis", "prompt", "reward", "with_step_api"])
def test_rewrite_must_stay_consistent(use, case):
    fn = _rewrite("XYZ", prompt="OTHER" if case == "prompt" else None, reward=0.0 if case == "reward" else None)
    if case == "with_step_api":
        inner = fn

        def fn(traj):
            inner(traj)
            traj.steps[0].weight = 0.5

    use(fn)
    kept = _run(_trial(), SimpleNamespace(use_tis=case == "tis", use_rollout_logprobs=False))
    assert kept[1].tokens == _trial()[1].tokens
    assert kept[0].metadata["prl_transform"]["chain"][0]["status"] == "error"


def test_failures_in_a_row_stop_the_run(use, monkeypatch):
    monkeypatch.setenv("PRL_TRANSFORM_MAX_FAILURES", "3")
    use(lambda traj: traj.skip("down"))
    _run(_trial())
    _run(_trial())
    with pytest.raises(RuntimeError, match="3 trials in a row"):
        _run(_trial())
    core._failures = 0
    _run(_trial())
    use(lambda traj: None, name="noop")  # a trial left as it is on purpose neither counts nor resets
    _run(_trial())
    assert core._failures == 1
    use(lambda traj: traj.steps[0].mask(), name="mask")  # a changed trial resets
    _run(_trial())
    assert core._failures == 0


def test_an_outage_stops_a_transform_that_passes_most_trials_through(use, monkeypatch):
    """rewrite: failed trials pass unchanged (ok); if the teacher is down, the successes are skipped."""
    monkeypatch.setenv("PRL_TRANSFORM_MAX_FAILURES", "3")

    async def teacher_down(traj):
        if traj.reward == 1.0:
            await core.ask("x")  # no answer
        return traj

    use(teacher_down)

    async def post(cfg, messages):
        raise ConnectionError("down")

    monkeypatch.setattr(llm_client, "_post", post)
    monkeypatch.setattr(llm_client.asyncio, "sleep", _no_sleep)
    _key(monkeypatch)
    with pytest.raises(RuntimeError, match="3 trials in a row"):
        for reward in (1.0, 0.0, 1.0, 0.0, 0.0, 1.0):
            _run(_trial_with_reward(reward))


@pytest.fixture
def labels(monkeypatch):
    def set_critical(critical):
        async def label(traj):
            return critical

        monkeypatch.setattr(atlas_module.labeler, "label", label)

    monkeypatch.setenv("PRL_TRANSFORM", "atlas")
    return set_critical


def test_atlas_hard_mask(labels):
    labels({1})  # step 2 (BB) only: the leaf with C has no trained token left
    kept = _run(_trial())
    assert [s.loss_mask for s in kept] == [[0, 0, 0, 1, 1, 0, 0]]
    record = kept[0].metadata["prl_transform"]
    assert record["chain"][0]["record"] == {"atlas": {"steps": 3, "critical": [2]}}
    assert (record["trained_tokens_before"], record["trained_tokens_after"], record["dropped_samples"]) == (4, 2, 1)


def test_atlas_soft_and_unlabeled(labels, monkeypatch):
    labels({0, 2})
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"other_weight": 0.3}')
    kept = _run(_trial())
    assert [s.loss_mask for s in kept] == [s.loss_mask for s in _trial()]
    assert [s.train_metadata["step_weights"] for s in kept] == [[1.0], [1.0, 0.3]]
    labels(None)  # unlabeled: skipped, every step trains
    kept = _run(_trial())
    assert [s.loss_mask for s in kept] == [s.loss_mask for s in _trial()] and kept[0].train_metadata == {}
    assert kept[0].metadata["prl_transform"]["chain"][0]["status"] == "skipped"



@pytest.fixture
def grades(monkeypatch):
    """Signed ATLAS with the judge's grades (good, bad: 0-based step sets, or None); returns the judge calls."""
    calls = []

    def set_grades(answer, mode):
        async def grade(traj):
            calls.append(traj.reward)
            return answer

        monkeypatch.setattr(atlas_signed_module.labeler, "grade", grade)
        monkeypatch.setenv("PRL_TRANSFORM_ARGS", json.dumps({"mode": mode}))
        return calls

    monkeypatch.setenv("PRL_TRANSFORM", "atlas_signed")
    return set_grades


def _weights(kept):
    return [(s.loss_mask, (s.train_metadata or {}).get("step_weights")) for s in kept]


def test_atlas_signed_mild(grades):
    # steps (commit order): A, BB in the TASK leaf (listed second), C in the other leaf
    calls = grades(({0}, {1}), "mild")
    kept = _run(_trial_with_reward(1.0))
    assert _weights(kept) == [([1, 0, 0], [1.0]), ([1, 0, 0, 1, 1, 0, 0], [1.0, -1.0])]  # the bad step BB pushed down
    record = kept[0].metadata["prl_transform"]["chain"][0]["record"]["atlas_signed"]
    assert record == {"mode": "mild", "steps": 3, "passed": True, "graded": True, "good": [1], "bad": [2]}
    kept = _run(_trial_with_reward(0.0))  # a failed trial trains as in GRPO, without asking the judge
    assert _weights(kept) == [(s.loss_mask, None) for s in _trial()] and calls == [1.0]


def test_atlas_signed_mild_extension(grades, monkeypatch):
    # bad_weight scales the push on a passed trial's bad step; failed_good_weight grades failed trials too and keeps
    # their good step (A) from being pushed down; bad (BB) and neutral (C) steps keep weight 1
    calls = grades(({0}, {1}), "mild")
    monkeypatch.setenv("PRL_TRANSFORM_ARGS",
                       json.dumps({"mode": "mild", "bad_weight": 0.5, "failed_good_weight": 0.0}))
    kept = _run(_trial_with_reward(1.0))
    assert _weights(kept) == [([1, 0, 0], [1.0]), ([1, 0, 0, 1, 1, 0, 0], [1.0, -0.5])]
    kept = _run(_trial_with_reward(0.0))
    assert _weights(kept) == [([1, 0, 0], [1.0]), ([1, 0, 0, 1, 1, 0, 0], [0.0, 1.0])]
    assert calls == [1.0, 0.0]  # the failed trial was graded
    record = kept[0].metadata["prl_transform"]["chain"][0]["record"]["atlas_signed"]
    assert record == {"mode": "mild", "steps": 3, "passed": False, "graded": True, "good": [1], "bad": [2]}


def _cmd_trial(reward=1.0):
    """Two steps with distinct text; the second holds a JSON-escaped newline (backslash, n) as commands do."""
    samples = [_sample("TASK", [("run ls -la", 1), ("o1", 0), ('cat x\\n; ok', 1), ("o2", 0)], reward=reward)]
    samples[0].metadata["reward"] = reward
    return samples


def test_weight_text_sets_a_token_range(use):
    use(lambda traj: traj.steps[0].weight_text("ls -l", -1.0))
    kept = _run(_cmd_trial())
    # one token per character: "ls -l" is characters 4-8 of step 0, which starts the response
    assert kept[0].train_metadata == {"token_weight_spans": [[4, 9, -1.0]]}
    assert kept[0].metadata["prl_transform"]["changed"]


class PieceTokenizer:
    """Multi-character tokens: a token only partly inside the text counts as inside."""

    PIECES = {1: "ru", 2: "n ", 3: "ls", 4: " -la", 5: "o1", 9: "P"}  # noqa: RUF012

    def decode(self, ids):
        return "".join(self.PIECES[i] for i in ids)


def test_weight_text_rounds_out_to_whole_tokens(use):
    use(lambda traj: traj.steps[0].weight_text("s -l", 0.5))
    sample = SimpleNamespace(tokens=[9, 1, 2, 3, 4, 5], response_length=5, loss_mask=[1, 1, 1, 1, 0], reward=1.0,
                             rollout_id=7, rollout_log_probs=[0.0] * 5, train_metadata=None,
                             metadata={"leaf": {"node_id": 0}, "reward": 1.0, "trial_dir": None})
    kept = asyncio.run(core.run([sample], PieceTokenizer()))
    assert kept[0].train_metadata == {"token_weight_spans": [[2, 4, 0.5]]}  # tokens "ls" and " -la"


@pytest.mark.parametrize("bad", ["twice", "rewritten", "masked"])
def test_weight_text_misuse_leaves_the_trial_unchanged(use, bad):
    def fn(traj):
        step = traj.steps[0]
        if bad == "twice":
            step.weight_text("l", -1.0)  # "l" occurs twice in "run ls -la"
        elif bad == "rewritten":
            step.action = step.action.replace("ls", "pwd")
            step.weight_text("pwd", -1.0)
        else:
            step.weight_text("ls", -1.0)
            step.mask()
    use(fn)
    kept = _run(_cmd_trial())
    assert not kept[0].train_metadata
    assert kept[0].metadata["prl_transform"]["chain"][0]["status"] == "error"


def test_a_token_span_must_cover_trained_tokens(use):
    def fn(traj):  # advanced use: a span set directly on an untrained token
        traj.samples[0].train_metadata = {"token_weight_spans": [[10, 12, -1.0]]}
    use(fn)
    kept = _run(_cmd_trial())
    assert kept[0].metadata["prl_transform"]["chain"][0]["status"] == "error"


def test_atlas_signed_quote_bad(monkeypatch):
    quotes = {}

    async def grade_quoted(traj):
        return set(), {0, 1}, quotes

    monkeypatch.setattr(atlas_signed_module.labeler, "grade_quoted", grade_quoted)
    monkeypatch.setenv("PRL_TRANSFORM", "atlas_signed")
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", json.dumps({"mode": "mild", "quote_bad": True}))
    # step 0: found verbatim; step 1: the judge wrote the command's \n as a newline: found JSON-escaped
    quotes.update({0: "ls -la", 1: "cat x\n"})
    kept = _run(_cmd_trial())
    assert kept[0].train_metadata == {"token_weight_spans": [[4, 10, -1.0], [12, 19, -1.0]]}
    record = kept[0].metadata["prl_transform"]["chain"][0]["record"]["atlas_signed"]
    assert (record["quoted"], record["quote_fallback"]) == (2, 0)
    quotes.clear()
    quotes.update({0: "not in the step"})  # step 0: not found; step 1: no quote: both fall back to the whole step
    kept = _run(_cmd_trial())
    assert kept[0].train_metadata == {"step_weights": [-1.0, -1.0]}
    record = kept[0].metadata["prl_transform"]["chain"][0]["record"]["atlas_signed"]
    assert (record["quoted"], record["quote_fallback"]) == (0, 2)
    quotes.clear()
    quotes.update({0: "l"})  # occurs twice in "run ls -la": both occurrences
    kept = _run(_cmd_trial())
    assert kept[0].train_metadata["token_weight_spans"] == [[4, 5, -1.0], [8, 9, -1.0]]
    kept = _run(_cmd_trial(reward=0.0))  # a failed trial: not graded (failed_good_weight unset), plain GRPO
    assert not kept[0].train_metadata


@pytest.mark.parametrize("args", [{"bad_weight": 0}, {"mode": "strict", "failed_good_weight": 0.0},
                                  {"failed_good_weight": 1.5}, {"mode": "strict", "quote_bad": True}])
def test_atlas_signed_rejects_bad_params(args):
    import asyncio
    traj = SimpleNamespace(reward=1.0, steps=[], record={})
    with pytest.raises(ValueError):
        asyncio.run(atlas_signed_module.atlas_signed(traj, **args))


def test_atlas_signed_strict(grades):
    grades(({0}, {1}), "strict")
    kept = _run(_trial_with_reward(1.0))  # passed: good 1, bad -1, neutral C leaves the loss (its leaf is dropped)
    assert _weights(kept) == [([1, 0, 0, 1, 1, 0, 0], [1.0, -1.0])]
    kept = _run(_trial_with_reward(0.0))  # failed: only the bad step trains, pushed down by the negative advantage
    assert _weights(kept) == [([0, 0, 0, 1, 1, 0, 0], None)]
    grades((set(), set()), "strict")  # nothing graded: no push, but the trial stays (weight 0, not a mask-all error)
    kept = _run(_trial_with_reward(0.0))
    assert _weights(kept) == [([1, 0, 0], [0.0]), ([1, 0, 0, 1, 1, 0, 0], [0.0, 0.0])]
    assert kept[0].metadata["prl_transform"]["chain"][0]["status"] == "ok"
    grades(None, "strict")  # the judge failed: skipped, plain GRPO
    kept = _run(_trial_with_reward(1.0))
    assert _weights(kept) == [(s.loss_mask, None) for s in _trial()]
    assert kept[0].metadata["prl_transform"]["chain"][0]["status"] == "skipped"

# --- rewriting: {traj, reward} -> {traj, reward} ---------------------------------------------------------------------

class TextTokenizer(CharTokenizer):
    """One token per character; a dict weight-version per call as Miles stores them."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def _turn(thinking, command):
    return f"{thinking}</think>\n\n{json.dumps(command, indent=2)}<|im_end|>"


def _message(text, last=False):
    return f"\n<|im_start|>user\n{text}<|im_end|>" + ("" if last else "\n<|im_start|>assistant\n<think>\n")


def _real_trial(trial_dir=None, reward=1.0):
    """Three turns with Qwen/Terminus-2 structure; rollout log probs 0.5 everywhere; one weight-version span per turn."""
    parts = [(_turn("look around", {"commands": [{"keystrokes": "ls\n"}]}), 1), (_message("a b"), 0),
             (_turn("list again", {"commands": [{"keystrokes": "ls -la\n"}]}), 1), (_message("a b"), 0),
             (_turn("done", {"commands": [], "task_complete": True}), 1)]
    s = _sample("TASK<think>\n", parts, reward=reward, trial_dir=trial_dir)
    s.rollout_log_probs = [0.5] * s.response_length
    prompt, spans, pos = len(s.tokens) - s.response_length, [], 0
    for text, trained in parts:
        if trained:
            spans.append([{"version": "1", "abs_start": prompt + pos, "abs_end": prompt + pos + len(text)}])
        pos += len(text)
    s.weight_versions = spans
    return [s]


def _text(samples):
    return CharTokenizer().decode(samples[0].tokens)


def _run_text(samples, args=None):
    return asyncio.run(core.run(samples, TextTokenizer(), args))


def test_parts_of_a_turn():
    traj = Trajectory.of(_real_trial(), TextTokenizer())
    first = traj.steps[0]
    assert first.thinking == "look around" and first.command == {"commands": [{"keystrokes": "ls\n"}]}
    assert first.terminal == "a b" and traj.steps[-1].terminal is None


def test_delete_a_turn(use):
    use(lambda traj: traj.steps[1].delete())
    kept = _run_text(_real_trial())
    s = kept[0]
    assert "list again" not in _text(kept) and _text(kept).count("a b") == 1
    assert (len(s.loss_mask), len(s.rollout_log_probs)) == (s.response_length, s.response_length)
    assert set(s.rollout_log_probs) == {0.5}  # nothing re-encoded: the kept tokens keep their log probs
    runs = [(a, b) for t, a, b in core._runs(s.loss_mask) if t]
    prompt = len(s.tokens) - s.response_length
    assert [(c[0]["abs_start"], c[0]["abs_end"]) for c in s.weight_versions] == [(prompt + a, prompt + b) for a, b in runs]
    record = s.metadata["prl_transform"]
    assert record["tokens_changed"] and record["chain"][0]["deleted_steps"] == [1]


def test_minimal_edits_keep_the_policys_tokens(use):
    def fn(traj):
        traj.steps[0].edit_thinking("around", "at the files")
        traj.steps[1].set_command({"commands": [{"keystrokes": "ls -l\n"}]})
        traj.steps[1].set_terminal("a b c")

    use(fn)
    s = _run_text(_real_trial())[0]
    text = _text([s])
    assert "look at the files</think>" in text and '"ls -l\\n"' in text and "user\na b c<|im_end|>" in text
    lps = s.rollout_log_probs
    assert lps[0] == 0.5 and 0.0 in lps and lps[-1] == 0.5  # untouched head and tail keep theirs, new text has 0
    traj = Trajectory.of([s], TextTokenizer())
    assert traj.steps[1].command == {"commands": [{"keystrokes": "ls -l\n"}]} and traj.steps[1].terminal == "a b c"


def test_reward_can_be_rewritten(use):
    def fn(traj):
        traj.steps[1].delete()
        traj.reward = 0.0

    use(fn)
    kept = _run_text(_real_trial())
    assert {(s.reward, s.metadata["reward"]) for s in kept} == {(0.0, 0.0)}
    assert kept[0].metadata["prl_transform"]["chain"][0]["reward"] == [1.0, 0.0]


@pytest.mark.parametrize("case", ["delete_last", "not_once", "several_leaves", "bad_reward", "teacher_log_probs"])
def test_rewrites_that_cannot_apply_keep_the_trial(use, case):
    def fn(traj):
        if case == "delete_last":
            traj.steps[-1].delete()
        if case == "not_once":
            traj.steps[0].edit_thinking("o", "0")  # "look around": several o's
        if case in ("several_leaves", "bad_reward", "teacher_log_probs"):
            traj.steps[0].edit_thinking("look", "peek")
        if case == "bad_reward":
            traj.reward = float("nan")

    use(fn)
    samples = _real_trial() if case != "several_leaves" else _real_trial() + _trial()
    if case == "teacher_log_probs":  # a per-token field the renderer does not rebuild
        samples[0].teacher_log_probs = [0.0] * samples[0].response_length
    before = [list(s.tokens) for s in samples]
    kept = _run_text(samples)
    assert [list(s.tokens) for s in kept] == before
    assert kept[0].metadata["prl_transform"]["chain"][0]["status"] == "error"


def test_rewrite_then_mask_in_one_call_and_in_a_chain(use, monkeypatch):
    def rewrite_and_mask(traj):
        traj.steps[0].edit_thinking("look around", "inspect the directory first")
        traj.steps[1].delete()
        traj.steps[0].mask()
        traj.steps[2].weight = 0.5

    use(rewrite_and_mask)
    s = _run_text(_real_trial())[0]
    trained = [t for t, _, _ in core._runs(s.loss_mask) if t]
    assert len(trained) == 1 and s.train_metadata["step_weights"] == [0.5]  # the masked and the deleted turns are gone
    seen = []
    use(lambda traj: traj.steps[1].delete(), name="a")
    use(lambda traj: seen.append([st.thinking for st in traj.steps]), name="b")
    monkeypatch.setenv("PRL_TRANSFORM", "a,b")
    _run_text(_real_trial())
    assert seen == [["look around", "done"]]  # the second transform sees the rewritten trial


def test_splice_keeps_a_non_canonical_tokenization_outside_the_edit():
    vocab = ["ab", "a", "b", "c", "x"]

    class Toy:
        def encode(self, text, add_special_tokens=False):
            out, i = [], 0
            while i < len(text):
                piece = next(v for v in vocab if text.startswith(v, i))
                out.append(vocab.index(piece))
                i += len(piece)
            return out

        def decode(self, ids):
            return "".join(vocab[i] for i in ids)

    ids = [1, 2, 3, 1, 2]  # "a","b","c","a","b": the canonical encoding would be "ab","c","ab"
    out, origin = core._splice(Toy(), ids, "abcab", "abxab")
    assert Toy().decode(out) == "abxab" and out[:2] == [1, 2] and out[-2:] == [1, 2] and origin == [0, 1, -1, 3, 4]


def test_splice_keeps_the_text_between_a_reasoning_edit_and_a_new_action():
    class Special(TextTokenizer):  # "</think>" is one token, as in Qwen's vocabulary
        def encode(self, text, add_special_tokens=False):
            head, sep, tail = text.partition("</think>")
            return [ord(c) for c in head] + ([0x110000] if sep else []) + (self.encode(tail) if sep else [])

        def decode(self, ids):
            return "".join("</think>" if i == 0x110000 else chr(i) for i in ids)

    tok = Special()
    old, new = "abc</think>xyz", "Abc</think>xyZ"
    out, origin = core._splice(tok, tok.encode(old), old, new)
    assert tok.decode(out) == new and origin == [-1, 1, 2, 3, 4, 5, -1]



def test_splice_keeps_the_policys_text_between_two_edits():
    # rewrite+mask: only what the transform wrote is new; the policy's text between two edits keeps its tokens, so it
    # stays in the loss (one splice from the first edit to the last masked 10 to 15% of a rewritten trial's reasoning)
    tok = TextTokenizer()
    middle = " Then I checked the listing carefully and every file was where the task said it would be."
    old = "The file has 3 lines." + middle + " So the answer is 3."
    new = "The file has 4 lines." + middle + " So the answer is 4."
    out, origin = core._splice(tok, tok.encode(old), old, new)
    assert tok.decode(out) == new and origin.count(-1) == 2  # the two digits the edits changed
    start = len("The file has 4 lines.")
    assert origin[start:start + len(middle)] == list(range(start, start + len(middle)))


# --- the framework's model call, time limit and records -------------------------------------------------------------

async def _no_sleep(seconds):
    return None


def _key(monkeypatch, attempts=2):
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".key", delete=False) as key:
        key.write("k")
    for name, value in {"API_BASE": "http://t/v1", "KEY_FILE": key.name, "MODEL": "m",
                        "ATTEMPTS": str(attempts)}.items():
        monkeypatch.setenv(f"PRL_JUDGE_{name}", value)


def _trial_with_reward(reward, trial_dir=None):
    samples = _trial(trial_dir)
    for s in samples:
        s.metadata["reward"] = reward
        s.reward = reward
    return samples


def _answers(monkeypatch, *contents):
    sent = []

    async def post(cfg, messages):
        sent.append(messages)
        content = contents[min(len(sent), len(contents)) - 1]
        return {"choices": [{"message": {"content": content}}], "model": "m",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2}}

    monkeypatch.setattr(llm_client, "_post", post)
    monkeypatch.setattr(llm_client.asyncio, "sleep", _no_sleep)
    _key(monkeypatch)
    return sent


def test_ask_checks_answers_and_saves_every_call(use, monkeypatch, tmp_path):
    sent = _answers(monkeypatch, "no number", "2")

    async def fn(traj):
        n = await core.ask("which step?", check=int)  # int("no number") raises: asked again
        traj.steps[n].mask()
        return traj

    use(fn)
    kept = _run(_trial(str(tmp_path)))
    entry = kept[0].metadata["prl_transform"]["chain"][0]
    assert len(sent) == 2 and entry["status"] == "ok"
    # the second request carries the rejected answer and why it was rejected
    assert sent[0] == [{"role": "user", "content": "which step?"}] and sent[1][1]["content"] == "no number"
    assert "invalid literal" in sent[1][2]["content"]
    assert entry["calls"] == {"n": 1, "prompt_tokens": 20, "completion_tokens": 4}
    calls = json.loads((tmp_path / "transform_calls.json").read_text())
    assert calls[0]["transform"] == "t" and calls[0]["answer"] == "2" and "no number" in calls[0]["errors"][0]


def test_no_acceptable_answer_skips_the_trial(use, monkeypatch):
    _answers(monkeypatch, "nonsense")

    async def fn(traj):
        traj.steps[0].mask()  # applied before the call fails: still nothing changes
        await core.ask("x", check=int)
        return traj

    use(fn)
    samples = _trial()
    before = [list(s.loss_mask) for s in samples]
    kept = _run(samples)
    entry = kept[0].metadata["prl_transform"]["chain"][0]
    assert entry["status"] == "skipped" and "no acceptable answer" in entry["reason"]
    assert sorted(list(s.loss_mask) for s in kept) == sorted(before)
    assert asyncio.run(core.run(_trial(), CharTokenizer(), strict=True))  # the pre-run check accepts a skip


def test_a_check_may_return_none(use, monkeypatch):
    _answers(monkeypatch, "fine")

    async def fn(traj):
        assert await core.ask("x", check=lambda content: None) is None
        traj.steps[0].mask()
        return traj

    use(fn)
    assert _run(_trial())[0].metadata["prl_transform"]["chain"][0]["status"] == "ok"


@pytest.mark.parametrize("kind", ["async", "sync"])
def test_a_transform_out_of_time_leaves_the_trial(use, monkeypatch, kind):
    import time as clock

    monkeypatch.setenv("PRL_TRANSFORM_TIMEOUT", "0.2")

    async def slow_async(traj):
        await asyncio.sleep(5)

    def slow_sync(traj):
        clock.sleep(1)

    use(slow_async if kind == "async" else slow_sync)
    samples = _trial()
    before = [list(s.loss_mask) for s in samples]
    t0 = clock.time()
    kept = _run(samples)
    entry = kept[0].metadata["prl_transform"]["chain"][0]
    assert entry["status"] == "error" and "PRL_TRANSFORM_TIMEOUT" in entry["reason"]
    assert sorted(list(s.loss_mask) for s in kept) == sorted(before) and clock.time() - t0 < 3
    assert core._failures == 1


def test_new_tokens_are_counted(use):
    use(lambda traj: traj.steps[0].edit_thinking("look around", "look"))
    record = _run_text(_real_trial())[0].metadata["prl_transform"]
    assert record["tokens_changed"] and record["changed"] and record["new_tokens"] == 0  # a deletion writes nothing
    use(lambda traj: traj.steps[0].edit_thinking("around", "at the files"))
    record = _run_text(_real_trial())[0].metadata["prl_transform"]
    assert 0 < record["new_tokens"] <= len("at the files") + 2


def test_miles_weight_version_objects_are_frozen(use):
    """In training the spans are Miles' frozen WeightVersionSpan inside WeightVersionsPerCall (found in a smoke)."""
    import dataclasses as dc

    @dc.dataclass(frozen=True)
    class Span:
        version: str
        abs_start: int
        abs_end: int

    @dc.dataclass
    class Call:
        spans: list

    samples = _real_trial()
    samples[0].weight_versions = [Call([Span(**d) for d in call]) for call in samples[0].weight_versions]
    use(lambda traj: traj.steps[1].delete() or traj.steps[0].edit_thinking("look", "peek"))
    s = _run_text(samples)[0]
    assert s.metadata["prl_transform"]["chain"][0]["status"] == "ok"
    prompt = len(s.tokens) - s.response_length
    runs = [(prompt + a, prompt + b) for t, a, b in core._runs(s.loss_mask) if t]
    assert [(c.spans[0].abs_start, c.spans[0].abs_end) for c in s.weight_versions] == runs
