"""transforms/rewrite.py: which trials it touches, the edit plan's rules, and what reaches the samples."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from prl.algo import llm_client
from prl.transforms import core


class TextTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _turn(thinking, keystrokes, done=False):
    command = {"analysis": "a", "plan": "p", "commands": [] if done else [{"keystrokes": keystrokes, "duration": 0.1}],
               "task_complete": done}
    return f"{thinking}</think>\n\n{json.dumps(command, indent=2)}<|im_end|>"


def _message(text):
    return f"\n<|im_start|>user\nNew Terminal Output:\n\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def _trial(reward=1.0, trial_dir=None):
    """Five turns: list, list again (a repeat), write, mark complete, confirm."""
    parts = [_turn("I will list the files first.", "ls\n"), _message("a.txt"),
             _turn("Let me list again to be sure.", "ls\n"), _message("a.txt"),
             _turn("Now write the answer.", "echo 1 > /out\n"), _message(""),
             _turn("Done.", "", done=True), _message("Are you sure?"), _turn("Yes.", "", done=True)]
    response = [ord(c) for c in "".join(parts)]
    mask = [int(i % 2 == 0) for i, p in enumerate(parts) for _ in p]
    return [SimpleNamespace(tokens=[ord(c) for c in "TASK<think>\n"] + response, response_length=len(response),
                            loss_mask=mask, reward=reward, rollout_id=3, rollout_log_probs=[0.0] * len(response),
                            train_metadata=None, metadata={"leaf": {"node_id": 1}, "reward": reward,
                                                           "trial_dir": trial_dir})]


@pytest.fixture
def teacher(monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_text("k")
    for name, value in {"API_BASE": "http://t/v1", "KEY_FILE": str(key), "MODEL": "m", "ATTEMPTS": "2"}.items():
        monkeypatch.setenv(f"PRL_JUDGE_{name}", value)
    monkeypatch.setenv("PRL_TRANSFORM", "rewrite")
    monkeypatch.delenv("PRL_TRANSFORM_ARGS", raising=False)
    monkeypatch.setattr(core, "_failures", 0)
    sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: sleep(0))
    calls = []

    def answers(*plans):
        async def post(cfg, messages):
            calls.append(messages[-1]["content"])
            plan = plans[min(len(calls), len(plans)) - 1]
            return {"choices": [{"message": {"content": json.dumps(plan)}}], "model": "m"}

        monkeypatch.setattr(llm_client, "_post", post)
        return calls

    return answers


def _run(samples):
    return asyncio.run(core.run(samples, TextTokenizer()))


def _text(samples):
    return "".join(chr(i) for i in samples[0].tokens)


GOOD = {"edits": [{"step": 2, "op": "delete", "why": "repeats step 1"},
                  {"step": 3, "op": "edit_thinking", "old": "Now write", "new": "The file is a.txt; write",
                   "why": "state the observation"}],
        "passes": True, "reason": "same commands minus a repeat"}


def test_failed_trials_are_left_alone(teacher):
    calls = teacher(GOOD)
    kept = _run(_trial(reward=0.0))
    assert calls == [] and kept[0].metadata["prl_transform"]["chain"][0]["status"] == "ok"
    assert kept[0].metadata["prl_transform"]["chain"][0]["record"]["rewrite"]["edited"] is False


def test_a_valid_plan_is_applied(teacher, tmp_path):
    calls = teacher(GOOD)
    kept = _run(_trial(trial_dir=str(tmp_path)))
    text = _text(kept)
    assert "list again" not in text and "The file is a.txt; write the answer." in text and text.count("ls\\n") == 1
    entry = kept[0].metadata["prl_transform"]["chain"][0]
    assert entry["status"] == "ok" and entry["deleted_steps"] == [1] and entry["record"]["rewrite"]["edited"]
    assert kept[0].reward == 1.0 and len(calls) == 1 and "Step 5" in calls[0]
    calls = json.loads((tmp_path / "transform_calls.json").read_text())  # saved by the framework
    assert json.loads(calls[0]["answer"]) == GOOD and entry["record"]["rewrite"]["passes"] is True


@pytest.mark.parametrize("bad", [
    {"edits": [{"step": 4, "op": "delete", "why": "x"}], "passes": True},                      # a protected step
    {"edits": [{"step": 1, "op": "edit_thinking", "old": "l", "new": "L"}], "passes": True},     # 'old' not unique
    {"edits": [{"step": 2, "op": "set_command", "command": {"analysis": "", "plan": "", "commands": [],
                                                            "task_complete": True}}], "passes": True},
    {"edits": [{"step": 2, "op": "delete"}], "passes": False},                                    # not expected to pass
    {"edits": [{"step": 1, "op": "delete"}, {"step": 2, "op": "delete"}], "passes": True},        # over the budget
    {"edits": [{"step": 1, "op": "set_terminal", "terminal": "x"}], "passes": True},             # terminal first
    "not json",
])
def test_invalid_plans_are_asked_again_then_skipped(teacher, bad):
    calls = teacher(bad)
    kept = _run(_trial())
    entry = kept[0].metadata["prl_transform"]["chain"][0]
    assert entry["status"] == "skipped" and len(calls) == 2 and _text(kept) == _text(_trial())


def test_retry_then_valid_and_an_empty_plan(teacher):
    calls = teacher({"edits": [{"step": 4, "op": "delete"}], "passes": True}, GOOD)
    assert "list again" not in _text(_run(_trial())) and len(calls) == 2
    teacher({"edits": [], "passes": True, "reason": "already clean"})
    kept = _run(_trial())
    assert _text(kept) == _text(_trial()) and kept[0].metadata["prl_transform"]["chain"][0]["status"] == "ok"


def test_the_prompt_version_is_a_parameter(teacher, monkeypatch):
    calls = teacher({"edits": [], "passes": True})
    _run(_trial())
    assert "1. Repeated work." in calls[-1] and "Steps 4, 5 complete the task" not in calls[-1]  # v2 by default
    assert "Steps 4, 5: they complete the task." in calls[-1] and "{budget}" not in calls[-1]
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"prompt": "v1"}')
    _run(_trial())
    assert "## What to improve" in calls[-1] and "Steps 4, 5 finish the task" in calls[-1]
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"prompt": "v9"}')
    kept = _run(_trial())
    entry = kept[0].metadata["prl_transform"]["chain"][0]
    assert entry["status"] == "error" and "no rewrite prompt 'v9'" in entry["reason"]


def test_a_finishing_step_may_fix_its_analysis_but_not_its_commands(teacher):
    done = {"analysis": "All done; the file holds 1.", "plan": "p", "commands": [], "task_complete": True}
    calls = teacher({"edits": [{"step": 4, "op": "set_command", "command": {**done, "commands": [{"keystrokes": "ls\n"}]}}],
                     "passes": True},
                    {"edits": [{"step": 4, "op": "set_command", "command": done}], "passes": True})
    kept = _run(_trial())
    assert len(calls) == 2 and "All done; the file holds 1." in _text(kept)  # new keystrokes refused, then text only


def _task_with_tests(tmp_path):
    task = tmp_path / "task_1"
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "expected_output.txt").write_text("EXPECTED-42\n")
    (task / "tests" / "test.sh").write_text("cmp expected_output.txt /output/out\n")
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "config.json").write_text(json.dumps({"task": {"path": str(task)}}))
    return str(trial)


FIX = {"edits": [{"step": 3, "op": "set_command", "command": {"analysis": "a", "plan": "p", "commands": [
           {"keystrokes": "echo 2 > /out\n", "duration": 0.1}], "task_complete": False}, "why": "wrong value"}],
       "passes": True, "reason": "now writes 2"}


def test_failed_trials_are_fixed_with_the_tests_in_view(teacher, monkeypatch, tmp_path):
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"failures": "fix"}')
    calls = teacher(FIX)
    kept = _run(_trial(reward=0.0, trial_dir=_task_with_tests(tmp_path)))
    entry = kept[0].metadata["prl_transform"]["chain"][0]
    assert "## The task's tests (the agent never saw them)" in calls[0] and "EXPECTED-42" in calls[0]
    assert "1. Find where the agent went wrong" in calls[0]  # the fix criteria, not the cleanup checklist
    assert kept[0].reward == 1.0 and entry["reward"] == [0.0, 1.0] and "echo 2 > /out" in _text(kept)
    assert entry["record"]["rewrite"]["mode"] == "fix" and entry["record"]["rewrite"]["fixed"] is True


@pytest.mark.parametrize("case", ["no_tests", "cannot_fix", "keep"])
def test_failed_trials_that_stay_failed(teacher, monkeypatch, tmp_path, case):
    if case != "keep":
        monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"failures": "fix"}')
    calls = teacher({"edits": [], "passes": False, "reason": "needs another turn"})
    kept = _run(_trial(reward=0.0, trial_dir=None if case == "no_tests" else _task_with_tests(tmp_path)))
    record = kept[0].metadata["prl_transform"]["chain"][0]["record"]["rewrite"]
    assert kept[0].reward == 0.0 and _text(kept) == _text(_trial())
    assert len(calls) == (1 if case == "cannot_fix" else 0)
    assert (record.get("why") or "") == {"no_tests": "the task's tests were not found", "cannot_fix": "",
                                         "keep": "failed trial (left to mask+rewrite)"}[case]


def test_fix_only_keeps_successes_and_caps_the_repair(teacher, monkeypatch, tmp_path):
    args = {"successes": "keep", "failures": "fix", "fix_prompt": "fix_v2", "fix_max_change_share": 0.25}
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", json.dumps(args))
    calls, trial_dir = teacher(FIX), _task_with_tests(tmp_path)
    kept = _run(_trial())  # a success: not sent to the teacher, trained as the policy wrote it
    record = kept[0].metadata["prl_transform"]["chain"][0]["record"]["rewrite"]
    assert calls == [] and _text(kept) == _text(_trial()) and record["why"] == "successful trial kept as the policy wrote it"
    kept = _run(_trial(reward=0.0, trial_dir=trial_dir))
    record = kept[0].metadata["prl_transform"]["chain"][0]["record"]["rewrite"]
    assert len(calls) == 1 and "change at\n   most 25% of the agent's text" in calls[0] and "Do not clean up" in calls[0]
    assert kept[0].reward == 1.0 and record["fixed"] is True and record["max_change_share"] == 0.25
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", json.dumps({**args, "fix_max_change_share": 0.001}))
    kept = _run(_trial(reward=0.0, trial_dir=trial_dir))  # the same repair, over a tiny cap
    assert len(calls) == 3 and "more than 0%" in calls[2] and kept[0].reward == 0.0 and _text(kept) == _text(_trial())


def test_v5_brings_its_own_opening_and_the_change_cap(teacher, monkeypatch):
    calls = teacher({"edits": [], "passes": True})
    _run(_trial())
    assert calls[-1].startswith("You review a trajectory") and "Turn it into a cleaner example" in calls[-1]  # v2
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"prompt": "v5", "max_change_share": 0.25}')
    kept = _run(_trial())
    assert calls[-1].startswith("You review a trajectory") and "Correct the agent's mistakes" in calls[-1]
    assert "cleaner example" not in calls[-1] and "## What to correct" in calls[-1] and "## Task" in calls[-1]
    assert "at most 25% of the\n    agent's text" in calls[-1] and "{" + "change_share}" not in calls[-1]
    assert kept[0].metadata["prl_transform"]["chain"][0]["record"]["rewrite"]["max_change_share"] == 0.25


def test_apply_edits_off_masks_the_removed_steps_in_the_original_text(teacher, monkeypatch):
    teacher(GOOD)
    original = _text(_trial())
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"apply_edits": false}')
    kept = _run(_trial())
    sample = kept[0]
    assert _text(kept) == original  # no token changes; the deleted step 2 leaves the loss, the reasoning edit stays
    response = sample.tokens[len(sample.tokens) - sample.response_length:]
    trained = "".join(chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m)
    assert "list again" not in trained and "I will list" in trained and "Now write" in trained and trained.endswith("<|im_end|>")
    record = sample.metadata["prl_transform"]["chain"][0]["record"]["rewrite"]
    assert record["applied"] is False and record["edited"] is True


def _trained(sample):
    response = sample.tokens[len(sample.tokens) - sample.response_length:]
    return "".join(chr(t) for t, m in zip(response, sample.loss_mask, strict=True) if m)


def test_a_correction_trains_only_the_teachers_tokens_after_the_visited_state(teacher, monkeypatch):
    teacher(GOOD)
    original = _text(_trial())
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"added": "correct", "added_reward": 1.5}')
    kept = _run(_trial())
    assert len(kept) == 2 and _text(kept) == original and kept[0].reward == kept[1].reward == 1.0
    assert not (kept[0].train_metadata or {}).get("group_reward")
    extra = kept[1]
    assert extra.train_metadata["group_reward"] == 1.5 and extra.rollout_id == kept[0].rollout_id
    text = "".join(chr(i) for i in extra.tokens)
    assert "Let me list again" in text and text.endswith("The file is a.txt;")  # the visited state, then the new text
    assert _trained(extra) == "The file is a.txt;"
    assert len(extra.loss_mask) == extra.response_length == len(extra.rollout_log_probs)
    record = kept[0].metadata["prl_transform"]["chain"][0]["record"]["rewrite"]
    assert record["corrected_step"] == 3 and record["applied"] is False


def test_a_twin_is_the_whole_rewritten_trial_besides_the_original(teacher, monkeypatch):
    teacher(GOOD)
    original = _text(_trial())
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"added": "twin"}')
    kept = _run(_trial())
    assert len(kept) == 2 and _text(kept) == original
    twin = "".join(chr(i) for i in kept[1].tokens)
    assert "list again" not in twin and "The file is a.txt; write the answer." in twin
    assert kept[1].train_metadata["group_reward"] == 1.0 and _trained(kept[1]).endswith("<|im_end|>")


def test_passing_and_failed_trials_both_get_a_correction_and_keep_their_reward(teacher, monkeypatch, tmp_path):
    teacher(GOOD)
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"failures": "fix", "added": "correct"}')
    passed = _run(_trial())
    failed = _run(_trial(reward=0.0, trial_dir=_task_with_tests(tmp_path)))
    for kept, reward in ((passed, 1.0), (failed, 0.0)):
        assert len(kept) == 2 and kept[0].reward == kept[1].reward == reward
        assert kept[1].train_metadata["group_reward"] == 1.0
    assert failed[0].metadata["reward"] == 0.0 and failed[0].metadata["prl_transform"]["chain"][0].get("reward") is None


def test_a_failed_trial_gets_a_whole_rewritten_twin_with_reward_1_and_keeps_its_own(teacher, monkeypatch, tmp_path):
    teacher(GOOD)
    original = _text(_trial(reward=0.0))
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"failures": "fix", "added": "twin"}')
    kept = _run(_trial(reward=0.0, trial_dir=_task_with_tests(tmp_path)))
    assert len(kept) == 2 and _text(kept) == original and kept[0].reward == kept[1].reward == 0.0
    assert kept[1].train_metadata["group_reward"] == 1.0 and "list again" not in "".join(chr(i) for i in kept[1].tokens)


@pytest.mark.parametrize("verdict", [1.0, 0.0])
def test_an_executed_twin_takes_the_verifiers_reward_and_real_outputs(teacher, monkeypatch, verdict):
    from prl.rollout import replay as replay_module

    plan = {"edits": [*GOOD["edits"], {"step": 3, "op": "set_terminal", "terminal": "New Terminal Output:\n\nTEACHER", "why": "x"}],
            "passes": False, "reason": "unsure"}
    teacher(plan)
    seen = []

    async def fake(trial_dir, turns, name="replay"):
        seen.append(turns)
        return verdict, [{"timed_out": False, "output": f"REAL{i}"} for i in range(len(turns))]

    monkeypatch.setattr(replay_module, "replay", fake)
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"added": "twin", "execute": true}')
    kept = _run(_trial())
    assert len(kept) == 2 and kept[1].train_metadata["group_reward"] == verdict
    text = "".join(chr(i) for i in kept[1].tokens)
    assert "TEACHER" not in text and "REAL" in text and "list again" not in text
    assert len(seen[0]) == 4  # the deleted turn does not run
    record = kept[0].metadata["prl_transform"]["chain"][0]["record"]["rewrite"]
    assert record["executed"]["reward"] == verdict and record["passes"] is False


def test_a_failed_execution_adds_nothing(teacher, monkeypatch):
    from prl.rollout import replay as replay_module

    teacher(GOOD)

    async def fake(trial_dir, turns, name="replay"):
        raise replay_module.ReplayFailed("sandbox")

    monkeypatch.setattr(replay_module, "replay", fake)
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"added": "twin", "execute": true}')
    kept = _run(_trial())
    assert len(kept) == 1 and kept[0].metadata["prl_transform"]["chain"][0]["record"]["rewrite"]["executed"]["failed"] == "sandbox"


def test_a_replay_retries_a_run_without_a_reward_and_a_timed_out_run_keeps_its_reward(monkeypatch, tmp_path):
    from prl.rollout import replay as replay_module

    results = [replay_module.ReplayFailed("no verifier reward: RuntimeError x"), (0.0, [{"timed_out": True, "output": "o"}])]

    async def once(work, base, turns):
        out = results.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(replay_module, "_once", once)
    assert asyncio.run(replay_module.replay(str(tmp_path), [[]])) == (0.0, [{"timed_out": True, "output": "o"}])
    assert results == []


def test_a_trial_without_its_config_adds_nothing_and_does_not_raise(teacher, monkeypatch, tmp_path):
    # The pre-run check's recorded trials have no Harbor config
    teacher(GOOD)
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"added": "twin", "execute": true}')
    kept = asyncio.run(core.run(_trial(trial_dir=str(tmp_path)), TextTokenizer(), strict=True))
    assert len(kept) == 1 and "failed" in kept[0].metadata["prl_transform"]["chain"][0]["record"]["rewrite"]["executed"]


def test_clean_off_and_the_teachers_share_leave_trials_alone(teacher, monkeypatch):
    calls = teacher(GOOD)
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"clean": false, "added": "correct"}')
    assert len(_run(_trial())) == 1 and calls == []
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"share": 0.0}')
    assert len(_run(_trial())) == 1 and calls == []


def test_the_change_cap_counts_removed_text_and_deleted_steps():
    from transforms import rewrite as rw

    traj = core.Trajectory.of(_trial(), TextTokenizer())
    total = sum(len(st.action) for st in traj.steps)
    cut = {"edits": [{"step": 1, "op": "edit_thinking", "old": "I will list the files first.", "new": ""}], "passes": True}
    rw.check_plan(traj, cut, 1, 2, 0.3)  # writes nothing: the old budget never sees it
    rw.check_plan(traj, cut, 1, 2, 0.3, max_change_share=28 / total)
    with pytest.raises(ValueError, match="characters of the agent's text changed"):
        rw.check_plan(traj, cut, 1, 2, 0.3, max_change_share=27 / total)
    drop = {"edits": [{"step": 2, "op": "delete"}], "passes": True}
    whole = len(traj.steps[1].action)  # a deleted step counts whole
    rw.check_plan(traj, drop, 1, 2, 0.3, max_change_share=whole / total)
    with pytest.raises(ValueError, match="more than"):
        rw.check_plan(traj, drop, 1, 2, 0.3, max_change_share=(whole - 1) / total)
    assert traj.steps[1].deleted is False  # plans are checked on a copy


def test_ops_restricts_the_teacher_to_some_operations(teacher, monkeypatch):
    calls = teacher({"edits": [{"step": 2, "op": "set_command", "command": {"analysis": "a", "plan": "p", "commands": [
                        {"keystrokes": "ls -a\n", "duration": 0.1}], "task_complete": False}}], "passes": True},
                    {"edits": [{"step": 2, "op": "delete", "why": "repeats step 1"}], "passes": True})
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"ops": "delete,edit_thinking"}')
    kept = _run(_trial())
    assert len(calls) == 2 and "set_command is not allowed here" in calls[1] and "list again" not in _text(kept)
    assert kept[0].metadata["prl_transform"]["chain"][0]["record"]["rewrite"]["ops"] == "delete,edit_thinking"


def test_mask_new_keeps_the_teachers_words_out_of_the_loss(teacher, monkeypatch):
    teacher({"edits": [{"step": 3, "op": "edit_thinking", "old": "Now write", "new": "The file is a.txt; write"}],
             "passes": True})
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"mask_new": true}')
    kept = _run(_trial())
    s = kept[0]
    prompt = len(s.tokens) - s.response_length
    trained = "".join(chr(t) for t, m in zip(s.tokens[prompt:], s.loss_mask) if m)
    context = "".join(chr(t) for t in s.tokens)
    assert "The file is a.txt; write the answer." in context  # in the context...
    assert "The file is a.txt" not in trained and " the answer." in trained  # ...but only the policy's words trained
    entry = s.metadata["prl_transform"]
    assert entry["new_tokens"] == 0 and entry["tokens_changed"] and entry["chain"][0]["record"]["rewrite"]["mask_new"]


def test_the_removed_budget_counts_only_the_agents_own_text():
    from transforms import rewrite as rw

    traj = core.Trajectory.of(_trial(), TextTokenizer())
    total = sum(len(st.action) for st in traj.steps)
    write = {"edits": [{"step": 3, "op": "edit_thinking", "old": "Now write", "new": "x" * 40 + " write"}], "passes": True}
    rw.check_plan(traj, write, 1, 2, 1.0, max_removed_share=3 / total)  # writing is free, "Now" removed
    with pytest.raises(ValueError, match="own text removed or replaced"):
        rw.check_plan(traj, write, 1, 2, 1.0, max_removed_share=2 / total)
    drop = {"edits": [{"step": 2, "op": "delete"}], "passes": True}
    with pytest.raises(ValueError, match="own text removed"):
        rw.check_plan(traj, drop, 1, 2, 1.0, max_removed_share=(len(traj.steps[1].action) - 1) / total)


def _templated(trial):
    """The trial with a chat-template prompt: a reroll replays its first user message."""
    s = trial[0]
    s.tokens = [ord(c) for c in "<|im_start|>user\nTASK<|im_end|>\n<|im_start|>assistant\n<think>\n"] + \
        s.tokens[len("TASK<think>\n"):]
    return trial


REROLL = {"edits": [{"step": 2, "op": "edit_thinking", "old": "Let me list again", "new": "The file is a.txt; next",
                     "why": "the listing was clear"}], "reroll": 3, "reason": "it wrote the wrong answer"}


def test_the_teacher_picks_a_reroll_step_and_corrections_reach_only_the_reroll(teacher, monkeypatch, tmp_path):
    calls = teacher(REROLL)
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"added": "reroll"}')
    original = _text(_templated(_trial(reward=0.0)))
    kept = _run(_templated(_trial(reward=0.0, trial_dir=str(tmp_path))))
    record = kept[0].metadata["prl_transform"]
    assert _text(kept) == original and len(kept) == 1  # the trial trains as sampled; the reroll comes later
    [anchor] = record["reroll"]["anchors"]
    assert anchor["step"] == 2 and anchor["rewritten_steps"] == [1] and (anchor["n"], anchor["credit"]) == (1, "member")
    assert anchor["turns"][1]["reasoning"].startswith("The file is a.txt; next")
    assert record["chain"][0]["record"]["rewrite"]["reroll"] == 3 and '"reroll": <the step' in calls[0]
    assert "did not pass the task's tests" in calls[0] and "Step 5" in calls[0]


def test_the_teachers_reroll_step_branches_n_times_with_the_chosen_credit(teacher, monkeypatch, tmp_path):
    teacher(REROLL)
    monkeypatch.setenv("PRL_TRANSFORM_ARGS",
                       '{"added": "reroll", "reroll_n": 3, "reroll_credit": "subgroup"}')
    kept = _run(_templated(_trial(reward=0.0, trial_dir=str(tmp_path))))
    [anchor] = kept[0].metadata["prl_transform"]["reroll"]["anchors"]
    assert (anchor["step"], anchor["n"], anchor["credit"]) == (2, 3, "subgroup")


@pytest.mark.parametrize("plan", [
    {"edits": [], "reroll": 9, "reason": "x"},                                                   # no such step
    {"edits": [], "reason": "x"},                                                                # no reroll key
    {"edits": [{"step": 3, "op": "delete", "why": "x"}], "reroll": 3, "reason": "x"},             # edit at the reroll
    {"edits": [{"step": 1, "op": "set_terminal", "terminal": "t", "why": "x"}], "reroll": 2, "reason": "x"},
    {"edits": [{"step": 1, "op": "delete", "why": "x"}], "reroll": None, "reason": "x"},          # edits without one
])
def test_a_bad_reroll_answer_is_asked_again_then_skipped(teacher, monkeypatch, plan):
    calls = teacher(plan)
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"added": "reroll"}')
    kept = _run(_templated(_trial(reward=0.0)))
    assert len(calls) == 2 and kept[0].metadata["prl_transform"]["chain"][0]["status"] == "skipped"
    assert "reroll" not in kept[0].metadata["prl_transform"]


def test_no_reroll_answer_and_passing_trials(teacher, monkeypatch):
    calls = teacher({"edits": [], "reroll": None, "reason": "impossible"})
    monkeypatch.setenv("PRL_TRANSFORM_ARGS", '{"added": "reroll"}')
    kept = _run(_templated(_trial(reward=0.0)))
    assert "reroll" not in kept[0].metadata["prl_transform"] and len(calls) == 1
    passed = _run(_templated(_trial()))  # reroll_successes off: not sent to the teacher
    assert len(calls) == 1 and passed[0].metadata["prl_transform"]["chain"][0]["record"]["rewrite"]["why"] \
        .startswith("successful trial")
