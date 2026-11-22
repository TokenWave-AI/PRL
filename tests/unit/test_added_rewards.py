"""prl/algo/added_rewards.py: Miles' normalization, with added samples as members of the group."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")  # prl.algo.added_rewards: the Miles image (CHAIN=unit)

from prl.algo.added_rewards import group_rewards

ARGS = SimpleNamespace(grpo_std_normalization=True, advantage_estimator="grpo")


def _sample(group, rollout, reward, added=None, tokens=10):
    return SimpleNamespace(group_index=group, rollout_id=rollout, train_metadata={"group_reward": added} if added else {},
                           loss_mask=[1] * tokens + [0], get_reward_value=lambda args: reward)


def test_without_added_samples_a_7_of_8_group_is_as_in_miles():
    samples = [_sample(0, r, 1.0 if r else 0.0) for r in range(8)]
    raw, out = group_rewards(ARGS, samples)
    assert raw == [0.0] + [1.0] * 7
    assert out[0] == pytest.approx(-2.47, abs=0.01) and out[1] == pytest.approx(0.354, abs=0.01)


def test_siblings_share_a_rollouts_value_and_groups_are_separate():
    samples = [_sample(0, 0, 1.0), _sample(0, 0, 1.0), _sample(0, 1, 0.0), _sample(1, 2, 1.0), _sample(1, 3, 1.0)]
    _, out = group_rewards(ARGS, samples)
    assert out[0] == out[1] > 0 > out[2] and out[3] == out[4] == 0.0


def test_an_added_sample_gives_an_all_fail_group_a_contrast():
    samples = [_sample(0, r, 0.0) for r in range(8)] + [_sample(0, 3, 0.0, added=1.0)]
    raw, out = group_rewards(ARGS, samples)
    assert raw == [0.0] * 9  # the metric still sees 8 failures
    assert out[8] == pytest.approx(2.67, abs=0.01) and out[0] == pytest.approx(-0.33, abs=0.01)
    assert sum(out) == pytest.approx(0.0, abs=1e-4)


def test_a_reroll_split_by_a_summary_is_one_member():
    reroll = [SimpleNamespace(**{**vars(_sample(0, 3, 0.0, added=1.0)),
                                 "train_metadata": {"group_reward": 1.0, "added_id": "reroll"}}) for _ in range(2)]
    one = [_sample(0, r, 0.0) for r in range(8)] + [_sample(0, 3, 0.0, added=1.0)]
    _, out = group_rewards(ARGS, [_sample(0, r, 0.0) for r in range(8)] + reroll)
    assert out == pytest.approx(group_rewards(ARGS, one)[1][:8] + [group_rewards(ARGS, one)[1][8]] * 2)


def test_an_all_pass_group_stays_at_zero():
    samples = [_sample(0, r, 1.0) for r in range(8)] + [_sample(0, 0, 1.0, added=1.0)]
    _, out = group_rewards(ARGS, samples)
    assert out == [0.0] * 9


def test_ppo_takes_raw_rewards_and_the_added_reward_as_it_is():
    samples = [_sample(0, 0, 0.0), _sample(0, 1, 1.0), _sample(0, 0, 0.0, added=1.0)]
    raw, out = group_rewards(SimpleNamespace(advantage_estimator="ppo"), samples)
    assert raw == [0.0, 1.0, 0.0] and out == [0.0, 1.0, 1.0]


def test_length_lowers_a_pass_only_and_leaves_the_raw_rewards(monkeypatch):
    monkeypatch.setenv("PRL_LENGTH_LAMBDA", "0.5")
    monkeypatch.setenv("PRL_LENGTH_CAP", "100")
    samples = [_sample(0, 0, 1.0, tokens=50), _sample(0, 1, 1.0, tokens=200), _sample(0, 2, 0.0, tokens=5)]
    raw, out = group_rewards(SimpleNamespace(advantage_estimator="ppo"), samples)
    assert raw == [1.0, 1.0, 0.0] and out == pytest.approx([0.75, 0.5, 0.0])


def test_a_rollouts_siblings_count_their_tokens_together(monkeypatch):
    monkeypatch.setenv("PRL_LENGTH_LAMBDA", "1")
    monkeypatch.setenv("PRL_LENGTH_CAP", "100")
    samples = [_sample(0, 0, 1.0, tokens=30), _sample(0, 0, 1.0, tokens=20)]
    _, out = group_rewards(SimpleNamespace(advantage_estimator="ppo"), samples)
    assert out == pytest.approx([0.5, 0.5])


def test_the_std_floor_keeps_an_all_pass_groups_length_signal_small(monkeypatch):
    monkeypatch.setenv("PRL_LENGTH_LAMBDA", "0.5")
    monkeypatch.setenv("PRL_LENGTH_CAP", "1000")
    samples = [_sample(0, 0, 1.0, tokens=100), _sample(0, 1, 1.0, tokens=200)]
    _, free = group_rewards(ARGS, samples)
    monkeypatch.setenv("PRL_REWARD_STD_FLOOR", "0.35")
    _, floored = group_rewards(ARGS, samples)
    assert free[0] == pytest.approx(0.707, abs=0.01)  # by the group's tiny std: as strong as passing against failing
    assert floored[0] == pytest.approx(0.025 / 0.35, abs=0.01) and floored[0] == pytest.approx(-floored[1], abs=1e-5)


# --- subgroup credit (step.reroll(credit="subgroup"), prl/algo/added_rewards.py) ------------------------------------

def _branch(group, rollout, step, j, reward, credit="subgroup"):
    return SimpleNamespace(group_index=group, rollout_id=rollout, loss_mask=[1] * 4, response_length=4,
                           get_reward_value=lambda args: 0.0,
                           train_metadata={"group_reward": reward, "added_id": f"reroll:{step}:{j}",
                                           "branch": {"anchor": step, "credit": credit}})


def _marked(group, rollout, reward, anchors, length=12):
    s = _sample(group, rollout, reward)
    s.response_length = length
    s.train_metadata = {"branch_anchors": [{"step": k, "start": start, "credit": "subgroup"} for k, start in anchors]}
    return s


def test_subgroup_branches_leave_the_group_and_set_the_trials_tail():
    """A worked example: 2 of 8 pass; trial 3 fails, one branch from its step 5 passes."""
    group = [_sample(0, r, 1.0 if r < 2 else 0.0) for r in range(8)]
    group[3] = _marked(0, 3, 0.0, [(5, 7)])
    samples = [*group, _branch(0, 3, 5, 0, 1.0)]
    _, out = group_rewards(ARGS, samples)
    alone = group_rewards(ARGS, [_sample(0, r, 1.0 if r < 2 else 0.0) for r in range(8)])[1]
    assert out[:8] == pytest.approx(alone)  # the branch is not a member: the group is as without it
    assert out[8] == pytest.approx(0.707, abs=0.01)  # against its one sibling, the trial's tail
    assert group[3].train_metadata["advantage_spans"] == [[7, 12, pytest.approx(-0.707, abs=0.01)]]
    assert "advantage_spans" not in group[0].train_metadata


def test_subgroup_siblings_that_tie_give_the_tail_zero_and_anchors_split_the_tail():
    trial = _marked(0, 0, 0.0, [(6, 9), (2, 4)])
    samples = [trial, _sample(0, 1, 1.0), _branch(0, 0, 2, 0, 0.0), _branch(0, 0, 2, 1, 0.0),
               _branch(0, 0, 6, 0, 1.0), _branch(0, 0, 6, 1, 1.0)]
    _, out = group_rewards(ARGS, samples)
    assert out[2] == out[3] == 0.0 and out[4] == out[5] > 0
    spans = trial.train_metadata["advantage_spans"]
    assert [s[:2] for s in spans] == [[4, 9], [9, 12]] and spans[0][2] == 0.0 and spans[1][2] < 0


def test_member_and_subgroup_branches_mix():
    trial = _marked(0, 0, 0.0, [(3, 5)])
    samples = [trial, _sample(0, 1, 0.0), _branch(0, 0, 1, 0, 1.0, credit="member"), _branch(0, 0, 3, 0, 1.0)]
    _, out = group_rewards(ARGS, samples)
    assert out[2] > 0 > out[0] == out[1]  # the member joined the group
    assert out[3] > 0 and trial.train_metadata["advantage_spans"][0][2] < 0


def test_subgroup_needs_grpo_and_its_trial():
    with pytest.raises(ValueError, match="GRPO"):
        group_rewards(SimpleNamespace(advantage_estimator="ppo"), [_sample(0, 0, 0.0), _branch(0, 0, 2, 0, 1.0)])
    with pytest.raises(ValueError, match="needs the trial"):
        group_rewards(ARGS, [_sample(0, 0, 0.0), _branch(0, 0, 2, 0, 1.0)])


# --- verified credit (step.reroll(credit="verified"), transforms/hint_reroll.py) --------------------------------------

def _verified(group, rollout, reward, anchors, length=12):
    s = _marked(group, rollout, reward, anchors, length)
    for anchor in s.train_metadata["branch_anchors"]:
        anchor["credit"] = "verified"
    return s


def test_a_confirmed_step_clears_the_trials_prefix_and_compares_its_tail():
    """2 of 8 pass; trial 3 fails, one of two branches from its step 5 (token 7) passes: the steps before it are not
    blamed, from it on the trial is compared with its branches, and the group is as without the branches."""
    group = [_sample(0, r, 1.0 if r < 2 else 0.0) for r in range(8)]
    group[3] = _verified(0, 3, 0.0, [(5, 7)])
    samples = [*group, _branch(0, 3, 5, 0, 1.0, credit="verified"), _branch(0, 3, 5, 1, 0.0, credit="verified")]
    _, out = group_rewards(ARGS, samples)
    alone = group_rewards(ARGS, [_sample(0, r, 1.0 if r < 2 else 0.0) for r in range(8)])[1]
    assert out[:8] == pytest.approx(alone)
    assert out[8] > 0 > out[9]  # the passing branch up, the failing one down
    spans = group[3].train_metadata["advantage_spans"]
    assert [s[:2] for s in spans] == [[0, 7], [7, 12]] and spans[0][2] == 0.0
    assert spans[1][2] == pytest.approx(out[9])  # the trial's tail failed like the second branch


def test_an_unconfirmed_step_leaves_the_trial_its_group_value():
    group = [_sample(0, r, 1.0 if r < 2 else 0.0) for r in range(8)]
    group[3] = _verified(0, 3, 0.0, [(5, 7)])
    samples = [*group, _branch(0, 3, 5, 0, 0.0, credit="verified"), _branch(0, 3, 5, 1, 0.0, credit="verified")]
    _, out = group_rewards(ARGS, samples)
    alone = group_rewards(ARGS, [_sample(0, r, 1.0 if r < 2 else 0.0) for r in range(8)])[1]
    assert out[:8] == pytest.approx(alone) and out[3] < 0  # pushed down as in GRPO, from start to end
    assert "advantage_spans" not in group[3].train_metadata
    assert out[8] == out[9] == 0.0  # tied with the trial: no signal


def test_verified_needs_grpo():
    with pytest.raises(ValueError, match="GRPO"):
        group_rewards(SimpleNamespace(advantage_estimator="ppo"),
                      [_sample(0, 0, 0.0), _branch(0, 0, 2, 0, 1.0, credit="verified")])
