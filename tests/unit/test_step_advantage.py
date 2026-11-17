import pytest

torch = pytest.importorskip("torch")  # the Miles image (CHAIN=unit)

from prl.algo.step_advantage import apply_step_weights, loss_mask_steps


def test_loss_mask_steps():
    m = torch.tensor([1, 1, 0, 0, 1, 0, 1, 1, 1])
    assert loss_mask_steps(m).tolist() == [0, 0, -1, -1, 1, -1, 2, 2, 2]
    assert loss_mask_steps(torch.tensor([0, 1, 1, 0])).tolist() == [-1, 0, 0, -1]


def test_apply_step_weights():
    loss_mask = torch.tensor([1, 1, 0, 0, 1, 0, 1, 1])  # 3 steps
    adv = [torch.full((8,), 0.5), torch.full((8,), -1.0)]
    ret = [a.clone() for a in adv]
    md = [{"step_weights": [1.0, 0.0, 1.0]}, {"step_weights": None}]
    a, r = apply_step_weights(adv, ret, [loss_mask, loss_mask], md)
    assert a[0].tolist() == [0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5]  # only step-1 tokens zeroed
    assert a[1].tolist() == [-1.0] * 8
    assert torch.equal(a[0], r[0])
    # explicit turn_ids override loss-mask segmentation (turn 1 untrained, steps = turn-1)
    md2 = [{"step_weights": [1.0, 0.0, 1.0], "turn_ids": [-1, -1, -1, 2, 2, -1, 3, 3]}]
    a2, _ = apply_step_weights([torch.full((8,), 0.5)], [torch.zeros(8)], [loss_mask], md2)
    assert a2[0].tolist() == [0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5]
    # no metadata at all -> identity
    a3, _ = apply_step_weights(adv, ret, [loss_mask, loss_mask], None)
    assert all(torch.equal(x, y) for x, y in zip(a3, adv))
    # context parallel: the rank holds a zigzag part of the response; weights are sliced the same way
    def zigzag(i, x):
        return torch.cat([x[2:5], x[6:8]])
    a4, _ = apply_step_weights([torch.full((5,), 0.5)], [torch.zeros(5)], [loss_mask], md[:1], zigzag)
    assert a4[0].tolist() == [0.5, 0.5, 0.0, 0.5, 0.5]


def test_ppo_returns_are_critic_targets():
    loss_mask = torch.tensor([1, 1, 0, 1])  # 2 steps
    adv, ret = [torch.full((4,), 2.0)], [torch.full((4,), 0.7)]
    md = [{"step_weights": [0.5, 0.0]}]
    a, r = apply_step_weights(adv, ret, [loss_mask], md, scale_returns=False)
    assert a[0].tolist() == [1.0, 1.0, 2.0, 0.0] and torch.equal(r[0], ret[0])


def test_token_weight_spans_override_the_step_weight():
    loss_mask = torch.tensor([1, 1, 1, 0, 1, 1])  # 2 steps
    md = [{"step_weights": [1.0, -1.0], "token_weight_spans": [[1, 2, -1.0]]},  # one token of step 0 pushed down
          {"token_weight_spans": [[4, 6, 0.0]]}]  # spans without step weights: 1 elsewhere
    a, _ = apply_step_weights([torch.full((6,), 0.5), torch.full((6,), 0.5)], [torch.zeros(6)] * 2,
                              [loss_mask, loss_mask], md)
    assert a[0].tolist() == [0.5, -0.5, 0.5, 0.5, -0.5, -0.5]
    assert a[1].tolist() == [0.5, 0.5, 0.5, 0.5, 0.0, 0.0]
    def zigzag(i, x):  # context parallel: spans are set on the whole response, then sliced like the loss mask
        return torch.cat([x[0:2], x[4:6]])
    a2, _ = apply_step_weights([torch.full((4,), 0.5)], [torch.zeros(4)], [loss_mask], md[:1], zigzag)
    assert a2[0].tolist() == [0.5, -0.5, -0.5, -0.5]


def test_an_added_sample_keeps_its_advantage_and_the_critic_keeps_its_value():
    advantages, returns = [torch.full((4,), 0.7), torch.full((4,), -0.3)], [torch.ones(4), torch.zeros(4)]
    values = [torch.full((4,), 0.3), torch.full((4,), 0.3)]
    adv, ret = apply_step_weights(advantages, returns, [torch.ones(4), torch.ones(4)], [{"group_reward": 1.0}, {}],
                                  scale_returns=False, values=values)
    assert adv[0] is advantages[0] and ret[0].tolist() == pytest.approx([0.3] * 4)
    assert adv[1] is advantages[1] and ret[1] is returns[1]


def test_advantage_spans_set_the_tail_then_the_weights_act():
    """Subgroup credit (added_rewards): from the rerolled step on, the trial takes its siblings' value."""
    loss_mask = torch.tensor([1, 1, 0, 1, 1, 0, 1, 1])  # 3 steps; step 1 starts at token 3
    md = [{"advantage_spans": [[3, 8, 0.7]], "step_weights": [1.0, 1.0, 0.5]}]
    a, r = apply_step_weights([torch.full((8,), -0.4)], [torch.full((8,), -0.4)], [loss_mask], md)
    assert a[0].tolist() == pytest.approx([-0.4, -0.4, -0.4, 0.7, 0.7, 0.7, 0.35, 0.35])
    assert torch.equal(a[0], r[0])  # GRPO: returns follow the advantages
    def zigzag(i, x):
        return torch.cat([x[0:2], x[5:8]])
    a2, _ = apply_step_weights([torch.full((5,), -0.4)], [torch.zeros(5)], [loss_mask], md, zigzag)
    assert a2[0].tolist() == pytest.approx([-0.4, -0.4, 0.7, 0.35, 0.35])
    # without weights the spans alone apply
    a3, _ = apply_step_weights([torch.full((8,), -0.4)], [torch.zeros(8)], [loss_mask], [{"advantage_spans": [[6, 8, 0.0]]}])
    assert a3[0].tolist() == pytest.approx([-0.4] * 6 + [0.0, 0.0])
