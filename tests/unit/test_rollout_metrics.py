from types import SimpleNamespace

from prl.rollout import rollout_metrics as rm


def _sample(trial, reward, lp, mask, atlas=None, turns=3, out=10, transform=None):
    md = {"trial_dir": trial, "agent_metrics": str({"n_episodes": turns, "n_output_tokens": out})}
    if atlas:
        md["prl_atlas"] = atlas
    if transform:
        md["prl_transform"] = transform
    return SimpleNamespace(metadata=md, tokens=[0] * (5 + len(lp)), response_length=len(lp), rollout_log_probs=lp,
                           loss_mask=mask, get_reward_value=lambda args, r=reward: r)


def test_compute_counts_each_trial_once_and_atlas_shares():
    atlas = {"steps": 4, "critical": 1, "kept_tokens": 2, "trained_tokens": 4}
    samples = [
        # an ATLAS trial in two leaves: 4 generated tokens, 2 kept (mask), log-probs 0 elsewhere
        _sample("a", 1.0, [-0.5, -0.5, 0.0], [1, 1, 0], atlas, turns=4, out=40),
        _sample("a", 1.0, [-1.0, -1.0], [0, 0], atlas, turns=4, out=40),
        # the judge gave no label: all steps kept, out of the ATLAS shares
        _sample("b", 0.0, [-0.2, -0.2], [1, 1], {"steps": 2, "critical": None, "kept_tokens": 2, "trained_tokens": 2},
                turns=2, out=20),
    ]
    out = rm.compute(SimpleNamespace(), samples)
    assert out["rollout/raw_reward_max"] == 1.0 and out["rollout/raw_reward_min"] == 0.0
    assert out["rollout/trial_turns/mean"] == 3 and out["rollout/trial_output_tokens/max"] == 40
    assert abs(out["rollout/sampled_token_nll"] - (3.0 + 0.4) / (4 + 2)) < 1e-9
    assert out["rollout/atlas/critical_step_frac"] == 0.25 and out["rollout/atlas/kept_token_frac"] == 0.5
    assert out["rollout/atlas/unlabeled_trials"] == 1
    assert out["rollout/prompt_len/max"] == 5


def _record(status, before, after, atlas=None, transform="atlas", **kw):
    entry = {"transform": transform, "status": status}
    if atlas is not None:
        entry["record"] = {"atlas": atlas}
    return {"chain": [entry], "trained_tokens_before": before, "trained_tokens_after": after,
            "tokens_changed": False, "new_tokens": 0, "changed": after != before, "dropped_samples": 0, **kw}


def test_compute_reads_the_transform_record():
    samples = [
        # atlas kept 1 of 4 steps: 2 of 4 generated tokens trained
        _sample("a", 1.0, [-0.5, -0.5, -1.0, -1.0], [1, 1, 0, 0], transform=_record("ok", 4, 2, {"steps": 4, "critical": [2]})),
        # the judge gave no usable answer: skipped, all steps kept
        _sample("b", 0.0, [-0.2, -0.2], [1, 1], transform=_record("skipped", 2, 2, {"steps": 2, "critical": None})),
        # the transform timed out: error, no record, trained unchanged
        _sample("c", 0.0, [-0.1], [1], transform=_record("error", 1, 1)),
        # a rewrite put in tokens the policy did not sample: out of the sampled-token NLL
        _sample("d", 1.0, [-9.0, 0.0], [1, 1], transform=_record("ok", 2, 2, None, "rewrite", tokens_changed=True,
                                                                 new_tokens=1, changed=True)),
        # two leaves, one dropped by the mask: its tokens are gone, so out of the NLL as well
        _sample("e", 1.0, [-9.0], [1], transform=_record("ok", 3, 1, {"steps": 3, "critical": [3]},
                                                         dropped_samples=1)),
    ]
    out = rm.compute(SimpleNamespace(), samples)
    assert abs(out["rollout/sampled_token_nll"] - (3.0 + 0.4 + 0.1) / (4 + 2 + 1)) < 1e-9
    assert out["rollout/atlas/critical_step_frac"] == 2 / 7 and out["rollout/atlas/kept_token_frac"] == 3 / 7
    assert out["rollout/atlas/unlabeled_trials"] == 2
    assert (out["rollout/transform/ok_trials"], out["rollout/transform/skipped_trials"],
            out["rollout/transform/error_trials"]) == (3, 1, 1)
    assert out["rollout/transform/changed_trials"] == 3
    assert out["rollout/transform/kept_token_frac"] == 8 / 12 and out["rollout/transform/new_token_frac"] == 1 / 8



def test_compute_reads_signed_atlas():
    def signed(trial, reward, status, grade):
        record = _record(status, 3, 3, transform="atlas_signed")
        record["chain"][0]["record"] = {"atlas_signed": grade}
        return _sample(trial, reward, [-0.1], [1], transform=record)

    samples = [
        signed("a", 1.0, "ok", {"mode": "mild", "steps": 4, "passed": True, "graded": True, "good": [1], "bad": [2, 3]}),
        signed("b", 0.0, "ok", {"mode": "strict", "steps": 6, "passed": False, "graded": True, "good": [], "bad": [4]}),
        signed("c", 0.0, "ok", {"mode": "mild", "steps": 5, "passed": False, "graded": False}),  # not asked: not counted
        signed("d", 1.0, "skipped", {"mode": "mild", "steps": 2, "passed": True, "graded": False}),  # the judge failed
    ]
    out = rm.compute(SimpleNamespace(), samples)
    assert out["rollout/atlas_signed/good_step_frac"] == 1 / 10 and out["rollout/atlas_signed/bad_step_frac"] == 3 / 10
    assert (out["rollout/atlas_signed/graded_trials"], out["rollout/atlas_signed/ungraded_trials"],
            out["rollout/atlas_signed/pushed_down_steps"]) == (2, 1, 2)
    assert "rollout/atlas_signed/protected_steps" not in out  # failed trials not graded (failed_good_weight None)


def test_compute_counts_protected_steps_of_failed_trials():
    def signed(trial, reward, grade):
        record = _record("ok", 3, 3, transform="atlas_signed")
        record["chain"][0].update(record={"atlas_signed": grade}, params={"mode": "mild", "failed_good_weight": 0.0})
        return _sample(trial, reward, [-0.1], [1], transform=record)

    out = rm.compute(SimpleNamespace(), [
        signed("a", 0.0, {"mode": "mild", "steps": 4, "passed": False, "graded": True, "good": [1, 2], "bad": [3]}),
        signed("b", 1.0, {"mode": "mild", "steps": 4, "passed": True, "graded": True, "good": [1], "bad": [2]})])
    assert out["rollout/atlas_signed/protected_steps"] == 2 and out["rollout/atlas_signed/pushed_down_steps"] == 1
    assert out["rollout/atlas_signed/graded_trials"] == 2
    assert "rollout/atlas_signed/quoted_steps" not in out


def test_compute_counts_quoted_steps():
    record = _record("ok", 3, 3, transform="atlas_signed")
    record["chain"][0]["record"] = {"atlas_signed": {"mode": "mild", "steps": 4, "passed": True, "graded": True,
                                                     "good": [], "bad": [1, 2, 3], "quoted": 2, "quote_fallback": 1}}
    out = rm.compute(SimpleNamespace(), [_sample("a", 1.0, [-0.1], [1], transform=record)])
    assert (out["rollout/atlas_signed/quoted_steps"], out["rollout/atlas_signed/quote_fallback_steps"]) == (2, 1)

def test_log_merges_into_miles_record(monkeypatch):
    import sys
    import types

    seen = {}
    mod = types.ModuleType("miles.ray.rollout.metrics")
    mod.log_rollout_data = lambda rid, args, samples, extra, t, *a, **k: seen.update(extra=extra, path=args.custom_rollout_log_function_path)
    for name in ("miles", "miles.ray", "miles.ray.rollout"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "miles.ray.rollout.metrics", mod)
    sys.modules["miles.ray.rollout"].metrics = mod
    args = SimpleNamespace(custom_rollout_log_function_path="x")
    assert rm.log_rollout_data(0, args, [_sample("a", 1.0, [-0.1], [1])], {"m": 1}, 1.0) is True
    assert seen["path"] is None and seen["extra"]["m"] == 1 and "rollout/raw_reward_std" in seen["extra"]
    assert args.custom_rollout_log_function_path == "x"
