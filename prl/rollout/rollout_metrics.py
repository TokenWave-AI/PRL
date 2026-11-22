"""Extra per-update rollout metrics, logged in the same ``perf N`` record as Miles' own (``--custom-rollout-log-function-path
prl.rollout.rollout_metrics.log_rollout_data``).

Miles logs the reward and response-length means; this adds what the usual RL dashboards also show:
- ``rollout/raw_reward_{std,min,max}``
- ``rollout/prompt_len/{mean,max}``
- per trial (a summarized trial has several samples, counted once): ``rollout/trial_output_tokens/{mean,median,p90,max}``
  and ``rollout/trial_turns/{mean,median,p90,max}`` from the agent's metrics
- ``rollout/sampled_token_nll``: mean -log p of the sampled tokens under the rollout policy over every generated
  token, whatever the loss mask: at temperature 1 an estimate of the policy entropy that is comparable between
  masked (ATLAS) and plain runs; ``train/entropy_loss`` covers the loss-masked tokens only
- a trajectory transform (``prl_transform`` in the trial metadata, prl/transforms/core.py):
  ``rollout/transform/{ok,skipped,error}_trials`` (a trial counts as its worst status in the chain),
  ``rollout/transform/changed_trials``, ``rollout/transform/kept_token_frac`` (trained tokens after / before) and
  ``rollout/transform/new_token_frac`` (trained tokens the policy did not sample, of those after)
- ATLAS: ``rollout/atlas/{critical_step_frac,kept_token_frac}`` over the labeled trials, ``rollout/atlas/unlabeled_trials``
  (no label: the judge gave no usable answer, or the transform failed; all steps kept). From the ``atlas`` transform's
  record, or ``prl_atlas`` (the metadata key of an earlier token-mask version of ATLAS, still read)
- signed ATLAS (transforms/atlas_signed.py): ``rollout/atlas_signed/{good_step_frac,bad_step_frac}`` over the graded
  trials' steps, ``rollout/atlas_signed/pushed_down_steps`` (bad steps of passed trials: weight -1),
  ``rollout/atlas_signed/{graded_trials,ungraded_trials}`` (ungraded: the judge gave no usable answer or the transform
  failed; mild leaves failed trials ungraded on purpose and does not count them), and with ``failed_good_weight``
  ``rollout/atlas_signed/protected_steps`` (good steps of failed trials: weight failed_good_weight instead of 1); with
  ``quote_bad`` ``rollout/atlas_signed/{quoted_steps,quote_fallback_steps}`` (bad steps of passed trials pushed down on
  the quoted tokens only / on the whole step because the quote was not found)
- rerolls (``step.reroll()``, prl/rollout/harbor_training.py): ``rollout/reroll/requested_trials`` (trials with a
  branch), ``rollout/reroll/branches`` and ``{ok,failed,subgroup}_branches``,
  ``rollout/reroll/reward_mean`` (the rerolls' verifier rewards) against ``rollout/reroll/source_reward_mean`` (the
  rerolled trials'), ``rollout/reroll/seconds_mean``; with ``credit="verified"`` ``rollout/reroll/verified_steps``
  (rerolled steps) and ``rollout/reroll/confirmed_steps`` (those with a branch that scored more than the trial); a
  reroll's own samples are not a trial in the other statistics
- ``rollout/abort_retried_trials``: trials kept after an aborted attempt was run again on its own
  (``prl_abort_retries``, prl/rollout/harbor_training.py); the trials that aborted every time are not in
  ``samples`` (Miles refilled their groups)
Then Miles' default logging runs with these merged in.
"""

from __future__ import annotations

import ast
import copy
import statistics as st


def _dict(v):
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return {}
    return v if isinstance(v, dict) else {}


def _p(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def _stats(prefix, values):
    if not values:
        return {}
    return {f"{prefix}/mean": float(st.mean(values)), f"{prefix}/median": float(st.median(values)),
            f"{prefix}/p90": float(_p(values, 0.9)), f"{prefix}/max": float(max(values))}


def compute(args, samples) -> dict:
    out = {}
    trial_samples = [s for s in samples if "prl_reroll" not in (s.metadata or {})]  # a reroll carries its trial's reward
    rewards = [float(s.get_reward_value(args)) for s in trial_samples]
    if rewards:
        out["rollout/raw_reward_std"] = float(st.pstdev(rewards))
        out["rollout/raw_reward_min"] = min(rewards)
        out["rollout/raw_reward_max"] = max(rewards)
    prompts = [len(s.tokens) - s.response_length for s in trial_samples if s.tokens is not None]
    if prompts:
        out["rollout/prompt_len/mean"] = float(st.mean(prompts))
        out["rollout/prompt_len/max"] = float(max(prompts))

    trials = {}
    for i, s in enumerate(samples):
        md = s.metadata or {}
        trials.setdefault(md.get("trial_dir") or f"sample{i}", []).append(s)
    turns, outs, nll, gen = [], [], 0.0, 0
    crit = steps = kept = trained = unlabeled = 0
    signed = {"steps": 0, "good": 0, "bad": 0, "pushed_down": 0, "protected": 0, "graded": 0, "ungraded": 0,
              "quoted": 0, "quote_fallback": 0}
    protects = quotes = False
    status = {"ok": 0, "skipped": 0, "error": 0}
    t_trials = t_changed = t_before = t_after = t_new = 0
    abort_retried = 0
    reroll = {"requested": 0, "branches": 0, "ok": 0, "failed": 0, "subgroup": 0, "reward": [], "source": [],
              "seconds": [], "verified": 0, "confirmed": 0}
    for ss in trials.values():
        md = ss[0].metadata or {}
        if "prl_reroll" in md:
            continue  # counted from the trial it rerolled
        abort_retried += bool(md.get("prl_abort_retries"))
        am = _dict(md.get("agent_metrics"))
        if am.get("n_episodes") is not None:
            turns.append(am["n_episodes"])
        if am.get("n_output_tokens") is not None:
            outs.append(am["n_output_tokens"])
        at = _dict(md.get("prl_atlas")) if "prl_atlas" in md else None
        tr = _dict(md.get("prl_transform")) if "prl_transform" in md else None
        generated = None  # the agent's generated tokens in these samples, when a transform changed the loss mask
        if at:
            generated = at["trained_tokens"]
        elif tr:
            t_trials += 1
            branches = [o for o in (tr.get("reroll") or []) if isinstance(o, dict) and o.get("status") in ("ok", "failed")]
            reroll["requested"] += bool(branches)
            for ro in branches:
                reroll["branches"] += 1
                reroll[ro["status"]] += 1
                reroll["subgroup"] += ro.get("credit") == "subgroup"
                reroll["seconds"].append(ro.get("seconds") or 0.0)
                if ro["status"] == "ok":
                    reroll["reward"].append(float(ro["reward"]))
                    reroll["source"].append(float(ss[0].get_reward_value(args)))
            # verified credit: a step is confirmed when one of its branches scored more than the trial
            verified = {ro.get("step") for ro in branches if ro.get("credit") == "verified"}
            if verified:
                own = float(ss[0].get_reward_value(args))
                reroll["verified"] += len(verified)
                reroll["confirmed"] += sum(any(ro["status"] == "ok" and ro.get("step") == k and float(ro["reward"]) > own
                                               for ro in branches) for k in verified)
            chain = tr.get("chain") or []
            worst = [e.get("status") for e in chain]
            status["error" if "error" in worst else "skipped" if "skipped" in worst else "ok"] += 1
            t_changed += bool(tr.get("changed"))
            t_before, t_after = t_before + tr["trained_tokens_before"], t_after + tr["trained_tokens_after"]
            t_new += tr.get("new_tokens") or 0
            # Rewritten tokens were not sampled, and a dropped sample (left without a trained token) takes its tokens
            # with it, although they count in trained_tokens_before: the sampled-token NLL skips such trials
            dropped = tr.get("dropped_samples", 0)
            generated = None if tr.get("tokens_changed") or dropped else tr["trained_tokens_before"]
            for e in chain:
                if e.get("transform") == "atlas_signed":
                    g = (e.get("record") or {}).get("atlas_signed") or {}
                    if e.get("status") == "ok" and g.get("graded"):
                        signed["graded"] += 1
                        signed["steps"] += g["steps"]
                        signed["good"], signed["bad"] = signed["good"] + len(g["good"]), signed["bad"] + len(g["bad"])
                        signed["pushed_down"] += len(g["bad"]) if g["passed"] else 0
                        if (e.get("params") or {}).get("failed_good_weight") is not None:
                            protects = True
                            signed["protected"] += 0 if g["passed"] else len(g["good"])
                        if "quoted" in g:
                            quotes = True
                            signed["quoted"] += g["quoted"]
                            signed["quote_fallback"] += g["quote_fallback"]
                    elif not (e.get("status") == "ok" and g.get("mode") == "mild" and g.get("passed") is False):
                        signed["ungraded"] += 1  # mild leaves failed trials ungraded on purpose
                    continue
                if e.get("transform") != "atlas":
                    continue
                a = (e.get("record") or {}).get("atlas")
                if e.get("status") != "ok" or not a or a.get("critical") is None:
                    unlabeled += 1
                else:
                    crit, steps = crit + len(a["critical"]), steps + a["steps"]
                    kept, trained = kept + tr["trained_tokens_after"], trained + tr["trained_tokens_before"]
        lps = [s.rollout_log_probs for s in ss if s.rollout_log_probs]
        if lps and not (tr and generated is None):
            # tokens the agent generated: the trial's trained tokens before a transform's mask, else the loss mask;
            # every other token has log-prob 0
            nll += -sum(sum(lp) for lp in lps)
            gen += generated if generated is not None else sum(sum(s.loss_mask or []) for s in ss)
        if at and at["critical"] is None:
            unlabeled += 1
        elif at:
            crit, steps = crit + at["critical"], steps + at["steps"]
            kept, trained = kept + at["kept_tokens"], trained + at["trained_tokens"]
    if reroll["requested"]:
        out["rollout/reroll/requested_trials"] = reroll["requested"]
        out |= {f"rollout/reroll/{k}_branches": reroll[k] for k in ("ok", "failed", "subgroup")}
        out["rollout/reroll/branches"] = reroll["branches"]
        out["rollout/reroll/seconds_mean"] = float(st.mean(reroll["seconds"]))
        if reroll["verified"]:
            out["rollout/reroll/verified_steps"] = reroll["verified"]
            out["rollout/reroll/confirmed_steps"] = reroll["confirmed"]
        if reroll["reward"]:
            out["rollout/reroll/reward_mean"] = float(st.mean(reroll["reward"]))
            out["rollout/reroll/source_reward_mean"] = float(st.mean(reroll["source"]))
    if any("prl_abort_retries" in (s.metadata or {}) for s in samples):
        out["rollout/abort_retried_trials"] = abort_retried
    out |= _stats("rollout/trial_output_tokens", outs)
    out |= _stats("rollout/trial_turns", turns)
    if gen:
        out["rollout/sampled_token_nll"] = nll / gen
    if steps:
        out["rollout/atlas/critical_step_frac"] = crit / steps
        out["rollout/atlas/kept_token_frac"] = kept / max(1, trained)
    if steps or unlabeled:
        out["rollout/atlas/unlabeled_trials"] = unlabeled
    if signed["steps"]:
        out["rollout/atlas_signed/good_step_frac"] = signed["good"] / signed["steps"]
        out["rollout/atlas_signed/bad_step_frac"] = signed["bad"] / signed["steps"]
    if signed["graded"] or signed["ungraded"]:
        out |= {f"rollout/atlas_signed/{k}_trials": signed[k] for k in ("graded", "ungraded")} | {
            "rollout/atlas_signed/pushed_down_steps": signed["pushed_down"]}
        if protects:
            out["rollout/atlas_signed/protected_steps"] = signed["protected"]
        if quotes:
            out["rollout/atlas_signed/quoted_steps"] = signed["quoted"]
            out["rollout/atlas_signed/quote_fallback_steps"] = signed["quote_fallback"]
    if t_trials:
        out |= {f"rollout/transform/{k}_trials": v for k, v in status.items()}
        out["rollout/transform/changed_trials"] = t_changed
        out["rollout/transform/kept_token_frac"] = t_after / max(1, t_before)
        out["rollout/transform/new_token_frac"] = t_new / max(1, t_after)
    return out


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time, *rest, **kw):
    from miles.ray.rollout import metrics as miles_metrics

    try:
        extra = compute(args, samples)
    except Exception as e:  # never lose Miles' own record over an extra metric
        print(f"[prl] rollout metrics failed: {e!r}", flush=True)
        extra = {}
    plain = copy.copy(args)
    plain.custom_rollout_log_function_path = None
    miles_metrics.log_rollout_data(rollout_id, plain, samples, {**(rollout_extra_metrics or {}), **extra},
                                   rollout_time, *rest, **kw)
    return True
