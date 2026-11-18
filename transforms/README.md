<p align="right"><b>English</b> · <a href="README.zh-CN.md">中文</a></p>

# Transform guide

Start with the repository [README](../README.md). This page has the full interface, the rules and the branch template.
Items marked **(planned)** are not implemented yet; several and repeated branches, `subgroup` and the branch template
are implemented and have passed the unit tests and a GPU smoke.

## 1. How your function is called

- Where: the function of the same name, `<name>(traj, **params)`, in `transforms/<name>.py`; it may be `async`. Its
  default parameter values are the run's settings; `PRL_TRANSFORM_ARGS='{"param": value}'` overrides them (in a
  chain, per name).
- When: after every **training** trajectory has finished and been graded, before the advantage. Evaluation
  trajectories never go through it.
- Returns: the `traj` it was given, or `None`. **Only a normal return takes effect**: if the function raises, calls
  `traj.skip(...)` or times out, the trajectory trains as it was, so a multi-step pipeline needs no rollback.
- A plain (not `async`) function runs in a thread, so blocking HTTP requests in it do not hold up other trajectories.

## 2. Interface

### Read

| | |
|---|---|
| `traj.task` | the task (the first prompt) |
| `traj.reward` | the test reward (0–1, writable) |
| `traj.steps` | one `Step` per model turn, in order; `step.index` starts at 0 |
| `traj.trial_dir` | the trajectory's directory (Harbor's records and the test results are in it) |
| `step.action` / `step.observation` | the model's output in this turn / what the environment returned after it (with the chat template) |
| `step.thinking` / `step.command` / `step.terminal` | the reasoning / the action JSON (Terminus-2: analysis, plan, commands, task_complete; None if it does not parse) / the terminal output (without the template; None on the last step) |

### Weights

| | |
|---|---|
| `step.mask()` | this step leaves the loss (under PPO the critic does not learn it either) |
| `step.weight = w` | this step's advantage × w. In GRPO a successful trajectory has A ≥ 0 and a failed one A ≤ 0, so a negative w reverses the direction |
| `step.weight_text(text, w, every=False)` | only the tokens of this text get w, the rest of the step keeps `step.weight`. `text` must occur exactly once in the step's output (`every=True`: every occurrence). Not in the same call as a rewrite |

### Rewrite

| | |
|---|---|
| `step.edit_thinking(old, new)` | replace `old` (which must occur exactly once) with `new` in the reasoning |
| `step.set_thinking(text)`, `step.set_command({...})`, `step.set_terminal("...")` | replace the whole reasoning, action or terminal output |
| `step.delete()` | delete this turn and the terminal output after it (not the last turn) |
| `traj.reward = r` | change the reward (under GRPO it is normalized with the rest of the group: turn every failure of a group into a success and the group has no signal left) |

- Only the changed text is re-encoded; the rest keeps the policy's own tokens.
- Rewritten text was not sampled by the policy, so turn off `--use-tis` and `--use-rollout-logprobs`.
- `rewrite` with `mask_new=True`: the teacher's text is context only, out of the loss; only the policy's own tokens
  train. In our experiments, training on the teacher's text directly pushed the entropy up and made the log-probs
  drift.
- A trajectory that the context summary split into several parts (`len(traj.samples) > 1`) cannot be rewritten.

### Branches (implemented)

`step.reroll(n=1, credit="member")`: after your function returns, the framework opens `n` new sandboxes (branches) for
this step. Each runs the commands before this step for real, in order (the rewritten ones, if there are any), then
the policy generates from this step on and the tests grade it. All branches run at once.

- A branch's part before this step is context only and does not train; the branch trains with its own reward, scored
  as `credit` says (see [credit](#credit-how-a-new-trajectory-is-scored) below): `member` counts it as one more member
  of the group; `subgroup` compares it only with its siblings from the same start (GRPO only).
- A trajectory can branch at several steps (call `reroll` on each), at most 16 times each.
- The original trajectory trains as usual (its masks and weights apply; a rewrite of the prefix goes into the replay
  only, the original stays as it was).
- Limits: no branch at or after a rewritten step; not for trajectories split by the context summary; only one
  transform of a chain may ask for branches.
- A run with branches needs the branch post-processing: the arm `ota_train_transform_reroll` (it adds
  `--custom-reward-post-process-path prl.algo.added_rewards.group_rewards`).
- Results: one `reroll/s<step>_b<j>/` per branch in the trajectory's directory (the request, the outcome, and
  `reroll_replay.json`: for each replayed turn, whether its terminal output matched the original, with both sides
  when it did not); metrics `rollout/reroll/*` (`requested_trials`, `branches`, `{ok,failed,subgroup}_branches`,
  `reward_mean`, `source_reward_mean`, `seconds_mean`).
- The replayed terminal output is real and differs a little from the original (times, inodes, git hashes and the
  like); the framework records this and does not block it.

### Calling a model

```python
k = await ask(prompt, check=fn)     # prompt is a string or a list of messages
```

- Uses the run's judge model (`PRL_JUDGE_*`, e.g. `configs/arms/ota_judge.env`).
- `check(answer)` returns the value to use; if it raises, the answer is rejected, and the framework tells the model
  why and asks again. If no answer passes, `ask` raises `Unanswered` and the trajectory trains as it was.
- Retries, waiting out 429s, concurrency, and switching in turn to the models of `PRL_JUDGE_FALLBACK_MODELS` when a
  content filter refuses are all handled by the framework.
- Every call (request, answer, time, token counts, the model that answered) is saved to
  `<trial_dir>/transform_calls.json`.

### Other

| | |
|---|---|
| `traj.skip("reason")` | leave this trajectory alone; it trains as it was (not a failure) |
| `traj.record["key"] = value` | saved to `<trial_dir>/transform.json` |
| `traj.add_sample(sample, reward)`, `traj.demonstrate(step, action, reward)` | advanced: build extra training samples yourself |

## 3. Branch template

`TRANSFORM=branch`. Every method that rolls out again from the middle answers three questions; each question is a
replaceable plug-in:

```bash
TRANSFORM=branch PRL_TRANSFORM_ARGS='{"select": "first_bad", "allocate": {"name": "fixed", "n": 2, "view": "trial"}, "credit": "subgroup"}'
```

A plug-in name is looked up among the built-ins first, then as the function of the same name in
`transforms/<name>.py` (as for a transform); with parameters, write `{"name": ..., params...}`. The template's own
parameters: `min_step` (default 1: a branch from step 0 is just a rerun and is dropped) and `max_anchors` (default 1:
at most this many steps, in the order select gives). The chosen steps are recorded in `traj.record["branch"]`. Code:
`transforms/branch.py`.

### select: where to branch

`select(traj, **params) -> list of steps` (`Step`s or step numbers from 0, optionally with a score as
`(step, score)`, best first). select may also edit the steps before (for the replay only).

| Built-in | Meaning |
|---|---|
| `middle` | the middle step (for testing the framework) |
| `teacher` | a teacher reads the trajectory and picks a step, possibly after small edits to the steps before it (= `rewrite` with `added="reroll"`) |
| `first_bad` | the first bad step the judge marks |
| `surprisal` | the step that surprised the policy most: the highest mean −logprob of its sampled tokens (an approximation of the entropy); parameter `top` |
| `entropy`, `post_tool_entropy` (later) | the true token entropy (FR3E); steps where the entropy rises after a tool output (ARPO). They need top-k logprobs from the inference server |

### allocate: how many times

`allocate(traj, anchors, **params) -> [(step, count)]`. Built-in `fixed`: `n` per step; `only` is `"failed"`
(default, reward 0), `"passed"` or `"all"`, checked before select, so an excluded trajectory costs no model call.
`view` decides when, and with how much information, the decision is made:

| `view` | Decided when | What it can do | Cost |
|---|---|---|---|
| `"trial"` (default) | as soon as this trajectory finishes | by its own outcome: `fixed`, n per step, failures only / successes only | almost no extra time: the branches run alongside the group's other trajectories |
| `"group"` (planned; configuring it now raises) | when the whole group has finished | by the whole group: `zero_signal_groups` only for all-correct or all-wrong groups; `by_score` by select's scores; a total budget per step | waits for the slowest trajectory of the group, so every step gets longer |

Why decide by group: in GRPO an all-correct or all-wrong group has zero advantage throughout and teaches nothing. In
the first 20 steps (640 groups) of our 9B training, 24% of the groups were all wrong and 31% all correct. In an
all-wrong group, one successful branch is the group's only positive signal.

### credit: how a new trajectory is scored

Example: a group of 8, 2 of them successful. A failed trajectory τ branches at step 5, and the branch succeeds.

| `credit` | How | In this example |
|---|---|---|
| `member` (default) | the branch is one more member of the group, normalized with everyone | the group mean goes from 0.25 to 0.33; all of τ's steps (the first 4 too) are still pushed down as a whole; the other 7 are pushed down a little too |
| `subgroup` | the branch is compared only with those from the same start: τ from step 5 on, plus the other branches at that step. If the siblings all score the same, τ's advantage from that step on is 0 (it is compared only with the alternatives, no longer pushed by the whole outcome) | τ after step 5 is pushed down, the branch is pushed up; τ's first 4 steps keep their group advantage; the other 7 are not affected |
| `value` (later) | estimate the step's value as the mean score of every continuation from the same start, and weight the original trajectory's steps by it (the idea of VinePPO / FR3E) | — |

Trade-offs: `member` is simple and verified, but a branch starts with a prefix it got for free and is not fully
comparable with trajectories run from the start. `subgroup` compares the same state with different moves, so the
signal lands exactly on the step where they part, but with one branch per step only 2 are comparable and it is noisy:
use n ≥ 2. Branches work with GRPO and PPO only (not REINFORCE++); `subgroup` with GRPO only (PPO's advantage comes
from the critic).

### Existing methods as templates

| Method | select | allocate | credit |
|---|---|---|---|
| a teacher edits, then the policy rolls on | `teacher` | `fixed`, n=1, failures only | `member` |
| FR3E | `entropy`, top-K | M per step | `value` |
| ARPO | `post_tool_entropy` | Z when the entropy rises past a threshold | `subgroup` |
| judge + branches | `first_bad` | `fixed`, n=2 | `subgroup`, with a negative weight on the original's bad steps too |

## 4. What the framework does for you

| | |
|---|---|
| Check before the run | every training job first runs your function on 8 recorded trajectories (model calls included); if it raises, returns an invalid result, hangs or skips them all, the job stops before training |
| Failures | a failure on one trajectory: it trains as it was and the reason is recorded; 32 failures in a row: training stops, so that it does not silently turn into plain GRPO |
| Consistency checks | every sample passes Miles' checks (tokens, loss mask, logprobs, length); the reward changes only if you wrote `traj.reward`; a sample with no trainable token is dropped |
| Records | `transform.json` per trajectory: each function's status and time, the steps it changed (`weighted_steps` {step: new weight}, `masked_steps`, `token_weighted_steps`, `rewritten_steps`, `deleted_steps`, `reroll_steps`), the reward before / after, the trainable tokens before / after; matching `rollout/*` metrics in the training log |
| Framework smoke | needed once after the framework code changes (two updates, about half an hour); it writes a receipt, and training jobs find the receipt that matches the current code |

## 5. Settings

| Environment variable | Default | |
|---|---|---|
| `PRL_TRANSFORM` | none | a name, or a chain joined by commas |
| `PRL_TRANSFORM_ARGS` | none | JSON; per name in a chain |
| `PRL_TRANSFORM_TIMEOUT` | 1800 | seconds allowed on one trajectory |
| `PRL_TRANSFORM_MAX_FAILURES` | 32 | failures in a row that stop training (0: never) |
| `PRL_REROLL_TIMEOUT` | 0 (no limit) | seconds allowed for one branch; a branch that takes longer is not added |
| `PRL_REROLL_CONCURRENCY` | 32 | branches running at once per rollout worker |
| `PRL_JUDGE_*`, `PRL_JUDGE_FALLBACK_MODELS` | see the judge settings | the model `ask` uses, its key file, concurrency, fallback models |

## 6. Ready transforms

| Name | What it does | Main parameters |
|---|---|---|
| `atlas` | a judge picks the critical steps; the others are masked (or scaled by `other_weight`) | `other_weight` |
| `atlas_signed` | a judge marks steps good / bad / neutral; bad steps of successful trajectories are pushed the other way | `mode` (mild / strict), `bad_weight`, `failed_good_weight`, `quote_bad` (push only the span that went wrong) |
| `rewrite` | a teacher makes small edits; or fixes failures only; or picks a step to branch from | `prompt`, `mask_new`, `successes` / `failures`, `added="reroll"`, limits on how much it may change |
| `reroll_smoke` | for testing the framework: branches from a middle step | `min_steps`, `edit`, `every` |

Each file's docstring describes all of its parameters.
