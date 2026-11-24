<p align="right"><a href="README.md">English</a> · <b>中文</b></p>

# transform 详细说明

入门看仓库根目录的 [README](../README.zh-CN.md)。这里是完整的接口、规则和分支模板。标 **（规划中）** 的还没实现；分支的多次 / 多处、
`subgroup` 和分支模板已实现，单测和 GPU 烟测都已通过。

## 1. 函数怎么被调用

- 位置：`transforms/<名字>.py` 里的同名函数 `<名字>(traj, **params)`，可以是 `async`。默认参数就是运行的设置，
  `PRL_TRANSFORM_ARGS='{"参数": 值}'` 覆盖（串联时按名字分开写）。
- 时机：每条**训练**轨迹跑完、拿到测试分数之后，进 advantage 之前。评测轨迹永远不经过它。
- 返回：传进来的 `traj`，或 `None`。**只有正常返回才生效**：中途抛错、`traj.skip(...)`、超时，这条轨迹都照原样训练，
  所以多步流程不需要自己回滚。
- 同步函数在线程里跑，里面做阻塞的 HTTP 请求不会卡住其他轨迹。

## 2. 接口参考

### 读

| | |
|---|---|
| `traj.task` | 题目的 prompt：harness 的整段第一条 prompt（先是它的说明，再是题目），带模型的 chat template（`<\|im_start\|>user` …… `<think>`） |
| `traj.reward` | 测试分数（0–1，可写） |
| `traj.steps` | 每轮模型输出一个 `Step`，按顺序；`step.index` 从 0 开始 |
| `traj.trial_dir` | 这条轨迹的目录（Harbor 的记录、测试结果都在里面） |
| `step.action` / `step.observation` | 这一轮模型的输出 / 之后环境返回的内容（带 chat 模板） |
| `step.thinking` / `step.command` / `step.terminal` | 推理部分 / 动作 JSON（Terminus-2：analysis、plan、commands、task_complete；解析不了是 None）/ 终端输出（去掉模板；最后一步是 None） |

Terminus-2 一步的 `command`：`{"analysis": "...", "plan": "...", "commands": [{"keystrokes": "ls -la\n", "duration": 0.1}], "task_complete": false}`；
keystrokes 以 `\n` 结尾才执行，`C-c` 这类是按键，空的表示等待。`step.terminal` 是这一步之后的屏幕，提示符和敲的命令
也在里面：从命令里取的字符串也会匹配到它的回显。


### 权重

| | |
|---|---|
| `step.mask()` | 这一步不进 loss（PPO 下 critic 也不学这一步） |
| `step.weight = w` | 这一步的 advantage × w。GRPO 里成功轨迹 A ≥ 0、失败轨迹 A ≤ 0，所以负的 w 会把方向反过来 |
| `step.weight_text(text, w, every=False)` | 只有这段文字的 token 用 w，其余用 `step.weight`。`text` 必须在这一步的输出里出现恰好一次（`every=True`：每次出现）；从 `step.command` 取的命令 keystrokes 会按回复里的写法（JSON 转义后）去找。不能和改写用在同一次调用里 |

串联时，后一个函数拿到的是前一个留下的状态：`step.weight = w` 会替换前面的权重，`step.weight *= w` 是叠加；
`step.masked` 表示这一步已被 mask。

### 改写

| | |
|---|---|
| `step.edit_thinking(old, new)` | 推理里把 `old`（必须恰好出现一次）换成 `new` |
| `step.set_thinking(text)`、`step.set_command({...})`、`step.set_terminal("...")` | 整段替换推理、动作、终端输出 |
| `step.delete()` | 删掉这一轮和它之后的终端输出（最后一轮不能删） |
| `traj.reward = r` | 改分数（GRPO 下会和组里其他轨迹一起归一化：把组里的失败都改成成功，这组就没有信号了） |

- 只重新编码被改的文字，其余保留策略自己的 token。
- 改写的文字不是策略采样的，所以要关掉 `--use-tis` 和 `--use-rollout-logprobs`。
- `rewrite` 的 `mask_new=True`：教师写的文字只做上下文、不进 loss，只训练策略自己的 token。我们的实验里，直接训练教师的文字会推高熵、
  让 logprob 漂移。
- 被上下文总结切成几段的轨迹（`len(traj.samples) > 1`）不能改写。

### 分支（已实现）

`step.reroll(n=1, credit="member", hint=None)`：函数返回后，框架为这一步开 `n` 个新沙盒（分支），每个都把这一步之前的命令按顺序真实执行
一遍（前面有改写的话用改写后的命令），然后策略从这一步接着生成，测试打分。所有分支同时跑。给了 `hint` 时，它接在这一步之前那条
环境消息的末尾给分支里的策略看：属于 prompt，不训练（是 reroll agent 加进发给策略的请求里的，所以分支的 Harbor
`trajectory.json` 里看不到；`reroll_replay.json` 记了它的长度 `hint_chars`）。

- 分支在这一步之前的部分只做上下文、不训练；它用自己的分数训练，怎么算由 `credit` 决定（见下面的 [credit](#credit新轨迹怎么算分)）：
  `member` 当作组里多一个成员；`subgroup` 只和同一起点的兄弟比（只支持 GRPO）；`verified` 在 subgroup 的基础上看分支结果定原轨迹前缀
  的 credit（只支持 GRPO）。
- 一条轨迹可以有多处分支（对几个步骤分别调 `reroll`），每处最多 16 次。
- 原轨迹照常训练（它的 mask 和权重照常生效；前缀的改写只进重放，原轨迹保持原样）。
- 限制：不能在被改写的步骤上或之后分支；被上下文总结切开的轨迹不行（`traj.split`：先判断，对它调 reroll 会报错）；一个串联里只能有
  一个 transform 要求分支。第 0 步可以：它的分支什么都不重放，等于多一条从头开始的轨迹（分支模板的 `min_step` 和 `hint_reroll` 都不选它）。
- 带提示之后训练的是什么：分支从这一步起的 token，所在的上下文里有提示，而测试时策略拿不到提示。能迁移到没有提示的情况多少还是开放
  问题；把答案说出来的提示迁移得最少。`hint_reroll` 的 `prompt="hint_v3"` 比默认的 `hint_v2` 更严（只说看哪一部分，不点出里面的细节）；
  同一个 transform 不给提示就是对照。
- 运行需要分支的后处理：arm 用 `ota_train_transform_reroll`（它加了
  `--custom-reward-post-process-path prl.algo.added_rewards.group_rewards`）；快速开始遇到调用 `reroll` 的 transform 会自动选它，
  训练前的检查也会拦下没用它的作业。
- 结果：轨迹目录下每个分支一个 `reroll/s<步>_b<序号>/`（请求、结果、`reroll_replay.json`：重放的每一轮终端输出和原来是否一致，
  不一致的把两边的内容也记下来）；指标 `rollout/reroll/*`（`requested_trials`、`branches`、`{ok,failed,subgroup}_branches`、`verified_steps`、`confirmed_steps`（有分支比原轨迹分高；`subgroup_branches` 只数 `credit="subgroup"`）、
  `reward_mean`、`source_reward_mean`、`seconds_mean`）。分支自己的 trial 目录和其它 trial 并列在 `harbor_trials/` 下
  （没有 `transform.json`；分支的 `outcome.json` 里写着它）。每个结果有 `beat_trial`（分数比原轨迹高；`verified` 下就是它让原轨迹前缀
  清零），原轨迹的 `transform.json` 在分支跑完后重写，带 `reroll`（各分支结果）和 `reroll_beaten_steps`。
- 重放的终端输出是真实的，和原轨迹会有少量差别（时间、inode、git hash 这类）；框架记录、不拦截。

### 调模型

```python
k = await ask(prompt, check=fn)     # prompt 是字符串或 messages 列表
```

- 用运行配置的判官模型（`PRL_JUDGE_*`，写在站点配置里；全部设置见 `configs/arms/ota_judge.env`）。
  判官之外再要一个教师：配一组前缀不同的同名变量（例如 `PRL_TEACHER_*`），调用时写 `ask(..., prefix="PRL_TEACHER_")`。
- `check(answer)` 返回要用的值；抛错表示不合格，框架会把原因告诉模型再问一次。问不出来就抛 `Unanswered`，这条轨迹照原样训练。
- 重试、429 等待、并发、被内容过滤拒绝时依次换 `PRL_JUDGE_FALLBACK_MODELS` 里的模型，都由框架处理。
- 一次请求最多试几次（出错、超时、回答被拒各用掉一次）、时限和 `max_tokens`：`PRL_JUDGE_ATTEMPTS`（3）、
  `PRL_JUDGE_TIMEOUT`（600 秒）、`PRL_JUDGE_MAX_TOKENS`（16384）。
- 每次调用（请求、回答、每个被拒的回答和原因、用时、token 数、实际作答的模型）存到 `<trial_dir>/transform_calls.json`。

### 其他

| | |
|---|---|
| `traj.skip("原因")` | 这条做不了：照原样训练，记下原因。不算出错，但计入 `PRL_TRANSFORM_MAX_FAILURES`（默认连续 32 条没改动就停掉训练：模型一直连不上的情况）。有意不处理的轨迹（比如成功的）直接 `return traj` |
| `traj.record["key"] = 值` | 存进 `<trial_dir>/transform.json` |
| `traj.add_sample(sample, reward)`、`traj.demonstrate(step, action, reward)` | 高级：自己构造额外的训练样本 |

在 REPL 里看录好的轨迹（和函数拿到的一样）：`from prl.transforms.check import load_trajectories; trajs = load_trajectories("tokenizer")`
（快速开始第 2 步的 tokenizer 目录），然后看 `trajs[0].steps[1].command`、`trajs[0].steps[1].terminal`、`trajs[0].split` 等。

## 3. 分支模板

`TRANSFORM=branch`。所有"从中间重新 rollout"的方法都在回答三个问题，每个问题是一个可替换的插件：

```bash
TRANSFORM=branch PRL_TRANSFORM_ARGS='{"select": "first_bad", "allocate": {"name": "fixed", "n": 2, "view": "trial"}, "credit": "subgroup"}'
```

插件名先找内置的，再找 `transforms/<名字>.py` 里的同名函数（和 transform 一样）；带参数时写成 `{"name": ..., 参数...}`。
模板自己的参数：`min_step`（默认 1：从第 0 步分支等于重跑一条，丢掉）、`max_anchors`（默认 1：最多几处，按 select 给的顺序）。
选中的步骤记在 `traj.record["branch"]`。代码：`transforms/branch.py`。

### select：在哪分支

`select(traj, **参数) -> 步骤列表`（`Step` 或从 0 开始的步号，可以带分数写成 `(步, 分数)`，或写成 `{"step": ..., "hint": "..."}`
给这一步的分支一句提示；最好的在前）。select 里可以顺便改前面几步（只进重放）。模板的 `hint` 参数给所有分支同一句提示，select 给了的除外。

| 内置 | 含义 |
|---|---|
| `middle` | 中间那一步（测框架用） |
| `teacher` | 教师读轨迹，选一步，可以先小改前面几步（= `rewrite` 的 `added="reroll"`） |
| `first_bad` | 判官标的第一个坏步骤 |
| `surprisal` | 策略最"意外"的一步：这一步采样 token 的平均 −logprob 最高处（熵的近似）；参数 `top` |
| `entropy`、`post_tool_entropy`（之后） | 真正的 token 熵（FR3E）；看完工具输出后熵升高的步（ARPO）。需要推理端返回 top-k logprob |

### allocate：分几次

`allocate(traj, 锚点, **参数) -> [(步, 次数)]`。内置 `fixed`：每处 `n` 次；`only` 为 `"failed"`（默认，分数 0）、`"passed"`
或 `"all"`，在 select 之前检查，所以被排除的轨迹不花模型调用。`view` 决定什么时候、看着多少信息做决定：

| `view` | 什么时候决定 | 能做什么 | 代价 |
|---|---|---|---|
| `"trial"`（默认） | 这条轨迹一跑完 | 按它自己的结果：`fixed` 每处 n 次，可只给失败的 / 只给成功的 | 几乎不多花时间：分支和同组其他轨迹同时跑 |
| `"group"`（规划中，现在配了会报错） | 同组都跑完 | 按整组：`zero_signal_groups` 只给全对或全错的组；`by_score` 按 select 的分数分；每步总预算 | 要等组里最慢的一条，每步变长 |

为什么会想按组：GRPO 里全对或全错的组 advantage 全是 0，学不到东西。我们 9B 训练的前 20 步（640 组）里，全错 24%、全对 31%。
全错的组里一条成功的分支就是这组唯一的正信号。

### credit：新轨迹怎么算分

例：一组 8 条，2 条成功。失败的轨迹 τ 从第 5 步分支，分支成功了。

| `credit` | 做法 | 这个例子里 |
|---|---|---|
| `member`（默认） | 分支当作组里多一个成员，和所有人一起归一化 | 组均值 0.25 → 0.33；τ 的所有步（包括前 4 步）照样被整体压低；其他 7 条也被压低一点 |
| `subgroup` | 分支只和"同一起点出发的"比：τ 从第 5 步起的部分 + 同一处的其他分支。兄弟分数全一样时，τ 从这一步起的 advantage 是 0（只和替代走法比，不再按整条结局推） | τ 的第 5 步之后被压低，分支被推高；τ 的前 4 步保持原来的组 advantage；其他 7 条不受影响 |
| `verified` | 有分支比 τ 分高（证实这一步之前的状态还能救）时同 `subgroup`，并且 τ 这一步之前的 advantage 记 0（不背锅）；没有分支比 τ 高时 τ 整条照组 advantage | τ 的前 4 步不再被压低；第 5 步之后被压低，分支被推高。要是分支也失败：τ 照 GRPO 整体压低，分支 advantage 0 |
| `value`（之后） | 用同一起点所有续写的平均分估计这一步的价值，给原轨迹的步骤加权（VinePPO / FR3E 的思路） | — |

取舍：`member` 简单、已验证，但分支有"前缀白送"的起点，和从头跑的轨迹不完全可比；`subgroup` 比的是"同一状态、不同走法"，
信号正好落在分开的那一步，但每处只分 1 次时只有 2 条可比，噪声大，建议 n ≥ 2。分支只支持 GRPO 和 PPO（REINFORCE++ 不行），
`subgroup` 只支持 GRPO（PPO 的 advantage 来自 critic）。

### 现有方法怎么写成模板

| 方法 | select | allocate | credit |
|---|---|---|---|
| 教师改完继续 roll | `teacher` | `fixed`, n=1, 只给失败的 | `member` |
| FR3E | `entropy`, top-K | 每处 M 次 | `value` |
| ARPO | `post_tool_entropy` | 熵升过阈值时 Z 次 | `subgroup` |
| 判官 + 分支 | `first_bad` | `fixed`, n=2 | `subgroup`，原轨迹的坏步骤同时打负权重 |
| 提示分支（`transforms/hint_reroll.py`） | 判官找第一个走错的步骤，写一句不超过 15 词的提示：只说回头看哪里，不说错在哪、不给命令 | n=2，只给失败的 | `verified`，提示接在分叉点之前的消息末尾，不训练 |

## 4. 框架替你做的事

| | |
|---|---|
| 开跑前检查 | 训练作业开头在 8 条录好的轨迹上跑你的函数（包括模型调用）；报错、返回非法结果、卡住、全部跳过，作业在训练前停下 |
| 失败兜底 | 某条轨迹上失败 → 照原样训练并记录原因；连续 32 条失败 → 训练停下，避免悄悄退化成原生 GRPO |
| 一致性检查 | 每个样本过 Miles 的校验（token、loss mask、logprob、长度）；分数只在你写了 `traj.reward` 时变；没有可训练 token 的样本丢掉 |
| 记录 | 每条轨迹 `transform.json`：每个函数的状态、用时、参数（含默认值）、改了哪些步（`weighted_steps` {步骤: 新权重}、`masked_steps`、`token_weighted_steps`、`rewritten_steps`、`deleted_steps`、`reroll_steps`）、分数前后、可训练 token 前后；训练日志里有对应的 `rollout/*` 指标 |
| 框架烟测 | 框架代码变了以后要跑一次（两次更新，约半小时），生成收据；训练作业自动找匹配当前代码的收据 |

### 离线检查一个函数

`python -m prl.transforms.check`（README ②）会在 8 条录好的轨迹上运行选定的函数：

- 终端里每条轨迹一行：`[transform check] <轨迹>: reward <分数>, <步数> steps, <状态>, steps changed: <如 5 weighted, 2 masked>, trained tokens <改前> -> <改后>, <秒数> s`，
  状态是 `ok` 或 `skipped`（skipped 的行末尾写着原因）；框架自己的 `[prl]` 日志行和问判官时的 HTTP 请求也会打印出来。最后一行是 `passed` 为通过；出错时打印 `FAILED`、出错的轨迹和原因。
- 输出目录（请用空目录）里每条轨迹一个子目录：`transform.json` 是框架的记录（每个函数的状态和用时、你写进 `traj.record` 的内容、
  改了哪些步骤，如 `weighted_steps` {步骤: 新权重}、可训练 token 改前 / 改后的数量），`transform_calls.json` 是每次 `ask` 的请求和回答（ATLAS 判官的在 `atlas_labels.json`）；`summary.json` 汇总全部 8 条。
- 步骤列表从 0 数，和 `step.index` 一样；判官自己的编号（ATLAS 的 `critical`、`good` / `bad`、`atlas_labels.json`；`hint_reroll` 的 `step`）和它们的 prompt 一样从 1 数。
- 录好的轨迹没有 rollout log prob（`surprisal` 看到的都是 0），也没有测试输出（`hint_reroll` 的判官看不到）：这类选择只在训练里看得出来。用自己的轨迹：`scripts/tools/make_transform_fixture.py` 从一次运行的 `traces/rollout_data/<update>.pt` 做 fixture，`PRL_TRANSFORM_FIXTURE=<它>` 就在它上面检查。
- reroll 在这里只记录请求（`transform.json` 的 `reroll`：从哪些步分支、要重放哪些轮），分支在训练时才真正跑；训练时快速开始会自动选 reroll 需要的 arm。
- 想按快速开始的默认设置来检查：`PRL_TRANSFORM=atlas_signed PRL_TRANSFORM_ARGS='{"failed_good_weight": 0.0}'`。训练 arm 可能比站点配置多设一些判官参数（signed ATLAS：`PRL_JUDGE_EXTRA_BODY='{"reasoning_effort": "low"}'`），检查时请一并设上；不在集群上运行时，还要 `export PRL_JUDGE_KEY_FILE=<本机的 key 文件>`。

## 5. 配置

作业里的设置写在站点配置（`configs/sites/<site>.env`）或 arm 里：作业看不到提交它的 shell 里的环境变量，
`TRANSFORM` 和 `PRL_TRANSFORM_ARGS` 除外。

| 环境变量 | 默认 | |
|---|---|---|
| `PRL_TRANSFORM` | 无 | 名字，或用 `+`（或逗号）串联 |
| `PRL_TRANSFORM_ARGS` | 无 | JSON；串联时按名字分开 |
| `PRL_TRANSFORM_TIMEOUT` | 1800 | 一条轨迹上的秒数上限 |
| `PRL_TRANSFORM_MAX_FAILURES` | 32 | 连续失败多少条停训练（0 不停） |
| `PRL_REROLL_TIMEOUT` | 0（不限） | 一次分支的秒数上限，超时就不加这条 |
| `PRL_REROLL_CONCURRENCY` | 32 | 每个 rollout worker 同时跑的分支数 |
| `PRL_JUDGE_*`、`PRL_JUDGE_FALLBACK_MODELS` | 见判官配置 | `ask` 用的模型、key 文件、并发、备选模型 |

## 6. 现成的 transform

| 名字 | 做什么 | 主要参数 |
|---|---|---|
| `atlas` | 判官选关键步，其余 mask（或乘 `other_weight`） | `other_weight`；`select="random"`（不用判官：随机选占 `keep_tokens` 比例训练 token 的步骤，做判官选择的对照；按 `seed` 和轨迹目录的名字抽） |
| `atlas_signed` | 判官分好 / 坏 / 一般；成功轨迹的坏步反着推 | `mode`（mild / strict）、`bad_weight`、`failed_good_weight`、`quote_bad`（只推出错的那段） |
| `rewrite` | 教师小改轨迹；或只修失败的；或选点分支 | `prompt`、`mask_new`、`successes` / `failures`、`added="reroll"`、各种改动比例上限 |
| `branch` | 分支模板（§3） | `select`、`allocate`、`credit`、`min_step`、`max_anchors` |
| `hint_reroll` | 判官找失败轨迹第一个走错的步骤（记录里从 1 数，和它的 prompt 一致），写一句不超过 `max_hint_words` 词的提示（只说回头看哪里，不给命令）；策略带着提示从那里再试 `n` 次 | `n`（2）、`credit`（`verified`）、`prompt`（`hint_v2`；`hint_v3` 更严：只说看哪一部分；`hint_v1` 还会点出错在哪）、`max_hint_words`（15）、`test_chars`（判官能看到的测试输出末尾，1500） |
| `reroll_smoke` | 测框架用：从中间一步分支 | `min_steps`、`edit`、`every` |

参数的完整说明在各文件开头的 docstring。
