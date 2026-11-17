"""Worst-case batch for a trainer memory check: a Miles rollout dump whose longest sample is grown to MAX_TOKENS.

  python3 scripts/tools/long_sample_batch.py <rollout_data/N.pt> <out.pt> <MAX_TOKENS> <rollout_id>

The response is repeated (tokens, loss mask, rollout log probs) until the sample has MAX_TOKENS tokens, all trained;
only the memory use of the step means anything (the replay case longest_sample, scripts/chain/replay_check.sh).
"""

import sys

import torch

src, out, max_tokens, rollout_id = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
dump = torch.load(src, weights_only=False)
s = max(dump["samples"], key=lambda x: len(x["tokens"]))
prompt_len = len(s["tokens"]) - s["response_length"]
response, logp = s["tokens"][prompt_len:], s["rollout_log_probs"]
n = max_tokens - prompt_len
s["tokens"] = s["tokens"][:prompt_len] + (response * (n // len(response) + 1))[:n]
s["rollout_log_probs"] = (logp * (n // len(logp) + 1))[:n]
s["loss_mask"] = [1] * n
s["response_length"] = n
dump["rollout_id"] = rollout_id
torch.save(dump, out)
print(f"{src} -> {out}: longest sample {prompt_len} + {n} = {len(s['tokens'])} tokens")
