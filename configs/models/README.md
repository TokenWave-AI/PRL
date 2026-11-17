# Model profiles

`MODEL=<name>` selects `configs/models/<name>.env`. Without `MODEL`, the profile is `qwen3.5-$MODEL_SIZE`, so runs that
predate `MODEL` keep their behaviour. `scripts/lib/env.sh` sources the profile on the host and
`scripts/lib/container_main.sh` sources it again inside the container (`scripts/run.sh` passes `MODEL` and
`MODEL_HF_DIR` through `--cleanenv`).

| Variable | Used for |
|---|---|
| `MODEL_HF_REPO` | HF id to download (`hf download $MODEL_HF_REPO`) |
| `MODEL_HF_DIR` | `--hf-checkpoint` (SGLang weights, tokenizer, chat template). Override with `MODEL_HF_DIR=<dir>`, e.g. a verified export from `scripts/chain/ckpt_to_hf.sh` |
| `MODEL_ARGS_NAME` | Megatron model args: Miles `scripts/models/$MODEL_ARGS_NAME.py` |
| `MODEL_TORCH_DIST` | `--ref-load` of training; `scripts/setup/convert_ckpt.sh` writes it |
| `TITO_MODEL` | `--tito-model` (Miles `TITOTokenizerType`) |
| `SGLANG_TOOL_CALL_PARSER` | must equal that TITO tokenizer's `tool_call_parser` (Miles does not apply it itself) |
| `SGLANG_REASONING_PARSER` | `--sglang-reasoning-parser` |
| `MODEL_MAX_CONTEXT` | `max_position_embeddings`; launchers refuse a larger `MAX_CONTEXT` |

`tests/unit/test_model_profiles.py` checks every profile against the vendored Miles tree.
