# Orbit conversion and finetuning

Orbit contains the low-precision checkpoint conversion and OFT/QOFT workflows
added by this branch. Run the conversion examples from the repository root.
The shell finetuning launchers resolve the repository root themselves and can
run from any directory.

## Workflow

1. Start with a Hugging Face checkpoint in BF16, FP8, INT4, or NVFP4.
2. For a BF16 model that needs weight-only INT4, quantize its routed-expert MLP
   weights:

   ```bash
   bash scripts/orbit/quantize_to_int4.sh /models/source-bf16 /models/source-int4 32
   ```

3. Convert the HF checkpoint directly to Megatron distributed-checkpoint
   format. Choose the converter matching the source format:

   ```bash
   uv run --project . python scripts/orbit/conversion/convert_fp8_checkpoint_direct.py \
       --hf-model-path /models/source-fp8 \
       --megatron-path /checkpoints/source-fp8-mcore

   uv run --project . python scripts/orbit/conversion/convert_int4_checkpoint_direct.py \
       --hf-model-path /models/source-int4 \
       --megatron-path /checkpoints/source-int4-mcore

   uv run --project . python scripts/orbit/conversion/convert_nvfp4_checkpoint_direct.py \
       --hf-model-path /models/source-nvfp4 \
       --megatron-path /checkpoints/source-nvfp4-mcore
   ```

   The destination must not already exist. A converter writes the checkpoint
   and tokenizer assets to a private sibling directory and publishes the whole
   directory only after validation succeeds. Failed conversions remove their
   staging directory rather than exposing a partial checkpoint.

4. Run PEFT on a regular or supported quantized checkpoint:

   ```bash
   PEFT=oft QUANT=nvfp4 HF_MODEL=/models/source-nvfp4 \
   MEGATRON_CKPT=/checkpoints/source-nvfp4-mcore NUM_GPUS=8 \
   bash scripts/orbit/run_peft_finetune.sh -- --train-iters 100
   ```

   INT4 and the specialized quantized-base paths use the QOFT launcher:

   ```bash
   QUANT=int4 HF_MODEL_PATH=/models/source-int4 \
   MEGATRON_CKPT=/checkpoints/source-int4-mcore NUM_GPUS=8 \
   bash scripts/orbit/run_qoft_finetune.sh -- --train-iters 100
   ```

   The QOFT entrypoint currently supports this architecture/format matrix:

   | Hugging Face architecture | Supported quantized base formats |
   | --- | --- |
   | `Qwen3ForCausalLM` | FP8 |
   | `Qwen3MoeForCausalLM` | FP8, INT4, NVFP4 |
   | `KimiK25ForConditionalGeneration` | INT4, NVFP4 |
   | `DeepseekV3ForCausalLM` (Moonlight) | INT4 |

   Other architecture/format pairs are rejected during preflight rather than
   falling through to a partially compatible load path. The generic PEFT
   entrypoint supports unquantized bases and its own `fp8`, `mxfp8`, and
   `nvfp4` base-quantization modes through `AutoBridge`; that is a separate
   contract from the specialized QOFT matrix above.

Both launchers execute `uv run --project <repo> python -m
torch.distributed.run`, so training uses the repository's locked environment.
`NUM_GPUS` is the positive number of local processes. Put extra entrypoint
arguments after `--`; the legacy `EXTRA_ARGS` string is rejected because a
string cannot preserve shell argument boundaries safely.

## Validation before a long run

Use `--skip-train` with `finetune_qoft.py` to exercise architecture detection,
checkpoint loading, and adapter construction without training. Match the
parallelism and INT4 group size to the converted checkpoint. Mismatched
geometry is rejected during checkpoint loading or direct-conversion preflight.

The direct converters validate source format, mapping coverage, tensor
geometry, dtype, scale values, and checkpoint completeness. A validation error
means the input model or selected converter does not satisfy the supported
contract; do not bypass it by copying a partially written staging directory.

## Dependency contracts

Orbit deliberately isolates assumptions about Megatron Core, PEFT, ModelOpt,
Transformers, and HF/SGLang checkpoint schemas. See
[`docs/orbit/UPSTREAM-SEAMS.md`](../../docs/orbit/UPSTREAM-SEAMS.md) before
changing a pinned dependency or adding a model family.
