# Orbit upstream seams

Orbit depends on several upstream projects, but this branch owns only the
adapter code at these boundaries. Treat upstream implementations as external
contracts: pin compatible revisions, validate inputs at the boundary, and test
observable behavior rather than private file layouts.

## Megatron Core distributed checkpoints

- Tensor metadata and object metadata are different contracts. Any path that
  requests optimizer or other sharded objects must inspect the full sharded
  metadata; tensor-only metadata is sufficient only for a tensor-only request.
- Load common state through the Megatron Core loader. Tests must verify the
  returned state, not require a particular physical file such as `common.pt`.
- Direct conversion derives its keys and shard geometry from the destination
  model's sharded state dictionary. A dependency update must rerun the
  conversion geometry and checkpoint round-trip tests.
- A direct converter publishes one complete checkpoint directory atomically.
  The destination is required to be absent, and tokenizer assets are part of
  the same transaction.

## PEFT and Megatron module contracts

- Target matching may use exact names or patterns. Safety decisions must use
  the actual matched module path, not only the pattern text that selected it.
- Wrapped linears and grouped-expert linears have different weight, bias, and
  output contracts. OFT code must preserve the disabled-path result and the
  enabled-path bias semantics for the concrete module type.
- Adapter merge is a structural mutation. Validate every target, alias, shape,
  dtype, and COFT projection before changing any base parameter.

## Quantized checkpoint schemas

- INT4, NVFP4, and FP8 are separate on-disk schemas. Format detection, scale
  dtype, scale finiteness/positivity, packing geometry, and source-to-target
  coverage must all pass before a checkpoint is saved.
- Tensor-parallel and expert-tensor-parallel axes are not interchangeable.
  Tests use distinct sentinel groups so an accidental group substitution cannot
  pass when both groups happen to have the same size.
- ModelOpt conversion must select only modules represented by the source
  checkpoint. A global quantization rule can create destination state for
  modules that have no source scales.

## Hugging Face, Transformers, and SGLang

- HF/SGLang export maps only explicitly supported parameter names. A bare
  `Parameter` mapping is rejected when its orientation or grouped-expert
  semantics cannot be established from the module contract.
- Weight-only INT4 export writes a complete Transformers compressed-tensors
  configuration. Its target list is the exact sorted set of modules converted
  in that run; inherited or unrelated quantization metadata is replaced.
- Routed-expert selection matches complete dotted path segments, including a
  numeric expert id. Substring matches are not a supported selector contract.

## Updating a dependency

When changing a gitlink or locked package version:

1. identify which seams above changed;
2. run the focused metadata, geometry, export, merge, and distributed failure
   tests for those seams;
3. run at least one small checkpoint round trip with the new dependency set;
4. record any newly required boundary adaptation in Orbit rather than patching
   code under `3rdparty/`.
