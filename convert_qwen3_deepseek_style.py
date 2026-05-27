"""Convert Qwen3 BF16 HF checkpoints to DeepSeek-style FP4/FP8 layouts.

This intentionally serializes FP4 weights as int8 packed bytes, matching the
DeepSeek-V4 HF checkpoint convention. A runtime can then view those tensors as
``torch.float4_e2m1fn_x2``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from dataclasses import dataclass, field

import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
from tqdm import tqdm


FP4_MAX = 6.0
FP8_MAX = 448.0
FP4_BLOCK_K = 32
FP8_BLOCK_SHAPE = (128, 128)

MLP_WEIGHT_SUFFIXES = (
    ".mlp.down_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
)
ATTN_WEIGHT_SUFFIXES = (
    ".self_attn.k_proj.weight",
    ".self_attn.o_proj.weight",
    ".self_attn.q_proj.weight",
    ".self_attn.v_proj.weight",
)


def ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def module_name(weight_name: str) -> str:
    return weight_name[: -len(".weight")] if weight_name.endswith(".weight") else weight_name


def round_power2_scale(amax: torch.Tensor, *, max_value: float, min_amax: float) -> torch.Tensor:
    """DeepSeek-style power-of-two scale: 2 ** ceil(log2(max(amax, floor) / max))."""
    safe_amax = torch.clamp(amax.to(torch.float32), min=min_amax)
    exp = torch.ceil(torch.log2(safe_amax / max_value))
    return torch.pow(torch.tensor(2.0, dtype=torch.float32, device=amax.device), exp)


def pack_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Quantize clipped values to E2M1 nibbles and pack low nibble first."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"FP4 packing requires an even last dimension, got {x.shape[-1]}.")

    x = x.to(torch.float32)
    result = torch.zeros_like(x, dtype=torch.uint8)

    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    result[(x >= -0.25) & (x < -0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    return (result[..., ::2] + result[..., 1::2] * 16).contiguous()


def classify_qwen_weight(name: str, mode: str) -> str | None:
    """Return ``fp4``, ``fp8``, or ``None`` for one Qwen HF weight name."""
    if not name.endswith(".weight"):
        return None
    is_mlp = name.endswith(MLP_WEIGHT_SUFFIXES)
    is_attn = name.endswith(ATTN_WEIGHT_SUFFIXES)

    if mode == "fp4_only":
        return "fp4" if (is_mlp or is_attn) else None
    if mode == "mixed":
        if is_mlp:
            return "fp4"
        if is_attn:
            return "fp8"
        return None
    raise ValueError(f"Unknown mode {mode!r}; expected 'fp4_only' or 'mixed'.")


def quantize_fp4_deepseek(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D BF16/FP32 weight to packed E2M1 bytes plus E8M0 scales."""
    if weight.dim() != 2:
        raise ValueError(f"FP4 quantization expects a 2D weight, got shape {tuple(weight.shape)}.")
    out_features, in_features = weight.shape
    if in_features % FP4_BLOCK_K != 0:
        raise ValueError(f"FP4 requires K divisible by {FP4_BLOCK_K}, got {in_features}.")
    if in_features % 2 != 0:
        raise ValueError(f"FP4 packing requires even K, got {in_features}.")

    w = weight.to(torch.float32).contiguous()
    blocks = w.view(out_features, in_features // FP4_BLOCK_K, FP4_BLOCK_K)
    amax = blocks.abs().amax(dim=-1)
    scale_f32 = round_power2_scale(amax, max_value=FP4_MAX, min_amax=FP4_MAX * (2.0**-126))
    scaled = (blocks / scale_f32.unsqueeze(-1)).clamp(min=-FP4_MAX, max=FP4_MAX)
    packed_u8 = pack_fp4_e2m1(scaled.reshape(out_features, in_features))
    return packed_u8.view(torch.int8).contiguous(), scale_f32.to(torch.float8_e8m0fnu).contiguous()


def quantize_fp8_deepseek(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to E4M3 plus 128x128 E8M0 block scales."""
    if weight.dim() != 2:
        raise ValueError(f"FP8 quantization expects a 2D weight, got shape {tuple(weight.shape)}.")
    block_m, block_k = FP8_BLOCK_SHAPE
    out_features, in_features = weight.shape
    out_tiles = ceildiv(out_features, block_m)
    in_tiles = ceildiv(in_features, block_k)

    w = weight.to(torch.float32).contiguous()
    padded = F.pad(
        w,
        (0, in_tiles * block_k - in_features, 0, out_tiles * block_m - out_features),
        mode="constant",
        value=0.0,
    )
    blocks = padded.view(out_tiles, block_m, in_tiles, block_k)
    block_amax = blocks.abs().amax(dim=1).amax(dim=2)
    scale_f32 = round_power2_scale(block_amax, max_value=FP8_MAX, min_amax=1e-4)

    scaled = (blocks / scale_f32[:, None, :, None]).clamp(min=-FP8_MAX, max=FP8_MAX)
    q_padded = scaled.reshape(out_tiles * block_m, in_tiles * block_k).to(torch.float8_e4m3fn)
    qweight = q_padded[:out_features, :in_features].contiguous()
    return qweight, scale_f32.to(torch.float8_e8m0fnu).contiguous()


@dataclass
class ConversionResult:
    weight_map: dict[str, str] = field(default_factory=dict)
    total_size: int = 0
    fp4_modules: set[str] = field(default_factory=set)
    fp8_modules: set[str] = field(default_factory=set)
    modules_to_not_convert: set[str] = field(default_factory=set)

    def add_tensor(self, filename: str, name: str, tensor: torch.Tensor) -> None:
        self.weight_map[name] = filename
        self.total_size += tensor.numel() * tensor.element_size()


def copy_sidecar_files(input_path: str, output_path: str) -> None:
    for filename in os.listdir(input_path):
        src = os.path.join(input_path, filename)
        dst = os.path.join(output_path, filename)
        if filename.endswith(".safetensors") or filename == "model.safetensors.index.json" or os.path.isdir(src):
            continue
        shutil.copyfile(src, dst)


def process_shard(input_path: str, output_path: str, filename: str, mode: str, result: ConversionResult) -> None:
    q_weights: dict[str, torch.Tensor] = {}
    print(f"[{mode}] reading {filename}", flush=True)
    with safetensors.safe_open(os.path.join(input_path, filename), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            quant_kind = classify_qwen_weight(key, mode)
            if quant_kind == "fp4":
                print(f"[{mode}] {filename}: FP4 {key} shape={tuple(tensor.shape)}", flush=True)
                qweight, scale = quantize_fp4_deepseek(tensor)
                q_weights[key] = qweight
                q_weights[key.replace(".weight", ".scale")] = scale
                result.fp4_modules.add(module_name(key))
            elif quant_kind == "fp8":
                print(f"[{mode}] {filename}: FP8 {key} shape={tuple(tensor.shape)}", flush=True)
                qweight, scale = quantize_fp8_deepseek(tensor)
                q_weights[key] = qweight
                q_weights[key.replace(".weight", ".scale")] = scale
                result.fp8_modules.add(module_name(key))
            else:
                q_weights[key] = tensor
                if key.endswith(".weight"):
                    result.modules_to_not_convert.add(module_name(key))

    print(f"[{mode}] writing {filename}", flush=True)
    safetensors.torch.save_file(q_weights, os.path.join(output_path, filename), metadata={"format": "pt"})
    for key, tensor in q_weights.items():
        result.add_tensor(filename, key, tensor)


def build_quantization_config(mode: str, result: ConversionResult) -> dict:
    fp4_info = {
        "fp4_weight_dtype": "float4_e2m1fn_x2",
        "fp4_storage_dtype": "int8",
        "fp4_scale_dtype": "float8_e8m0fnu",
        "fp4_block_size": [1, FP4_BLOCK_K],
        "fp4_modules": sorted(result.fp4_modules),
    }
    if mode == "fp4_only":
        return {
            "quant_method": "mxfp4",
            "fmt": "e2m1",
            "scale_fmt": "ue8m0",
            "weight_block_size": [1, FP4_BLOCK_K],
            "activation_scheme": "dynamic",
            **fp4_info,
            "modules_to_not_convert": sorted(result.modules_to_not_convert),
            "qwen_dense_mapping_note": (
                "All eligible Qwen3 dense Linear weights are stored as DeepSeek-style "
                "packed FP4 bytes (safetensors I8) with one E8M0 scale per 32 K values."
            ),
        }
    if mode == "mixed":
        return {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": list(FP8_BLOCK_SHAPE),
            "mixed_precision": {
                **fp4_info,
                "fp8_weight_dtype": "float8_e4m3fn",
                "fp8_scale_dtype": "float8_e8m0fnu",
                "fp8_modules": sorted(result.fp8_modules),
                "qwen_dense_mapping_note": (
                    "Qwen3-4B has no MoE experts; dense MLP projections are used as "
                    "expert-like FP4 weights and attention projections remain FP8."
                ),
            },
            "modules_to_not_convert": sorted(result.modules_to_not_convert),
        }
    raise ValueError(f"Unknown mode {mode!r}.")


def write_index_and_config(input_path: str, output_path: str, mode: str, result: ConversionResult) -> None:
    with open(os.path.join(input_path, "config.json")) as f:
        cfg = json.load(f)
    cfg["expert_dtype"] = "fp4"
    cfg["quantization_config"] = build_quantization_config(mode, result)
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    index_dict = {
        "weight_map": result.weight_map,
        "metadata": {"total_size": result.total_size},
    }
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index_dict, f, indent=2)


def convert_model(model_dir: str, save_dir: str, mode: str) -> None:
    input_path = os.path.abspath(model_dir)
    output_path = os.path.abspath(save_dir)
    if mode not in {"fp4_only", "mixed"}:
        raise ValueError("--mode must be either fp4_only or mixed.")
    if not os.path.exists(os.path.join(input_path, "config.json")):
        raise FileNotFoundError(f"Missing config.json in {input_path}.")
    os.makedirs(output_path, exist_ok=True)

    copy_sidecar_files(input_path, output_path)
    result = ConversionResult()
    shards = sorted(f for f in os.listdir(input_path) if f.endswith(".safetensors"))
    for filename in tqdm(shards, desc=f"Converting {mode}"):
        process_shard(input_path, output_path, filename, mode, result)
        gc.collect()

    write_index_and_config(input_path, output_path, mode, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Source BF16 Qwen3 HF checkpoint directory.")
    parser.add_argument("--save-dir", required=True, help="Output checkpoint directory.")
    parser.add_argument("--mode", required=True, choices=["fp4_only", "mixed"])
    args = parser.parse_args()
    convert_model(args.model_dir, args.save_dir, args.mode)


if __name__ == "__main__":
    main()
