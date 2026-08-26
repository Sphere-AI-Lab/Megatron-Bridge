from __future__ import annotations

import importlib
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOTS = (
    REPOSITORY_ROOT / "src",
    REPOSITORY_ROOT / "scripts",
    REPOSITORY_ROOT / "examples",
    REPOSITORY_ROOT / "tutorials",
)
TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
PREVIOUS_CODEBASE_NAME = "sph" + "ere"
LEGACY_ORBIT_MODULES = (
    "megatron.bridge.models.conversion.low_precision",
    "megatron.bridge.models.conversion.low_precision.common",
    "megatron.bridge.models.conversion.low_precision.fp4",
    "megatron.bridge.models.conversion.low_precision.fp8",
    "megatron.bridge.models.conversion.low_precision.int4",
    "megatron.bridge.models.conversion.low_precision.int8",
    "megatron.bridge.models.conversion.low_precision.nvfp4",
    "megatron.bridge.models.deepseek.deepseek_v3_int4_bridge",
    "megatron.bridge.models.deepseek.deepseek_v4_bridge",
    "megatron.bridge.models.kimi_vl.kimi_k25_vl_nvfp4_bridge",
    "megatron.bridge.models.kimi_vl.utils",
    "megatron.bridge.models.llama.llama_int4_bridge",
    "megatron.bridge.models.qwen.qwen3_int4_bridge",
    "megatron.bridge.models.qwen.qwen3_moe_fp8_bridge",
    "megatron.bridge.models.qwen.qwen3_nvfp4_direct",
    "megatron.bridge.peft.canonical_oft",
    "megatron.bridge.peft.fp8_utils",
    "megatron.bridge.peft.int4_utils",
    "megatron.bridge.peft.nvfp4_utils",
    "megatron.bridge.peft.oft",
    "megatron.bridge.peft.oft_layers",
    "megatron.bridge.peft.param_names",
    "megatron.bridge.peft.qwen3_fp8_gemm",
    "megatron.bridge.utils.model_metadata_compare",
)
CONSUMED_ORBIT_SYMBOLS = {
    "megatron.bridge.orbit.low_precision.common": (
        "TensorSpillManager",
        "build_single_rank_meta_provider",
        "patch_meta_init_for_te_modules",
    ),
    "megatron.bridge.orbit.low_precision.fp8": (
        "apply_modelopt_fp8_to_meta_model",
        "build_fp8_direct_model_state_dict",
    ),
    "megatron.bridge.orbit.low_precision.int4": (
        "build_int4_direct_model_state_dict",
        "dequantize_int4",
        "quantize_to_int4",
        "register_int4_buffers_after_load_dense",
        "transform_sharded_state_dict_for_int4_dense",
    ),
    "megatron.bridge.orbit.low_precision.nvfp4": (
        "apply_modelopt_nvfp4_to_meta_model",
        "build_nvfp4_direct_model_state_dict",
        "collect_nvfp4_target_module_names",
        "is_nvfp4_source",
        "register_nvfp4_buffers_after_load_dense",
        "transform_sharded_state_dict_for_nvfp4_dense",
    ),
    "megatron.bridge.orbit.model_bridges.deepseek_v3_int4_bridge": ("DeepSeekV3INT4Bridge",),
    "megatron.bridge.orbit.model_bridges.deepseek_v4_bridge": ("DSV4OFT", "DeepSeekV4Bridge"),
    "megatron.bridge.orbit.model_bridges.llama_int4_bridge": ("LlamaINT4Bridge",),
    "megatron.bridge.orbit.model_bridges.qwen3_int4_bridge": ("Qwen3INT4Bridge", "Qwen3MoEINT4Bridge"),
    "megatron.bridge.orbit.oft.canonical_oft": ("CanonicalOFT",),
    "megatron.bridge.orbit.oft.oft": ("OFT",),
    "megatron.bridge.orbit.oft.oft_layers": ("OFTVocabParallelEmbedding",),
    "megatron.bridge.orbit.oft.param_names": (
        "CANONICAL_OFT_SLICE_NAMES",
        "is_peft_adapter_param_name",
    ),
    "megatron.bridge.orbit.quant.fp8_utils": (
        "merge_gated_mlp_scale_inv",
        "merge_qkv_scale_inv",
        "register_fp8_scale_inv_buffers_after_load",
        "transform_sharded_state_dict_for_fp8",
    ),
    "megatron.bridge.orbit.quant.int4_utils": (
        "register_int4_buffers_after_load",
        "transform_sharded_state_dict_for_int4",
    ),
    "megatron.bridge.orbit.quant.nvfp4_utils": (
        "register_nvfp4_buffers_after_load",
        "transform_sharded_state_dict_for_nvfp4",
    ),
}


def test_orbit_directories_replace_previous_codebase_directories():
    bridge_package = REPOSITORY_ROOT / "src" / "megatron" / "bridge"

    assert (bridge_package / "orbit").is_dir()
    assert not (bridge_package / PREVIOUS_CODEBASE_NAME).exists()
    assert (REPOSITORY_ROOT / "scripts" / "orbit").is_dir()
    assert not (REPOSITORY_ROOT / "scripts" / PREVIOUS_CODEBASE_NAME).exists()


def test_no_previous_codebase_name_remains_in_orbit_source_surface():
    stale_paths = []
    for root in SEARCH_ROOTS:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                if PREVIOUS_CODEBASE_NAME in path.read_text(errors="ignore").casefold():
                    stale_paths.append(path.relative_to(REPOSITORY_ROOT).as_posix())

    assert stale_paths == []


@pytest.mark.parametrize("module_name", LEGACY_ORBIT_MODULES)
def test_legacy_orbit_module_is_unavailable(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError) as exc_info:
        importlib.import_module(module_name)

    missing_module = exc_info.value.name
    assert missing_module is not None
    assert module_name == missing_module or module_name.startswith(f"{missing_module}.")


@pytest.mark.parametrize(("module_name", "symbol_names"), CONSUMED_ORBIT_SYMBOLS.items())
def test_orbit_module_exports_consumed_symbols(module_name: str, symbol_names: tuple[str, ...]) -> None:
    module = importlib.import_module(module_name)

    missing = [name for name in symbol_names if not hasattr(module, name)]
    assert not missing, f"{module_name} is missing expected symbols: {', '.join(missing)}"
