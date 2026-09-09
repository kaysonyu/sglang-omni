#!/usr/bin/env python3
"""Convert a mossLite logical checkpoint into SGLang-Omni's Local layout.

This is deliberately a serving-target adapter, not a second general-purpose
MOSS checkpoint converter.  mossLite remains authoritative for logical DCP
selection and tensor metadata; this script changes only the published names
and runtime assets consumed by SGLang-Omni.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

PROCESSOR_RUNTIME_FILES = (
    "__init__.py",
    "configuration_moss_tts.py",
    "processing_moss_tts.py",
    "processor_config.json",
)
TOKENIZER_FILES = (
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


def _directory(path: Path, description: str) -> Path:
    resolved = path.expanduser().absolute()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{description} must be a real directory: {path}")
    return resolved


def _json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{description} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mosslite_api(repository: Path):
    repository = _directory(repository, "mossLite repository")
    if not (repository / "toolkits/model_ckpt_convertor/common.py").is_file():
        raise ValueError(
            f"mossLite repository has no checkpoint converter API: {repository}"
        )
    sys.path.insert(0, str(repository))
    from toolkits.model_ckpt_convertor.common import (  # type: ignore[import-not-found]
        NativeTensorReader,
        StreamingSafeTensorWriter,
        publish_staging_directory,
    )
    from toolkits.model_ckpt_convertor.moss_tts_local.convert import (  # type: ignore[import-not-found]
        topology_from_identity,
    )

    return (
        NativeTensorReader,
        StreamingSafeTensorWriter,
        publish_staging_directory,
        topology_from_identity,
    )


def _validate_topology(topology: Any, identity: dict[str, Any]) -> None:
    expected = {
        "num_layers": 36,
        "hidden_size": 2560,
        "num_attention_heads": 32,
        "num_query_groups": 8,
        "head_dim": 128,
        "ffn_hidden_size": 9728,
        "text_vocab_size": 151936,
        "max_position_embeddings": 40960,
        "n_vq": 12,
        "speech_vocab_size": 1024,
        "local_num_layers": 1,
        "local_hidden_size": 2560,
        "local_num_attention_heads": 32,
        "local_ffn_hidden_size": 9728,
        "audio_end_token_id": 151653,
    }
    mismatches = {
        name: (value, getattr(topology, name, None))
        for name, value in expected.items()
        if getattr(topology, name, None) != value
    }
    semantics = identity.get("forward_semantics")
    schema = identity.get("tensor_schema")
    if not isinstance(semantics, dict) or not isinstance(schema, dict):
        raise ValueError("mossLite checkpoint has no complete model identity.")
    exact_identity = {
        "architecture": identity.get("architecture"),
        "embedding_head_usage": semantics.get("embedding_head_usage"),
        "embedding_head_storage": schema.get("embedding_head_storage"),
        "local_projection_topology": schema.get("local_projection_topology"),
    }
    expected_identity = {
        "architecture": "MOSS-TTS-Local",
        "embedding_head_usage": "untied",
        "embedding_head_storage": "split_v1",
        "local_projection_topology": (
            "global_hidden_plus_teacher_forced_rvq_embeddings_v1"
        ),
    }
    mismatches.update(
        {
            name: (value, exact_identity.get(name))
            for name, value in expected_identity.items()
            if exact_identity.get(name) != value
        }
    )
    if mismatches:
        raise ValueError(
            f"Unsupported MOSS-TTS Local checkpoint identity: {mismatches}"
        )


def _export_weights(
    reader: Any,
    topology: Any,
    output: Path,
    writer_type: Any,
    max_shard_size: str,
) -> set[str]:
    writer = writer_type(output, max_shard_size)
    written: set[str] = set()

    def add(name: str, value: torch.Tensor) -> None:
        writer.add(name, value)
        written.add(name)

    text = reader.read("text_embedding.word_embeddings.weight")
    add("transformer.embed_tokens.weight", text)
    add("text_lm_head.weight", text)
    del text

    for index in range(topology.n_vq):
        add(
            f"audio_embeddings.{index}.weight",
            reader.read(f"audio_embeddings.{index}.word_embeddings.weight"),
        )
        add(
            f"audio_lm_heads.{index}.weight",
            reader.read(f"audio_lm_heads.{index}.weight"),
        )

    q_per_group = topology.num_attention_heads // topology.num_query_groups
    query_width = topology.num_attention_heads * topology.head_dim
    kv_width = topology.num_query_groups * topology.head_dim
    for layer in range(topology.num_layers):
        target = f"transformer.layers.{layer}"
        for target_suffix, source_suffix in (
            (
                "input_layernorm.weight",
                "self_attention.linear_qkv.layer_norm_weight",
            ),
            ("post_attention_layernorm.weight", "mlp.linear_fc1.layer_norm_weight"),
            ("self_attn.q_norm.weight", "self_attention.q_layernorm.weight"),
            ("self_attn.k_norm.weight", "self_attention.k_layernorm.weight"),
        ):
            add(
                f"{target}.{target_suffix}",
                reader.read(f"decoder.layers.{source_suffix}", layer=layer),
            )
        packed = reader.read(
            "decoder.layers.self_attention.linear_qkv.weight", layer=layer
        ).view(
            topology.num_query_groups,
            q_per_group + 2,
            topology.head_dim,
            topology.hidden_size,
        )
        add(
            f"{target}.self_attn.q_proj.weight",
            packed[:, :q_per_group].reshape(query_width, topology.hidden_size),
        )
        add(
            f"{target}.self_attn.k_proj.weight",
            packed[:, q_per_group].reshape(kv_width, topology.hidden_size),
        )
        add(
            f"{target}.self_attn.v_proj.weight",
            packed[:, q_per_group + 1].reshape(kv_width, topology.hidden_size),
        )
        add(
            f"{target}.self_attn.o_proj.weight",
            reader.read(
                "decoder.layers.self_attention.linear_proj.weight", layer=layer
            ),
        )
        fc1 = reader.read("decoder.layers.mlp.linear_fc1.weight", layer=layer)
        gate, up = fc1.chunk(2, dim=0)
        add(f"{target}.mlp.gate_proj.weight", gate)
        add(f"{target}.mlp.up_proj.weight", up)
        add(
            f"{target}.mlp.down_proj.weight",
            reader.read("decoder.layers.mlp.linear_fc2.weight", layer=layer),
        )

    add(
        "transformer.norm.weight",
        reader.read("decoder.final_layernorm.weight"),
    )
    for key in sorted(reader.keys()):
        if key.startswith("local_transformer."):
            add(key, reader.read(key))
    add(
        "local_text_lm_head.weight",
        reader.read("local_text_lm_head.weight"),
    )
    writer_written = writer.finish()
    if writer_written != written:
        raise RuntimeError(
            "Streaming writer tensor manifest mismatch: "
            f"missing={sorted(written - writer_written)}, "
            f"unexpected={sorted(writer_written - written)}"
        )
    return written


def _build_config(
    template: dict[str, Any], topology: Any, dtype: torch.dtype
) -> dict[str, Any]:
    config = json.loads(json.dumps(template))
    language = config.get("qwen3_config") or config.get("language_config")
    local = config.get("gpt2_config")
    if not isinstance(language, dict) or not isinstance(local, dict):
        raise ValueError(
            "Processor runtime template must contain qwen3/language and gpt2 configs."
        )
    language.update(
        {
            "model_type": "qwen3",
            "num_hidden_layers": topology.num_layers,
            "hidden_size": topology.hidden_size,
            "num_attention_heads": topology.num_attention_heads,
            "num_key_value_heads": topology.num_query_groups,
            "head_dim": topology.head_dim,
            "intermediate_size": topology.ffn_hidden_size,
            "vocab_size": topology.text_vocab_size,
            "max_position_embeddings": topology.max_position_embeddings,
            "max_window_layers": topology.num_layers,
            "hidden_act": "silu",
            "attention_bias": False,
            "tie_word_embeddings": True,
            "rope_theta": topology.rope_theta,
            "rms_norm_eps": topology.norm_epsilon,
            "initializer_range": topology.init_method_std,
            "use_cache": True,
        }
    )
    if "layer_types" in language:
        language["layer_types"] = ["full_attention"] * topology.num_layers
    config["qwen3_config"] = json.loads(json.dumps(language))
    config["language_config"] = json.loads(json.dumps(language))
    local.update(
        {
            "model_type": "gpt2",
            "vocab_size": topology.text_vocab_size,
            "n_embd": topology.local_hidden_size,
            "n_layer": topology.local_num_layers,
            "n_head": topology.local_num_attention_heads,
            "n_inner": topology.local_ffn_hidden_size,
            "n_positions": topology.n_vq + 1,
            "n_ctx": topology.n_vq + 1,
            "activation_function": topology.local_activation,
            "layer_norm_epsilon": topology.local_norm_epsilon,
            "rope_base": topology.local_rope_base,
            "position_embedding_type": "rope",
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
            "use_cache": True,
        }
    )
    config["gpt2_config"] = local
    auto_map = dict(config.get("auto_map") or {})
    auto_map.pop("AutoModel", None)
    auto_map["AutoConfig"] = "configuration_moss_tts.MossTTSLocalConfig"
    auto_map["AutoProcessor"] = "processing_moss_tts.MossTTSLocalProcessor"
    config.update(
        {
            "model_type": "moss_tts_local",
            "architectures": ["MossTTSLocalModel"],
            "auto_map": auto_map,
            "dtype": str(dtype).removeprefix("torch."),
            "tie_word_embeddings": True,
            "tie_audio_embeddings": False,
            "tie_audio_embeddings_and_output_weights": False,
            "embedding_head_storage": "split_v1",
            "n_vq": topology.n_vq,
            "audio_vocab_size": topology.speech_vocab_size,
            "audio_codebook_sizes": [topology.speech_vocab_size] * topology.n_vq,
            "audio_pad_token_id": topology.speech_vocab_size,
            "audio_pad_code": topology.speech_vocab_size,
            "pad_token_id": 151643,
            "im_start_token_id": 151644,
            "im_end_token_id": 151645,
            "audio_start_token_id": 151652,
            "audio_end_token_id": topology.audio_end_token_id,
            "audio_user_slot_token_id": 151654,
            "audio_assistant_slot_token_id": 151656,
            "audio_assistant_gen_slot_token_id": 151656,
            "sampling_rate": 48000,
            "local_transformer_layers": topology.local_num_layers,
            "local_text_head_mode": "binary",
            "sglang_omni_weight_layout": "mosslite_split_v1",
        }
    )
    return config


def _copy_runtime_assets(
    processor_runtime: Path, tokenizer: Path, output: Path
) -> None:
    for root, names, description in (
        (processor_runtime, PROCESSOR_RUNTIME_FILES, "processor runtime"),
        (tokenizer, TOKENIZER_FILES, "tokenizer"),
    ):
        for name in names:
            source = root / name
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(f"{description} asset is missing: {source}")
            shutil.copy2(source, output / name)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def convert(args: argparse.Namespace) -> None:
    (
        reader_type,
        writer_type,
        publish_staging_directory,
        topology_from_identity,
    ) = _load_mosslite_api(args.mosslite_repo)
    checkpoint = _directory(args.megatron_input_dir, "Megatron checkpoint root")
    processor_runtime = _directory(
        args.processor_runtime_dir, "processor runtime template"
    )
    tokenizer = _directory(args.tokenizer_dir, "tokenizer source")
    output = args.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Staging output already exists: {staging}")
    staging.mkdir()

    try:
        reader = reader_type(
            checkpoint,
            expected_model_identity=None,
            checkpoint_step=args.checkpoint_step,
        )
        topology = topology_from_identity(reader.model_identity)
        _validate_topology(topology, reader.model_identity)
        config = _build_config(
            _json(processor_runtime / "config.json", "processor runtime config"),
            topology,
            reader.dtype("text_embedding.word_embeddings.weight"),
        )
        written = _export_weights(
            reader,
            topology,
            staging,
            writer_type,
            args.max_shard_size,
        )
        _copy_runtime_assets(processor_runtime, tokenizer, staging)
        _write_json(staging / "config.json", config)
        manifest_path = reader.payload.parent / "checkpoint.json"
        conversion_manifest = {
            "format": "sglang_omni_moss_tts_local_v1",
            "source_checkpoint_root": str(reader.root),
            "source_model_payload": str(reader.payload),
            "source_checkpoint_manifest_sha256": _sha256(manifest_path),
            "source_iteration": _json(manifest_path, "checkpoint manifest").get(
                "iteration"
            ),
            "source_model_identity": reader.model_identity,
            "tensor_count": len(written),
            "tensor_names_sha256": hashlib.sha256(
                "\n".join(sorted(written)).encode("utf-8")
            ).hexdigest(),
            "processor_runtime_source": str(processor_runtime),
            "tokenizer_source": str(tokenizer),
        }
        _write_json(staging / "sglang_omni_conversion.json", conversion_manifest)
        (staging / "README.md").write_text(
            "# MOSS-TTS Local for SGLang-Omni\n\n"
            "This serving artifact was converted from a mossLite logical "
            "Megatron checkpoint. It preserves independent RVQ embeddings and "
            "output heads (`split_v1`) and uses the MossFlux v2 token contract. "
            "It intentionally publishes AutoConfig/AutoProcessor only; model "
            "execution is provided by SGLang-Omni. See "
            "`sglang_omni_conversion.json` for exact provenance.\n",
            encoding="utf-8",
        )
        publish_staging_directory(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosslite-repo", type=Path, required=True)
    parser.add_argument("--megatron-input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--processor-runtime-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="2GB")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    convert(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
