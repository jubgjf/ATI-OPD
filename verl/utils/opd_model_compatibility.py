# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preflight validation for token-aligned Qwen3 on-policy distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoTokenizer

from verl.tools.python_sandbox_client import CODE_INTERPRETER_SCHEMA
from verl.utils.tokenizer import normalize_token_ids

_CRITICAL_TOKENS = (
    "<tool_call>",
    "</tool_call>",
    "<think>",
    "</think>",
    "<|im_end|>",
)
_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


class ModelPairCompatibilityError(ValueError):
    """Raised when student token IDs cannot safely be scored by the teacher."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _token_ids_hash(token_ids: list[int]) -> str:
    return _sha256_bytes(json.dumps(token_ids, separators=(",", ":")).encode())


def _vocab_hash(vocab: dict[str, int]) -> str:
    digest = hashlib.sha256()
    for token, token_id in sorted(vocab.items()):
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(token_id)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _metadata_hashes(model_path: str) -> dict[str, str]:
    root = Path(os.path.realpath(os.path.expanduser(model_path)))
    if not root.is_dir():
        return {}
    hashes = {}
    for filename in _METADATA_FILES:
        path = root / filename
        if path.is_file():
            hashes[filename] = _sha256_bytes(path.read_bytes())
    return hashes


def _validate_qwen3_config(label: str, config: Any) -> dict[str, Any]:
    model_type = str(getattr(config, "model_type", ""))
    architectures = [str(value) for value in (getattr(config, "architectures", None) or [])]
    if model_type != "qwen3" or "Qwen3ForCausalLM" not in architectures:
        raise ModelPairCompatibilityError(
            f"{label} must be Qwen3ForCausalLM, got model_type={model_type!r}, architectures={architectures!r}."
        )
    vocab_size = int(getattr(config, "vocab_size", 0))
    context_length = int(getattr(config, "max_position_embeddings", 0))
    if vocab_size <= 0 or context_length <= 0:
        raise ModelPairCompatibilityError(
            f"{label} config must expose positive vocab_size and max_position_embeddings, "
            f"got vocab_size={vocab_size}, max_position_embeddings={context_length}."
        )
    return {
        "model_type": model_type,
        "architectures": architectures,
        "vocab_size": vocab_size,
        "max_position_embeddings": context_length,
        "commit_hash": getattr(config, "_commit_hash", None),
    }


def _first_vocab_difference(student_vocab: dict[str, int], teacher_vocab: dict[str, int]) -> str:
    for token in sorted(student_vocab.keys() | teacher_vocab.keys()):
        student_id = student_vocab.get(token)
        teacher_id = teacher_vocab.get(token)
        if student_id != teacher_id:
            return f"token={token!r}, student_id={student_id!r}, teacher_id={teacher_id!r}"
    return "unknown difference"


def _render_probe(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[CODE_INTERPRETER_SCHEMA],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=True,
    )
    return normalize_token_ids(rendered)


def validate_loaded_qwen3_pair(
    *,
    student_config: Any,
    teacher_config: Any,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    sequence_length: int,
) -> dict[str, Any]:
    """Validate an already-loaded pair and return a compact reproducibility manifest."""
    if sequence_length <= 0:
        raise ModelPairCompatibilityError(f"sequence_length must be positive, got {sequence_length}.")

    student = _validate_qwen3_config("student", student_config)
    teacher = _validate_qwen3_config("teacher", teacher_config)
    if student["vocab_size"] != teacher["vocab_size"]:
        raise ModelPairCompatibilityError(
            f"Model vocab_size mismatch: student={student['vocab_size']}, teacher={teacher['vocab_size']}."
        )

    student_vocab = {str(token): int(token_id) for token, token_id in student_tokenizer.get_vocab().items()}
    teacher_vocab = {str(token): int(token_id) for token, token_id in teacher_tokenizer.get_vocab().items()}
    if len(student_tokenizer) != len(teacher_tokenizer) or student_vocab != teacher_vocab:
        difference = _first_vocab_difference(student_vocab, teacher_vocab)
        raise ModelPairCompatibilityError(
            "Tokenizer vocabulary mismatch: "
            f"student_len={len(student_tokenizer)}, teacher_len={len(teacher_tokenizer)}, first_diff={difference}."
        )
    if len(student_tokenizer) > student["vocab_size"]:
        raise ModelPairCompatibilityError(
            f"Tokenizer length {len(student_tokenizer)} exceeds model vocab_size {student['vocab_size']}."
        )

    critical_token_ids = {}
    for token in _CRITICAL_TOKENS:
        if token not in student_vocab:
            raise ModelPairCompatibilityError(
                f"Required Qwen3 tool token is missing from both vocabularies: {token!r}."
            )
        critical_token_ids[token] = student_vocab[token]
    for attribute in ("eos_token_id", "pad_token_id"):
        student_id = getattr(student_tokenizer, attribute, None)
        teacher_id = getattr(teacher_tokenizer, attribute, None)
        if student_id != teacher_id or student_id is None:
            raise ModelPairCompatibilityError(
                f"Tokenizer {attribute} mismatch: student={student_id!r}, teacher={teacher_id!r}."
            )
        critical_token_ids[attribute] = int(student_id)

    probes = {
        "initial": [{"role": "user", "content": "Compute 6 * 7 and verify the result."}],
        "after_observation": [
            {"role": "user", "content": "Compute 6 * 7 and verify the result."},
            {
                "role": "assistant",
                "content": (
                    "<think>I will verify the multiplication.</think>\n"
                    '<tool_call>\n{"name":"code_interpreter","arguments":{"code":"6 * 7"}}\n</tool_call>'
                ),
            },
            {"role": "tool", "content": "42\n"},
        ],
    }
    probe_hashes = {}
    probe_lengths = {}
    for name, messages in probes.items():
        student_ids = _render_probe(student_tokenizer, messages)
        teacher_ids = _render_probe(teacher_tokenizer, messages)
        if student_ids != teacher_ids:
            mismatch_index = next(
                (index for index, pair in enumerate(zip(student_ids, teacher_ids, strict=False)) if pair[0] != pair[1]),
                min(len(student_ids), len(teacher_ids)),
            )
            raise ModelPairCompatibilityError(
                f"enable_thinking=True tool chat-template mismatch for probe={name!r} at token index "
                f"{mismatch_index}: student_len={len(student_ids)}, teacher_len={len(teacher_ids)}."
            )
        probe_hashes[name] = _token_ids_hash(student_ids)
        probe_lengths[name] = len(student_ids)

    for label, summary in (("student", student), ("teacher", teacher)):
        if sequence_length > summary["max_position_embeddings"]:
            raise ModelPairCompatibilityError(
                f"Configured sequence length {sequence_length} exceeds {label} max_position_embeddings="
                f"{summary['max_position_embeddings']}."
            )

    return {
        "status": "compatible",
        "sequence_length": sequence_length,
        "student": student,
        "teacher": teacher,
        "tokenizer": {
            "length": len(student_tokenizer),
            "vocab_sha256": _vocab_hash(student_vocab),
            "critical_token_ids": critical_token_ids,
            "student_class": type(student_tokenizer).__name__,
            "teacher_class": type(teacher_tokenizer).__name__,
        },
        "chat_template_probes": {
            "enable_thinking": True,
            "token_sha256": probe_hashes,
            "token_lengths": probe_lengths,
        },
    }


def validate_local_qwen3_pair(*, student_path: str, teacher_path: str, sequence_length: int) -> dict[str, Any]:
    """Load a local model pair and run the Qwen3 compatibility checks."""
    load_kwargs = {"local_files_only": True, "trust_remote_code": False}
    student_config = AutoConfig.from_pretrained(student_path, **load_kwargs)
    teacher_config = AutoConfig.from_pretrained(teacher_path, **load_kwargs)
    student_tokenizer = AutoTokenizer.from_pretrained(student_path, **load_kwargs)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path, **load_kwargs)
    report = validate_loaded_qwen3_pair(
        student_config=student_config,
        teacher_config=teacher_config,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        sequence_length=sequence_length,
    )
    report["student"].update(
        {
            "path": str(Path(student_path).expanduser().resolve()),
            "metadata_sha256": _metadata_hashes(student_path),
        }
    )
    report["teacher"].update(
        {
            "path": str(Path(teacher_path).expanduser().resolve()),
            "metadata_sha256": _metadata_hashes(teacher_path),
        }
    )
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, help="Local Hugging Face student model path.")
    parser.add_argument("--teacher", required=True, help="Local Hugging Face teacher model path.")
    parser.add_argument("--sequence-length", required=True, type=int, help="Total configured prompt/response budget.")
    parser.add_argument("--report", required=True, type=Path, help="Destination JSON report.")
    args = parser.parse_args()

    try:
        report = validate_local_qwen3_pair(
            student_path=args.student,
            teacher_path=args.teacher,
            sequence_length=args.sequence_length,
        )
    except Exception as error:
        report = {
            "status": "incompatible",
            "student_path": str(Path(args.student).expanduser()),
            "teacher_path": str(Path(args.teacher).expanduser()),
            "sequence_length": args.sequence_length,
            "error": f"{type(error).__name__}: {error}",
        }
        _write_report(args.report, report)
        print(f"SOD model compatibility preflight failed: {report['error']}")
        return 2

    _write_report(args.report, report)
    print(
        "SOD model compatibility preflight passed: "
        f"tokenizer_len={report['tokenizer']['length']}, sequence_length={args.sequence_length}, "
        f"report={args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ModelPairCompatibilityError",
    "validate_loaded_qwen3_pair",
    "validate_local_qwen3_pair",
]
