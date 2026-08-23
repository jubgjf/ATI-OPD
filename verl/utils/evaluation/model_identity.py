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

"""Stable, content-addressed identities for standalone evaluation models."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping

from verl.utils.evaluation.checkpoint_evaluation import payload_fingerprint, write_json_atomic

MODEL_MANIFEST_SCHEMA_VERSION = 1
MODEL_MANIFEST_CACHE_SCHEMA_VERSION = 1
MODEL_FINGERPRINT_ALIASES_SCHEMA_VERSION = 1
FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_MODEL_METADATA_NAMES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "quant_config.json",
    "quantize_config.json",
    "quantization_config.json",
    "adapter_config.json",
)
_LEGACY_METADATA_NAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
)
_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_files(model_path: Path) -> tuple[list[Path], list[Path]]:
    metadata = [model_path / name for name in _MODEL_METADATA_NAMES if (model_path / name).is_file()]
    metadata.extend(path for path in model_path.glob("*.py") if path.is_file())
    weights = sorted({path for pattern in _WEIGHT_PATTERNS for path in model_path.glob(pattern) if path.is_file()})
    return sorted(set(metadata)), weights


def _cache_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _read_manifest_cache(cache_path: Path | None, model_path: Path) -> dict[str, dict[str, Any]]:
    if cache_path is None or not cache_path.is_file():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("schema_version") != MODEL_MANIFEST_CACHE_SCHEMA_VERSION or payload.get("model_path") != str(
        model_path
    ):
        return {}
    entries = payload.get("files")
    if not isinstance(entries, dict):
        return {}
    return {str(key): dict(value) for key, value in entries.items() if isinstance(value, Mapping)}


def _content_descriptor(
    path: Path,
    *,
    model_path: Path,
    cached: Mapping[str, Mapping[str, Any]],
    force_rehash: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    logical_path = path.relative_to(model_path).as_posix()
    signature = _cache_signature(path)
    cached_entry = cached.get(logical_path) or {}
    sha256 = None
    if not force_rehash and all(cached_entry.get(key) == value for key, value in signature.items()):
        candidate = cached_entry.get("sha256")
        if isinstance(candidate, str) and FINGERPRINT_PATTERN.fullmatch(candidate):
            sha256 = candidate
    if sha256 is None:
        sha256 = _sha256_file(path)
        refreshed_signature = _cache_signature(path)
        if refreshed_signature != signature:
            raise RuntimeError(f"Model file changed while its content fingerprint was being computed: {path}")
    semantic = {
        "logical_path": logical_path,
        "size": signature["size"],
        "sha256": sha256,
    }
    cache_entry = {**signature, "sha256": sha256}
    return semantic, cache_entry


def stable_model_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the path/mtime-independent portion covered by a model fingerprint."""

    return {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "metadata": sorted(
            (
                {
                    "logical_path": str(entry["logical_path"]),
                    "size": int(entry["size"]),
                    "sha256": str(entry["sha256"]),
                }
                for entry in manifest.get("metadata", [])
            ),
            key=lambda entry: str(entry["logical_path"]),
        ),
        "weights": sorted(
            (
                {
                    "logical_path": str(entry["logical_path"]),
                    "size": int(entry["size"]),
                    "sha256": str(entry["sha256"]),
                }
                for entry in manifest.get("weights", [])
            ),
            key=lambda entry: str(entry["logical_path"]),
        ),
    }


def validate_stable_model_manifest(manifest: Mapping[str, Any]) -> tuple[bool, str]:
    if not isinstance(manifest, Mapping):
        return False, "model manifest is not an object"
    if manifest.get("schema_version") != MODEL_MANIFEST_SCHEMA_VERSION:
        return False, f"unsupported model manifest schema: {manifest.get('schema_version')!r}"
    try:
        payload = stable_model_manifest_payload(manifest)
        for group in ("metadata", "weights"):
            logical_paths: set[str] = set()
            for entry in payload[group]:
                logical_path = str(entry["logical_path"])
                size = int(entry["size"])
                sha256 = str(entry["sha256"])
                if not logical_path or logical_path.startswith("/") or ".." in Path(logical_path).parts:
                    return False, f"invalid logical model path: {logical_path!r}"
                if logical_path in logical_paths:
                    return False, f"duplicate logical model path: {logical_path!r}"
                if size < 0 or not FINGERPRINT_PATTERN.fullmatch(sha256):
                    return False, f"invalid descriptor for {logical_path!r}"
                logical_paths.add(logical_path)
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"invalid model manifest: {exc}"
    expected = payload_fingerprint(payload)
    if manifest.get("fingerprint") != expected:
        return False, "model manifest fingerprint is invalid"
    return True, "valid"


def capture_stable_model_manifest(
    model_path: str | Path,
    *,
    cache_path: str | Path | None = None,
    force_rehash: bool = False,
) -> dict[str, Any]:
    """Hash model semantics while using stat data only as a local checksum-cache key."""

    resolved = Path(model_path).expanduser().resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"Model directory is unavailable: {resolved}")
    resolved_cache = None if cache_path is None else Path(cache_path).expanduser().resolve()
    cached = _read_manifest_cache(resolved_cache, resolved)
    metadata_files, weight_files = _model_files(resolved)
    cache_entries: dict[str, dict[str, Any]] = {}

    def describe(files: list[Path]) -> list[dict[str, Any]]:
        descriptors: list[dict[str, Any]] = []
        for file_path in files:
            semantic, cache_entry = _content_descriptor(
                file_path,
                model_path=resolved,
                cached=cached,
                force_rehash=force_rehash,
            )
            descriptors.append(semantic)
            cache_entries[semantic["logical_path"]] = cache_entry
        return descriptors

    manifest = {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "metadata": describe(metadata_files),
        "weights": describe(weight_files),
    }
    manifest["fingerprint"] = payload_fingerprint(stable_model_manifest_payload(manifest))
    if resolved_cache is not None:
        write_json_atomic(
            resolved_cache,
            {
                "schema_version": MODEL_MANIFEST_CACHE_SCHEMA_VERSION,
                "model_path": str(resolved),
                "files": cache_entries,
                "model_fingerprint": manifest["fingerprint"],
                "updated_at_unix": time.time(),
            },
        )
    return manifest


def model_manifest_cache_path(cache_root: str | Path, model_path: str | Path) -> Path:
    resolved = Path(model_path).expanduser().resolve()
    resource_hash = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
    return Path(cache_root).expanduser().resolve() / f"{resource_hash}.json"


def legacy_stat_model_fingerprint(model_path: str | Path) -> str:
    """Reproduce endpoint schema v1 exactly for safe in-place upgrades."""

    resolved = Path(model_path).expanduser().resolve()
    metadata = {
        name: (resolved / name).read_text(encoding="utf-8")
        for name in _LEGACY_METADATA_NAMES
        if (resolved / name).is_file()
    }
    weight_files = sorted({path for pattern in _WEIGHT_PATTERNS for path in resolved.glob(pattern) if path.is_file()})
    return payload_fingerprint(
        {
            "path": str(resolved),
            "metadata": metadata,
            "weights": [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in weight_files
            ],
        }
    )


def load_model_fingerprint_aliases(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    alias_path = Path(path).expanduser().resolve()
    if not alias_path.is_file():
        return {}
    try:
        payload = json.loads(alias_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read teacher model aliases from {alias_path}: {exc}") from exc
    if payload.get("schema_version") != MODEL_FINGERPRINT_ALIASES_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported teacher model alias schema in {alias_path}.")
    aliases = payload.get("aliases")
    if not isinstance(aliases, dict):
        raise RuntimeError(f"Teacher model alias file has no aliases object: {alias_path}")
    validated: dict[str, dict[str, Any]] = {}
    for legacy_fingerprint, raw_entry in aliases.items():
        if not FINGERPRINT_PATTERN.fullmatch(str(legacy_fingerprint)) or not isinstance(raw_entry, Mapping):
            raise RuntimeError(f"Invalid teacher model alias entry in {alias_path}: {legacy_fingerprint!r}")
        entry = dict(raw_entry)
        stable_fingerprint = str(entry.get("stable_model_fingerprint") or "")
        manifest = entry.get("model_manifest")
        if not FINGERPRINT_PATTERN.fullmatch(stable_fingerprint) or not isinstance(manifest, Mapping):
            raise RuntimeError(f"Teacher model alias {legacy_fingerprint} is incomplete in {alias_path}.")
        valid, reason = validate_stable_model_manifest(manifest)
        if not valid or manifest.get("fingerprint") != stable_fingerprint:
            raise RuntimeError(f"Teacher model alias {legacy_fingerprint} is invalid in {alias_path}: {reason}")
        validated[str(legacy_fingerprint)] = entry
    return validated


def register_model_fingerprint_aliases(
    alias_path: str | Path,
    *,
    legacy_fingerprints: list[str],
    model_path: str | Path,
    expected_stable_fingerprint: str,
    cache_path: str | Path | None = None,
    force_rehash: bool = False,
    note: str | None = None,
) -> dict[str, Any]:
    """Bind explicitly asserted legacy identities to a verified current content manifest."""

    if not legacy_fingerprints:
        raise ValueError("At least one legacy model fingerprint is required.")
    invalid = [value for value in legacy_fingerprints if not FINGERPRINT_PATTERN.fullmatch(value)]
    if invalid:
        raise ValueError(f"Legacy model fingerprints must be 64 lowercase hex characters: {invalid}")
    if not FINGERPRINT_PATTERN.fullmatch(expected_stable_fingerprint):
        raise ValueError("Expected stable model fingerprint must be 64 lowercase hex characters.")
    manifest = capture_stable_model_manifest(model_path, cache_path=cache_path, force_rehash=force_rehash)
    current_fingerprint = str(manifest["fingerprint"])
    if current_fingerprint != expected_stable_fingerprint:
        raise RuntimeError(
            "Current model content does not match --expected-stable-fingerprint: "
            f"expected={expected_stable_fingerprint}, current={current_fingerprint}."
        )

    destination = Path(alias_path).expanduser().resolve()
    existing_aliases = load_model_fingerprint_aliases(destination)
    aliases = dict(existing_aliases)
    resolved_model_path = str(Path(model_path).expanduser().resolve())
    for legacy_fingerprint in legacy_fingerprints:
        existing = aliases.get(legacy_fingerprint)
        if existing is not None and existing.get("stable_model_fingerprint") != current_fingerprint:
            raise RuntimeError(f"Legacy fingerprint {legacy_fingerprint} is already bound to a different stable model.")
        aliases[legacy_fingerprint] = {
            "stable_model_fingerprint": current_fingerprint,
            "model_manifest": manifest,
            "registered_at_unix": time.time(),
            "registered_model_path": resolved_model_path,
            "note": None if note is None else str(note),
        }
    payload = {
        "schema_version": MODEL_FINGERPRINT_ALIASES_SCHEMA_VERSION,
        "aliases": aliases,
    }
    write_json_atomic(destination, payload)
    return payload
