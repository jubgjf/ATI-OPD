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

"""Offline runtime checks for the vendored IFBench evaluator."""

from __future__ import annotations

_NLTK_RESOURCES = (
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
    "corpora/stopwords",
    "taggers/averaged_perceptron_tagger_eng",
)


def require_nltk_resources() -> None:
    """Fail early when IFBench's pre-provisioned NLTK data is unavailable."""
    try:
        import nltk
        from langdetect import DetectorFactory
    except ImportError as exc:
        raise RuntimeError("IFBench evaluation dependencies are not installed.") from exc

    # Import the registry during launcher preflight so every vendored Python
    # dependency is checked before a long evaluation starts.
    from verl.third_party.ifbench import instructions_registry  # noqa: F401

    DetectorFactory.seed = 0

    missing: list[str] = []
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(resource)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "IFBench requires pre-provisioned NLTK resources and never downloads them at runtime. "
            f"Missing: {joined}. Set NLTK_DATA to the staged resource directory before launching training."
        )


__all__ = ["require_nltk_resources"]
