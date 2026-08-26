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

import json
import re

import pytest

from verl.third_party import ifevalg
from verl.third_party.ifevalg import instructions_util as ifevalg_instructions_util
from verl.utils.reward_score import default_compute_score, nemotron_cascade_mopd


def _payload(verifier, **kwargs):
    return json.dumps({"verifier": verifier, **kwargs})


@pytest.fixture(autouse=True)
def _use_resource_free_ifevalg_constraints(monkeypatch):
    """The focused fixtures use verifier rules that do not access NLTK data."""
    monkeypatch.setattr(ifevalg, "require_nltk_resources", lambda: None)
    monkeypatch.setattr(
        ifevalg_instructions_util,
        "count_sentences",
        lambda value: len(re.findall(r"[.!?]+", value)),
    )


def test_acereason_uses_math_verify_and_default_reward_route_on_cpu(monkeypatch):
    calls = []

    def fake_math_verify(model_output, ground_truth):
        calls.append((model_output, ground_truth))
        return 1.0

    monkeypatch.setattr(nemotron_cascade_mopd.math_verify, "compute_score", fake_math_verify)
    result = default_compute_score(
        data_source="nvidia/Nemotron-Cascade-2-RL-data",
        solution_str=r"The final answer is \boxed{42}.",
        ground_truth=_payload("math_answer", expected_answer="42"),
        extra_info={"reward_data_source": "nemotron_cascade_mopd"},
    )

    assert calls == [(r"The final answer is \boxed{42}.", "42")]
    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["parse_failure"] is False
    assert result["math_verify_score"] == 1.0


def test_instruction_following_scores_all_constraints_after_qwen_thinking_on_cpu():
    result = nemotron_cascade_mopd.compute_score(
        "<think>The hidden draft contains secret.</think>\nThe signal is useful. The signal is clear.",
        _payload(
            "instruction_following",
            instruction_id_list=[
                "keywords:existence",
                "keywords:forbidden_words",
                "length_constraints:number_sentences",
            ],
            kwargs=[
                {"keywords": ["signal"]},
                {"forbidden_words": ["secret"]},
                {"num_sentences": 2, "relation": "at least"},
            ],
        ),
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["parse_failure"] is False
    assert result["constraint_count"] == 3
    assert result["constraints_satisfied"] == 3
    assert result["constraint_pass_rate"] == 1.0
    assert result["thinking_stripped"] is True


def test_instruction_following_score_is_constraint_ratio_but_acc_requires_all_on_cpu():
    result = nemotron_cascade_mopd.compute_score(
        "The signal is present, but so is secret.",
        _payload(
            "instruction_following",
            instruction_id_list=["keywords:existence", "keywords:forbidden_words"],
            kwargs=[{"keywords": ["signal"]}, {"forbidden_words": ["secret"]}],
        ),
    )

    assert result["score"] == 0.5
    assert result["constraint_pass_rate"] == 0.5
    assert result["acc"] is False
    assert result["instruction_backends"] == "vendored_ifevalg"


def test_instruction_following_does_not_hide_missing_offline_verifier_resources_on_cpu(monkeypatch):
    def missing_resources():
        raise RuntimeError("offline resources missing")

    monkeypatch.setattr(ifevalg, "require_nltk_resources", missing_resources)

    with pytest.raises(RuntimeError, match="offline resources missing"):
        nemotron_cascade_mopd.compute_score(
            "signal",
            _payload(
                "instruction_following",
                instruction_id_list=["keywords:existence"],
                kwargs=[{"keywords": ["signal"]}],
            ),
        )


def test_structured_output_strips_qwen_thinking_and_json_fence_then_validates_schema_on_cpu():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}, "units": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    result = nemotron_cascade_mopd.compute_score(
        '<think>Use an integer.</think>\n```json\n{"answer": 42, "units": "m"}\n```',
        _payload("json_schema", schema_type="json", schema_str=json.dumps(schema)),
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["parse_failure"] is False
    assert result["json_parsed"] is True
    assert result["schema_valid"] is True
    assert result["schema_normalized"] is False
    assert result["schema_normalization_count"] == 0
    assert result["thinking_stripped"] is True


def test_structured_output_normalizes_released_nvidia_schema_dialect_on_cpu():
    schema = {
        "$id": "#/definitions/Example",
        "type": "object",
        "required": True,
        "properties": {
            "name": {"type": "string", "required": True},
            "optional": {"type": "string", "required": False},
            "nested": {
                "type": "object",
                "description": {},
                "properties": {
                    "value": {"type": "integer", "required": True},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
            "cost": {"type": "number", "minimum": 0, "exclusiveMinimum": False},
            "description": "Top-level description",
            "additionalProperties": False,
            "minProperties": 1,
        },
    }

    result = nemotron_cascade_mopd.compute_score(
        '{"name":"x","optional":"y","nested":{"value":1},"cost":0}',
        _payload("json_schema", schema=schema),
    )

    assert result["score"] == 1.0
    assert result["schema_valid"] is True
    assert result["schema_normalized"] is True
    assert result["schema_normalization_count"] == 12


def test_structured_output_distinguishes_json_parse_failure_from_schema_failure_on_cpu():
    payload = _payload(
        "json_schema",
        schema={"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]},
    )

    invalid_json = nemotron_cascade_mopd.compute_score("```json\n{answer: 42}\n```", payload)
    invalid_schema_instance = nemotron_cascade_mopd.compute_score('{"answer": "forty-two"}', payload)

    assert invalid_json["score"] == 0.0
    assert invalid_json["parse_failure"] is True
    assert invalid_json["json_parsed"] is False
    assert invalid_schema_instance["score"] == 0.0
    assert invalid_schema_instance["parse_failure"] is False
    assert invalid_schema_instance["json_parsed"] is True
    assert invalid_schema_instance["schema_valid"] is False
    assert "validation_error" in invalid_schema_instance


def test_stem_mcqa_uses_sample_regex_capture_and_last_formatted_answer_on_cpu():
    result = nemotron_cascade_mopd.compute_score(
        "<think>I first considered A.</think>Answer: *A*\nCorrection: Answer: *C*",
        _payload(
            "mcqa_regex",
            expected_answer="C",
            template_metadata={"output_regex": r"Answer:\s*\*([A-Za-z0-9])\*"},
        ),
    )

    assert result["score"] == 1.0
    assert result["acc"] is True
    assert result["parse_failure"] is False
    assert result["predicted_answer"] == "C"
    assert result["regex_match_count"] == 2


def test_stem_mcqa_missing_required_format_is_a_parse_failure_on_cpu():
    result = nemotron_cascade_mopd.compute_score(
        "I think the correct option is B.",
        _payload("mcqa_regex", expected_answer="B", output_regex=r"\*\*([A-Za-z0-9])\*\*"),
    )

    assert result["score"] == 0.0
    assert result["acc"] is False
    assert result["parse_failure"] is True
    assert result["error"] == "output_regex_did_not_match"


def test_bundled_instruction_registry_covers_every_constraint_in_local_mopd_on_cpu():
    expected_ids = {
        "change_case:capital_word_frequency",
        "change_case:english_capital",
        "change_case:english_lowercase",
        "combination:two_responses",
        "copy:repeat_phrase",
        "count:count_increment_word",
        "count:count_unique",
        "count:counting_composition",
        "count:lowercase_counting",
        "detectable_content:number_placeholders",
        "detectable_content:postscript",
        "detectable_format:bigram_wrapping",
        "detectable_format:constrained_response",
        "detectable_format:json_format",
        "detectable_format:multiple_sections",
        "detectable_format:number_bullet_lists",
        "detectable_format:number_highlighted_sections",
        "detectable_format:sentence_hyphens",
        "detectable_format:square_brackets",
        "detectable_format:title",
        "first_word:first_word_answer",
        "first_word:first_word_sent",
        "keywords:existence",
        "keywords:forbidden_words",
        "keywords:frequency",
        "keywords:keyword_specific_position",
        "keywords:letter_frequency",
        "keywords:no_adjacent_consecutive",
        "keywords:palindrome",
        "keywords:start_end",
        "keywords:word_count_different_numbers",
        "keywords:word_once",
        "language:response_language",
        "last_word:last_word_answer",
        "last_word:last_word_sent",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_paragraphs",
        "length_constraints:number_sentences",
        "length_constraints:number_words",
        "letters:letter_counting",
        "letters:letter_counting2",
        "paragraphs:paragraphs",
        "paragraphs:paragraphs2",
        "punctuation:no_comma",
        "punctuation:punctuation_dot",
        "punctuation:punctuation_exclamation",
        "startend:end_checker",
        "startend:quotation",
    }

    assert nemotron_cascade_mopd.SUPPORTED_INSTRUCTION_IDS == expected_ids


@pytest.mark.parametrize(
    ("solution", "payload", "error"),
    [
        ("answer", "not-json", "valid JSON object"),
        ("answer", _payload("unknown"), "Unsupported MOPD task_type/verifier"),
        (
            "answer",
            _payload("mcqa_regex", expected_answer="A", output_regex="no capture group"),
            "capture group 1",
        ),
        (
            "answer",
            _payload(
                "instruction_following",
                instruction_id_list=["keywords:frequency"],
                kwargs=[{"keyword": "x", "frequency": 1}],
            ),
            "missing=['relation']",
        ),
    ],
)
def test_malformed_payload_or_scoring_config_raises_on_cpu(solution, payload, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        nemotron_cascade_mopd.compute_score(solution, payload)


def test_malformed_model_answer_returns_zero_instead_of_raising_on_cpu():
    result = nemotron_cascade_mopd.compute_score(
        None,
        _payload("mcqa_regex", expected_answer="A", output_regex=r"\*\*([A-Za-z0-9])\*\*"),
    )

    assert result["score"] == 0.0
    assert result["acc"] is False
    assert result["parse_failure"] is True
