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

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.utils.evaluation import opd_benchmark as benchmark
from verl.utils.evaluation import opd_result_status, teacher_intervention
from verl.utils.evaluation.checkpoint_evaluation import write_evaluation_spec, write_evaluation_state


def _generation_preparation(
    *,
    weight_mtime_ns: int = 20,
    weight_size: int = 10,
    temperature: float = 0.7,
) -> tuple[benchmark.GenerationPreparation, dict]:
    request = {
        "experiment": "exp",
        "global_step": 20,
        "benchmark": "math-ai/aime24",
        "dataset_root": "/datasets",
        "student_model_path": "/models/step-20",
        "student_tokenizer_model": "/models/step-20",
        "chat_template_model": "/models/teacher",
        "rollout_n": 16,
        "temperature": temperature,
    }
    tokenizer_info = {
        "model_path": "/models/step-20",
        "vocab_size": 2,
        "fingerprint_sha256": "tokenizer",
        "chat_template_sha256": "template",
        "special_token_ids": {},
    }
    legacy_manifest = {
        "dataset": [],
        "student_model_metadata": [],
        "student_model_weights": [
            {
                "path": "/models/step-20/model.safetensors",
                "size": 10,
                "mode": "stat",
                "mtime_ns": 10,
            }
        ],
        "student_tokenizer": [],
        "chat_template": [],
    }
    legacy_manifest["fingerprint"] = benchmark.payload_fingerprint(legacy_manifest)
    stable_manifest = {
        "schema_version": benchmark.INPUT_MANIFEST_SCHEMA_VERSION,
        "student_model_identity": {"kind": "committed_checkpoint", "fingerprint": "source-step-20"},
        "dataset": [],
        "student_model_metadata": [],
        "student_model_weights": [
            {
                "path": "/models/step-20/model.safetensors",
                "logical_path": "model.safetensors",
                "size": weight_size,
                "mode": "stat",
                "mtime_ns": weight_mtime_ns,
            }
        ],
        "student_tokenizer": [],
        "chat_template": [],
    }
    stable_manifest["fingerprint"] = benchmark.stable_input_manifest_fingerprint(stable_manifest)
    current_fingerprint = benchmark.generation_fingerprint(
        request=request,
        input_manifest=stable_manifest,
        dataset_fingerprint="dataset",
        tokenizer_info=tokenizer_info,
    )
    preparation = benchmark.GenerationPreparation(
        tokenizer=None,
        problems=[],
        tasks=[],
        dataset_fingerprint="dataset",
        input_manifest=stable_manifest,
        tokenizer_info=tokenizer_info,
        request=request,
        analysis={},
        fingerprint=current_fingerprint,
    )
    existing = {
        "schema_version": benchmark.SCHEMA_VERSION,
        "generation_schema_version": 2,
        "generation_complete": False,
        "generation_fingerprint": benchmark.payload_fingerprint(
            {
                "generation_schema_version": 2,
                "request": request,
                "input_manifest_fingerprint": legacy_manifest["fingerprint"],
                "dataset_fingerprint": "dataset",
                "tokenizer": tokenizer_info,
            }
        ),
        "request": request,
        "benchmark": {"dataset_fingerprint": "dataset"},
        "student_tokenizer": tokenizer_info,
        "input_manifest": legacy_manifest,
    }
    return preparation, existing


def _preflight_args(result_root: Path):
    return benchmark.build_parser().parse_args(
        [
            "generate",
            "--served-model-name",
            "student",
            "--result-root",
            str(result_root),
            "--benchmark",
            "math-ai/aime24",
            "--global-step",
            "20",
            "--experiment",
            "exp",
            "--dataset-root",
            "/datasets",
            "--student-model-path",
            "/models/step-20",
            "--student-model-identity",
            "source-step-20",
            "--student-tokenizer-model",
            "/models/step-20",
            "--chat-template-model",
            "/models/teacher",
            "--preflight-only",
        ]
    )


def test_benchmark_id_uses_full_path_and_rejects_traversal_on_cpu():
    first = benchmark.benchmark_id("math-ai/aime24")
    second = benchmark.benchmark_id("other/aime24")

    assert first != second
    assert first.startswith("math-ai__aime24--")
    with pytest.raises(ValueError, match="relative"):
        benchmark.benchmark_id("../aime24")


def test_atomic_result_writers_use_unique_temporary_files_on_cpu(tmp_path):
    json_path = tmp_path / "metadata.json"
    csv_path = tmp_path / "summary.csv"
    gzip_path = tmp_path / "rollouts.jsonl.gz"

    def write(index: int) -> None:
        benchmark.write_json_atomic(json_path, {"index": index})
        benchmark.write_csv_atomic(csv_path, [{"index": index}], ["index"])
        benchmark.write_jsonl_gz_atomic(gzip_path, [{"index": index}])

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(32)))

    assert json.loads(json_path.read_text(encoding="utf-8"))["index"] in range(32)
    assert csv_path.read_text(encoding="utf-8").startswith("index\n")
    assert benchmark.read_single_json_gz(gzip_path)["index"] in range(32)
    assert list(tmp_path.glob(".*.tmp")) == []


def test_prepare_tasks_uses_order_independent_seed_on_cpu(monkeypatch):
    samples = [
        SimpleNamespace(sample_id="b", problem="second", ground_truth="2"),
        SimpleNamespace(sample_id="a", problem="first", ground_truth="1"),
    ]
    monkeypatch.setattr(benchmark, "load_benchmark_samples", lambda *args, **kwargs: list(samples))

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            del messages, kwargs
            return [10, 11]

    args = SimpleNamespace(
        benchmark="math-ai/aime24",
        dataset_root="unused",
        max_samples_per_benchmark=-1,
        max_prompt_tokens=10,
        rollout_n=2,
        base_seed=42,
        enable_thinking=True,
    )
    _, tasks, _ = benchmark.prepare_tasks(args, FakeTokenizer())
    seeds = {(task.sample_id, task.rollout_index): task.seed for task in tasks}

    samples.reverse()
    _, reordered_tasks, _ = benchmark.prepare_tasks(args, FakeTokenizer())
    reordered_seeds = {(task.sample_id, task.rollout_index): task.seed for task in reordered_tasks}

    assert seeds == reordered_seeds
    assert seeds[("a", 0)] == benchmark.stable_seed(42, "math-ai/aime24", "a", 0)


def test_apply_chat_template_accepts_transformers_mapping_result_on_cpu():
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            del messages, kwargs
            return {"input_ids": [[10, 11, 12]]}

    assert benchmark.apply_chat_template(
        FakeTokenizer(),
        [{"role": "user", "content": "question"}],
        enable_thinking=True,
    ) == [10, 11, 12]


def test_validate_hf_checkpoint_requires_every_indexed_shard_and_tokenizer_on_cpu(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"first")

    ready, reason = benchmark.validate_hf_checkpoint(model_dir)
    assert ready is False
    assert "model-00002-of-00002.safetensors" in reason

    (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"second")
    assert benchmark.validate_hf_checkpoint(model_dir) == (True, "ready")


def test_stable_input_manifest_ignores_weight_mtime_but_detects_size_change_on_cpu(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"0123456789")
    descriptor = benchmark._file_descriptor(weight, content_hash=False, logical_path=weight.name)
    manifest = {
        "schema_version": benchmark.INPUT_MANIFEST_SCHEMA_VERSION,
        "student_model_identity": {"kind": "committed_checkpoint", "fingerprint": "source-step-20"},
        "dataset": [],
        "student_model_metadata": [],
        "student_model_weights": [descriptor],
        "student_tokenizer": [],
        "chat_template": [],
    }
    manifest["fingerprint"] = benchmark.stable_input_manifest_fingerprint(manifest)

    original_mtime = weight.stat().st_mtime_ns
    os.utime(weight, ns=(original_mtime + 1_000_000_000, original_mtime + 1_000_000_000))
    assert benchmark.validate_input_manifest(manifest) == (True, "valid")

    weight.write_bytes(b"changed-size")
    valid, reason = benchmark.validate_input_manifest(manifest)
    assert valid is False
    assert "fingerprint changed" in reason


def test_direct_generation_weight_identity_hashes_content_on_cpu(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"first")
    first = benchmark._weight_content_identity([weight], tmp_path)

    weight.write_bytes(b"other")
    second = benchmark._weight_content_identity([weight], tmp_path)

    assert first["kind"] == "weight_content_sha256"
    assert first["fingerprint"] != second["fingerprint"]


def test_generation_fingerprint_is_independent_of_deployment_paths_on_cpu():
    preparation, _ = _generation_preparation()
    relocated_request = dict(preparation.request)
    relocated_request.update(
        {
            "dataset_root": "/alternate/datasets",
            "student_model_path": "/alternate/model",
            "student_tokenizer_model": "/alternate/model",
            "chat_template_model": "/alternate/teacher",
        }
    )
    relocated_tokenizer = dict(preparation.tokenizer_info)
    relocated_tokenizer["model_path"] = "/alternate/model"

    assert (
        benchmark.generation_fingerprint(
            request=relocated_request,
            input_manifest=preparation.input_manifest,
            dataset_fingerprint=preparation.dataset_fingerprint,
            tokenizer_info=relocated_tokenizer,
        )
        == preparation.fingerprint
    )


def test_legacy_generation_accepts_only_mtime_change_and_preserves_canonical_fingerprint_on_cpu(tmp_path):
    preparation, existing = _generation_preparation()
    assessment = benchmark.assess_generation_compatibility(existing, preparation)

    assert assessment["status"] == "legacy_mtime_only"
    assert assessment["effective_fingerprint"] == existing["generation_fingerprint"]
    assert assessment["mtime_changes"] == [
        {
            "path": "/models/step-20/model.safetensors",
            "size": 10,
            "existing_mtime_ns": 10,
            "current_mtime_ns": 20,
        }
    ]

    output_dir = tmp_path / "unit"
    output_dir.mkdir()
    benchmark.write_json_atomic(output_dir / benchmark.METADATA_FILENAME, existing)
    migrated = benchmark._apply_generation_compatibility_migration(
        output_dir,
        existing,
        preparation,
        assessment,
    )
    reloaded = json.loads((output_dir / benchmark.METADATA_FILENAME).read_text(encoding="utf-8"))
    assert migrated == reloaded
    assert reloaded["generation_fingerprint"] == existing["generation_fingerprint"]
    assert preparation.fingerprint in reloaded["generation_compatibility"]["compatible_fingerprints"]
    assert reloaded["input_manifest"]["fingerprint"] == preparation.input_manifest["fingerprint"]


def test_legacy_generation_rejects_real_weight_or_sampling_change_with_structured_diff_on_cpu():
    preparation, existing = _generation_preparation(weight_size=11)
    assessment = benchmark.assess_generation_compatibility(existing, preparation)
    assert assessment["status"] == "incompatible"
    weight_diff = assessment["differences"]["input_manifest_groups"]["student_model_weights"]
    assert weight_diff["changed"]["/models/step-20/model.safetensors"]["size"] == {
        "existing": 10,
        "current": 11,
    }

    sampling_preparation, sampling_existing = _generation_preparation()
    changed_request = dict(sampling_preparation.request)
    changed_request["temperature"] = 1.0
    changed_preparation = benchmark.GenerationPreparation(
        **{
            **sampling_preparation.__dict__,
            "request": changed_request,
            "fingerprint": "different",
        }
    )
    assessment = benchmark.assess_generation_compatibility(sampling_existing, changed_preparation)
    assert assessment["status"] == "incompatible"
    assert assessment["differences"]["request"]["temperature"] == {"existing": 0.7, "current": 1.0}


def test_evaluate_one_prefers_exact_vllm_token_ids_on_cpu():
    class FakeCompletions:
        async def create(self, **kwargs):
            assert kwargs["extra_body"]["return_token_ids"] is True
            choice = SimpleNamespace(
                text=r"final answer is \boxed{1}",
                finish_reason="stop",
                stop_reason=151645,
                token_ids=[101, 102, 151645],
            )
            return SimpleNamespace(choices=[choice], usage=SimpleNamespace(completion_tokens=3))

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [999]

    client = SimpleNamespace(completions=FakeCompletions())
    args = SimpleNamespace(
        served_model_name="student",
        max_tokens=32,
        temperature=0.7,
        top_p=0.95,
        top_k=-1,
        require_exact_token_ids=True,
        score_student_logprobs=False,
        verifier="math_dapo",
        repetition_tail_chars=100,
        repetition_threshold=10.0,
        repetition_tail_token_window=32,
        ngram_size=4,
    )
    task = benchmark.RolloutTask(
        flat_index=0,
        task_id="task",
        benchmark_id="bench",
        benchmark_name="aime24",
        benchmark_path="math-ai/aime24",
        question_index=0,
        sample_id="0",
        problem="1+0?",
        ground_truth="1",
        prompt_token_ids=[1, 2],
        rollout_index=0,
        seed=42,
    )

    record = asyncio.run(
        benchmark.evaluate_one(
            client=client,
            generation_semaphore=asyncio.Semaphore(1),
            scoring_semaphore=asyncio.Semaphore(1),
            args=args,
            tokenizer=FakeTokenizer(),
            task=task,
            fingerprint="fingerprint",
        )
    )

    assert record["error"] is None
    assert record["response_token_ids"] == [101, 102, 151645]
    assert record["response_token_ids_source"] == "vllm"


def test_evaluate_one_reuses_generation_logprobs_without_echo_forward_on_cpu():
    requests: list[dict] = []

    class FakeCompletions:
        async def create(self, **kwargs):
            requests.append(kwargs)
            assert kwargs.get("echo") is None
            choice = SimpleNamespace(
                text=r"final answer is \boxed{1}",
                finish_reason="stop",
                stop_reason=2,
                token_ids=[101, 102],
                logprobs=SimpleNamespace(token_logprobs=[-0.2, -0.3]),
            )
            return SimpleNamespace(choices=[choice], usage=SimpleNamespace(completion_tokens=2))

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            del text, add_special_tokens
            return [999]

    args = SimpleNamespace(
        served_model_name="student",
        max_tokens=32,
        temperature=0.7,
        top_p=0.95,
        top_k=-1,
        require_exact_token_ids=True,
        score_student_logprobs=True,
        verifier="math_dapo",
        repetition_tail_chars=100,
        repetition_threshold=10.0,
        repetition_tail_token_window=32,
        ngram_size=4,
    )
    task = benchmark.RolloutTask(
        flat_index=0,
        task_id="task",
        benchmark_id="bench",
        benchmark_name="aime24",
        benchmark_path="math-ai/aime24",
        question_index=0,
        sample_id="0",
        problem="1+0?",
        ground_truth="1",
        prompt_token_ids=[1, 2],
        rollout_index=0,
        seed=42,
    )
    record = asyncio.run(
        benchmark.evaluate_one(
            client=SimpleNamespace(completions=FakeCompletions()),
            generation_semaphore=asyncio.Semaphore(1),
            scoring_semaphore=asyncio.Semaphore(1),
            args=args,
            tokenizer=FakeTokenizer(),
            task=task,
            fingerprint="fingerprint",
        )
    )

    assert len(requests) == 1
    assert requests[0]["logprobs"] == 1
    assert record["student_logprobs"] == [-0.2, -0.3]
    assert record["student_logprobs_source"] == "generation_response"


def _write_complete_unit(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True)
    empty_manifest = {"fingerprint": benchmark.payload_fingerprint({})}
    metadata = {
        "schema_version": benchmark.SCHEMA_VERSION,
        "experiment": "exp",
        "global_step": 20,
        "benchmark": {
            "id": benchmark.benchmark_id("math-ai/aime24"),
            "name": "aime24",
            "path": "math-ai/aime24",
            "dataset_fingerprint": "dataset",
        },
        "generation_complete": True,
        "generation_fingerprint": "generation",
        "rollout_count": 1,
        "request": {"rollout_n": 1},
        "analysis": {},
        "input_manifest": empty_manifest,
    }
    benchmark.write_json_atomic(output_dir / benchmark.METADATA_FILENAME, metadata)
    benchmark.write_jsonl_gz_atomic(output_dir / benchmark.ROLLOUTS_FILENAME, [])
    benchmark.write_json_atomic(output_dir / benchmark.SUMMARY_FILENAME, {})
    benchmark.write_csv_atomic(output_dir / benchmark.PER_PROBLEM_FILENAME, [], ["sample_id"])
    benchmark.write_csv_atomic(output_dir / benchmark.LENGTH_BINS_FILENAME, [], ["length_bin"])
    benchmark.write_completion_marker(output_dir, metadata)
    return metadata


def test_completed_unit_checks_expectations_and_result_sizes_on_cpu(tmp_path):
    output_dir = tmp_path / "unit"
    _write_complete_unit(output_dir)

    assert benchmark.validate_completed_unit(output_dir, expectations={"rollout_n": 1}) == (True, "complete")
    valid, reason = benchmark.validate_completed_unit(output_dir, expectations={"rollout_n": 2})
    assert valid is False
    assert "configuration changed" in reason

    with (output_dir / benchmark.SUMMARY_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write("changed")
    valid, reason = benchmark.validate_completed_unit(output_dir)
    assert valid is False
    assert "size changed" in reason


def test_lightweight_watcher_status_matches_unit_layout_and_checks_repair_and_checksum_on_cpu(tmp_path):
    output_dir = benchmark.unit_output_dir(tmp_path, "math-ai/aime24", 20)
    metadata = _write_complete_unit(output_dir)

    assert opd_result_status.unit_output_dir(tmp_path, "math-ai/aime24", 20) == output_dir
    assert opd_result_status.validate_completed_unit(output_dir, require_repair=False) == (True, "complete")
    valid, reason = opd_result_status.validate_completed_unit(output_dir)
    assert valid is False
    assert reason == "paired repair is incomplete"

    metadata["repair_complete"] = True
    benchmark.write_json_atomic(output_dir / benchmark.METADATA_FILENAME, metadata)
    benchmark.write_completion_marker(output_dir, metadata)
    assert opd_result_status.validate_completed_unit(output_dir) == (True, "complete")

    summary_path = output_dir / benchmark.SUMMARY_FILENAME
    assert summary_path.read_text(encoding="utf-8") == "{}\n"
    summary_path.write_text("[]\n", encoding="utf-8")
    valid, reason = opd_result_status.validate_completed_unit(output_dir)
    assert valid is False
    assert reason == "result file checksum changed: summary.json"


def test_completed_unit_normalizes_expected_paths_on_cpu(tmp_path, monkeypatch):
    output_dir = tmp_path / "unit"
    metadata = _write_complete_unit(output_dir)
    metadata["request"].update(
        {
            "benchmark": "math-ai/aime24",
            "dataset_root": str(tmp_path / "datasets"),
        }
    )
    benchmark.write_json_atomic(output_dir / benchmark.METADATA_FILENAME, metadata)
    benchmark.write_completion_marker(output_dir, metadata)
    monkeypatch.chdir(tmp_path)

    assert benchmark.validate_completed_unit(
        output_dir,
        expectations={"benchmark": "math-ai//aime24", "dataset_root": "datasets"},
    ) == (True, "complete")


def test_configurable_repetition_metrics_have_stable_generic_key_on_cpu():
    metrics = benchmark.token_repetition_metrics(
        [1, 2, 1, 2, 1, 2],
        tail_token_window=6,
        threshold=100.0,
        ngram_size=2,
    )

    assert metrics["ngram_size"] == 2
    assert metrics["tail_ngram_repeat_fraction"] == pytest.approx(0.6)
    assert "tail_4gram_repeat_fraction" not in metrics


def test_paired_bootstrap_reports_randomization_p_value_on_cpu():
    result = benchmark.paired_bootstrap([(0.0, 1.0), (0.0, 1.0), (1.0, 1.0)], resamples=200, seed=7)

    assert result["difference_right_minus_left"] == pytest.approx(2 / 3)
    assert 0.0 < result["paired_randomization_p"] <= 1.0
    assert "two_sided_bootstrap_p" not in result


def test_status_parser_supports_teacher_completion_requirement_on_cpu():
    args = benchmark.build_parser().parse_args(
        [
            "status",
            "--result-root",
            "/tmp/results",
            "--benchmark",
            "math-ai/aime24",
            "--global-step",
            "20",
            "--require-teacher",
        ]
    )

    assert args.require_teacher is True


def test_generate_preflight_only_does_not_contact_model_server_on_cpu(tmp_path, monkeypatch, capsys):
    preparation, _ = _generation_preparation()
    monkeypatch.setattr(benchmark, "prepare_generation", lambda args: preparation)
    result_root = tmp_path / "results"
    args = _preflight_args(result_root)

    assert asyncio.run(benchmark.generate_main(args)) == 0
    output = capsys.readouterr().out
    assert '"event":"generation_preflight"' in output
    assert '"status":"new"' in output
    assert not (benchmark.unit_output_dir(result_root, "math-ai/aime24", 20) / "metadata.json").exists()


def test_generate_preflight_migrates_legacy_mtime_without_changing_canonical_fingerprint_on_cpu(
    tmp_path,
    monkeypatch,
):
    preparation, existing = _generation_preparation()
    result_root = tmp_path / "results"
    output_dir = benchmark.unit_output_dir(result_root, "math-ai/aime24", 20)
    output_dir.mkdir(parents=True)
    benchmark.write_json_atomic(output_dir / benchmark.METADATA_FILENAME, existing)
    monkeypatch.setattr(benchmark, "prepare_generation", lambda args: preparation)

    assert asyncio.run(benchmark.generate_main(_preflight_args(result_root))) == 0

    migrated = json.loads((output_dir / benchmark.METADATA_FILENAME).read_text(encoding="utf-8"))
    assert migrated["generation_fingerprint"] == existing["generation_fingerprint"]
    assert preparation.fingerprint in migrated["generation_compatibility"]["compatible_fingerprints"]


def test_generate_preflight_reports_real_input_difference_before_model_server_on_cpu(tmp_path, monkeypatch):
    preparation, existing = _generation_preparation(weight_size=11)
    result_root = tmp_path / "results"
    output_dir = benchmark.unit_output_dir(result_root, "math-ai/aime24", 20)
    output_dir.mkdir(parents=True)
    benchmark.write_json_atomic(output_dir / benchmark.METADATA_FILENAME, existing)
    monkeypatch.setattr(benchmark, "prepare_generation", lambda args: preparation)

    with pytest.raises(RuntimeError, match='"student_model_weights"') as exc_info:
        asyncio.run(benchmark.generate_main(_preflight_args(result_root)))

    assert '"existing": 10' in str(exc_info.value)
    assert '"current": 11' in str(exc_info.value)


def test_migrate_legacy_splits_benchmarks_without_deleting_source_on_cpu(tmp_path):
    result_root = tmp_path / "experiment"
    legacy_dir = result_root / "global_step_20"
    legacy_dir.mkdir(parents=True)
    legacy_metadata = {
        "schema_version": 1,
        "experiment": "experiment",
        "global_step": 20,
        "generation_complete": True,
        "generation": {
            "benchmarks": ["math-ai/aime24", "math-ai/amc23"],
            "dataset_root": str(tmp_path / "datasets"),
            "rollout_n": 1,
        },
        "analysis": {},
        "student_tokenizer": {"fingerprint_sha256": "tokenizer", "vocab_size": 10},
    }
    benchmark.write_json_atomic(legacy_dir / benchmark.METADATA_FILENAME, legacy_metadata)
    records = [
        {
            "benchmark": name,
            "question_index": 0,
            "sample_id": f"{name}-0",
            "rollout_index": 0,
            "problem": "1+1?",
            "ground_truth": "2",
            "acc": True,
            "response_tokens": 8,
            "repetition": {"tail_4gram_repeat_fraction": 0.0},
            "error": None,
        }
        for name in ("aime24", "amc23")
    ]
    benchmark.write_jsonl_gz_atomic(legacy_dir / benchmark.ROLLOUTS_FILENAME, records)

    assert benchmark.migrate_legacy_main(SimpleNamespace(result_root=str(result_root), overwrite=False)) == 0

    units = benchmark.discover_units(result_root)
    assert set(units) == {("math-ai/aime24", 20), ("math-ai/amc23", 20)}
    assert legacy_dir.is_dir()
    assert not (result_root / "all_checkpoints_summary.csv").exists()
    for unit_dir in units.values():
        metadata = json.loads((unit_dir / benchmark.METADATA_FILENAME).read_text(encoding="utf-8"))
        assert metadata["legacy_migrated"] is True
        assert metadata["analysis"]["length_bin_edges"] == benchmark.DEFAULT_LENGTH_BIN_EDGES
        assert benchmark.validate_completed_unit(unit_dir, allow_legacy=True) == (True, "complete")


def test_teacher_scoring_failure_preserves_generation_and_resumes_on_cpu(tmp_path, monkeypatch):
    result_root = tmp_path / "results"
    output_dir = benchmark.unit_output_dir(result_root, "math-ai/aime24", 20)
    metadata = _write_complete_unit(output_dir)

    class FakeTokenizer:
        chat_template = "template"
        bos_token_id = 0
        eos_token_id = 1
        pad_token_id = 0
        unk_token_id = None

        def get_vocab(self):
            return {"a": 0, "b": 1}

        def __len__(self):
            return 2

    tokenizer = FakeTokenizer()
    metadata["request"]["score_student_logprobs"] = True
    metadata["student_tokenizer"] = benchmark.tokenizer_metadata(tokenizer, "student")
    metadata["analysis"] = benchmark._legacy_analysis({})
    record = {
        "schema_version": benchmark.SCHEMA_VERSION,
        "generation_fingerprint": "generation",
        "task_id": "task-0",
        "flat_index": 0,
        "benchmark": "aime24",
        "benchmark_path": "math-ai/aime24",
        "question_index": 0,
        "sample_id": "0",
        "rollout_index": 0,
        "problem_fingerprint": "problem",
        "prompt_token_ids": [1],
        "response_token_ids": [2, 3],
        "student_logprobs": [-0.2, -0.3],
        "response_tokens": 2,
        "acc": True,
        "repetition": {},
        "error": None,
    }
    benchmark.write_json_atomic(output_dir / benchmark.METADATA_FILENAME, metadata)
    benchmark.write_jsonl_gz_atomic(output_dir / benchmark.ROLLOUTS_FILENAME, [record])
    benchmark.write_completion_marker(output_dir, metadata)

    state = {"fail": True}

    class FakeModels:
        async def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id="teacher")])

    class FakeCompletions:
        async def create(self, **kwargs):
            del kwargs
            if state["fail"]:
                raise RuntimeError("temporary server failure")
            logprobs = SimpleNamespace(token_logprobs=[None, -0.1, -0.2])
            return SimpleNamespace(choices=[SimpleNamespace(logprobs=logprobs)])

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            del kwargs
            self.models = FakeModels()
            self.completions = FakeCompletions()

        async def close(self):
            return None

    monkeypatch.setattr(benchmark.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(
        benchmark,
        "capture_model_manifest",
        lambda path: {"fingerprint": benchmark.payload_fingerprint({})},
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    args = SimpleNamespace(
        result_root=str(result_root),
        benchmark="math-ai/aime24",
        global_step=20,
        score_max_in_flight=1,
        score_batch_size=1,
        served_model_name="teacher",
        teacher_model_path="teacher",
        teacher_tokenizer_model="teacher",
        trust_remote_code=False,
        allow_tokenizer_mismatch=False,
        high_gap_threshold=1.0,
        repetition_tail_token_window=4096,
        bootstrap_resamples=0,
        bootstrap_seed=7,
        overwrite=False,
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        request_timeout_seconds=10.0,
        max_retries=0,
        print_every=0,
    )

    assert asyncio.run(benchmark.teacher_score_main(args)) == 1
    assert benchmark.validate_completed_unit(output_dir) == (True, "complete")
    assert (output_dir / benchmark.WORK_DIRNAME / "teacher-state.json").is_file()

    state["fail"] = False
    assert asyncio.run(benchmark.teacher_score_main(args)) == 0
    completed_metadata = json.loads((output_dir / benchmark.METADATA_FILENAME).read_text(encoding="utf-8"))
    assert completed_metadata["teacher_scored"] is True
    assert completed_metadata["teacher_scoring_complete"] is True
    assert not (output_dir / benchmark.WORK_DIRNAME).exists()
    assert benchmark.validate_completed_unit(output_dir) == (True, "complete")


def test_repair_failure_preserves_raw_unit_and_resumes_only_missing_shards_on_cpu(tmp_path, monkeypatch):
    checkpoint_root = tmp_path / "checkpoints"
    config = OmegaConf.create(
        {
            "trainer": {
                "experiment_name": "exp",
                "project_name": "DMPD",
                "test_benchmarks_max_response_length": 32,
            },
            "actor_rollout_ref": {
                "model": {"path": "student", "tokenizer_path": None, "chat_template_from_model": "teacher"},
                "rollout": {
                    "max_model_len": 64,
                    "test_kwargs": {"n": 1, "temperature": 0.7, "top_p": 0.95, "top_k": -1},
                },
            },
            "data": {
                "trust_remote_code": False,
                "max_prompt_length": 32,
                "max_response_length": 32,
                "seed": 42,
                "test_benchmarks": {
                    "root": "datasets",
                    "names": ["math-ai/aime24"],
                    "reward_data_source": "math_dapo",
                },
            },
            "reward": {"reward_manager": {"name": "naive"}},
            "distillation": {
                "enabled": True,
                "teacher_key": "default",
                "teacher_models": {"default": {"model_path": "teacher"}},
                "distillation_loss": {"log_prob_min_clamp": -10.0},
                "intervention": {
                    "_target_": "verl.workers.config.DistillationInterventionConfig",
                    "enabled": True,
                    "protect_thinking_blocks": False,
                    "gap_metric": "k3",
                },
            },
        }
    )
    spec = write_evaluation_spec(checkpoint_root, config, training_run_id="training-id")
    checkpoint_dir = checkpoint_root / "global_step_20"
    write_evaluation_state(
        checkpoint_dir,
        checkpoint_step=20,
        spec_fingerprint=spec["spec_fingerprint"],
        teacher_nll_calibration=None,
    )

    class FakeTokenizer:
        chat_template = "template"
        bos_token_id = 0
        eos_token_id = 1
        pad_token_id = 0
        unk_token_id = None

        def get_vocab(self):
            return {"a": 0, "b": 1, "c": 2, "d": 3}

        def __len__(self):
            return 4

        def decode(self, token_ids, skip_special_tokens=True):
            del token_ids, skip_special_tokens
            return r"final answer is \boxed{1}"

    tokenizer = FakeTokenizer()
    result_root = tmp_path / "results"
    output_dir = benchmark.unit_output_dir(result_root, "math-ai/aime24", 20)
    metadata = _write_complete_unit(output_dir)
    metadata.update(
        {
            "experiment": "exp",
            "student_tokenizer": benchmark.tokenizer_metadata(tokenizer, "student"),
            "request": {
                "rollout_n": 1,
                "score_student_logprobs": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": -1,
                "max_tokens": 32,
                "verifier": "math_dapo",
            },
            "analysis": benchmark._legacy_analysis({}),
        }
    )
    records = []
    for index, task_id in enumerate(("task-a", "task-b")):
        records.append(
            {
                "schema_version": benchmark.SCHEMA_VERSION,
                "generation_fingerprint": "generation",
                "task_id": task_id,
                "flat_index": index,
                "benchmark": "aime24",
                "benchmark_path": "math-ai/aime24",
                "question_index": index,
                "sample_id": str(index),
                "rollout_index": 0,
                "problem_fingerprint": f"problem-{index}",
                "prompt_token_ids": [1],
                "response_token_ids": [2, 3],
                "student_logprobs": [-0.2, -0.3],
                "response_tokens": 2,
                "ground_truth": "1",
                "finish_reason": "stop",
                "acc": False,
                "repetition": {},
                "error": None,
            }
        )
    benchmark.write_json_atomic(output_dir / benchmark.METADATA_FILENAME, metadata)
    benchmark.write_jsonl_gz_atomic(output_dir / benchmark.ROLLOUTS_FILENAME, records)
    benchmark.write_completion_marker(output_dir, metadata)

    class FakeModels:
        async def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id="student"), SimpleNamespace(id="teacher")])

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            del kwargs
            self.models = FakeModels()

        async def close(self):
            return None

    class FakeAdapter:
        def __init__(self, **kwargs):
            del kwargs

    class FakeAdapterConfig:
        def __init__(self, **kwargs):
            del kwargs

    class FakeRunner:
        def __init__(self, **kwargs):
            del kwargs

    failed_once = {"value": True}
    calls: list[str] = []

    async def fake_repair_one_record(**kwargs):
        record = dict(kwargs["record"])
        task_id = record["task_id"]
        calls.append(task_id)
        if task_id == "task-b" and failed_once["value"]:
            record["repair"] = {"fingerprint": kwargs["repair_fingerprint"], "error": "temporary failure"}
            return record
        record["repair"] = {
            "fingerprint": kwargs["repair_fingerprint"],
            "text": r"final answer is \boxed{1}",
            "response_token_ids": [2, 3],
            "response_tokens": 2,
            "finish_reason": "stop",
            "eos_success": True,
            "truncated": False,
            "acc": True,
            "pred": "1",
            "answer_extractable": True,
            "has_answer_pattern": True,
            "has_boxed_pattern": True,
            "accepted_answer_format": True,
            "repetition": {},
            "trigger_detected": True,
            "repair_executed": True,
            "intervention_count": 1,
            "first_intervention_position": 0,
            "teacher_interventions": [],
            "teacher_intervention_skipped": [],
            "gap_diagnostics": [],
            "error": None,
        }
        return record

    monkeypatch.setattr(benchmark.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(benchmark, "repair_one_record", fake_repair_one_record)
    monkeypatch.setattr(teacher_intervention, "OpenAICompletionAdapter", FakeAdapter)
    monkeypatch.setattr(teacher_intervention, "OpenAICompletionAdapterConfig", FakeAdapterConfig)
    monkeypatch.setattr(teacher_intervention, "StandaloneTeacherInterventionRunner", FakeRunner)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    args = SimpleNamespace(
        result_root=str(result_root),
        benchmark="math-ai/aime24",
        global_step=20,
        evaluation_spec=str(checkpoint_root / "evaluation_spec.json"),
        evaluation_state=str(checkpoint_dir / "evaluation_state.json"),
        student_base_url="http://student/v1",
        student_api_key="EMPTY",
        student_served_model_name="student",
        teacher_base_url="http://teacher/v1",
        teacher_api_key="EMPTY",
        teacher_served_model_name="teacher",
        teacher_model_fingerprint="teacher-fingerprint",
        student_tokenizer_model="student",
        teacher_tokenizer_model="teacher",
        trust_remote_code=False,
        allow_tokenizer_mismatch=False,
        request_timeout_seconds=10,
        max_retries=0,
        max_in_flight=2,
        max_pending=2,
        print_every=0,
        overwrite=False,
    )

    assert asyncio.run(benchmark.repair_main(args)) == 1
    assert benchmark.validate_completed_unit(output_dir) == (True, "complete")
    assert calls.count("task-a") == 1
    assert calls.count("task-b") == 1

    failed_once["value"] = False
    assert asyncio.run(benchmark.repair_main(args)) == 0
    assert calls.count("task-a") == 1
    assert calls.count("task-b") == 2
    completed_metadata = json.loads((output_dir / benchmark.METADATA_FILENAME).read_text(encoding="utf-8"))
    assert completed_metadata["repair_complete"] is True
    assert benchmark.validate_completed_unit(output_dir) == (True, "complete")
