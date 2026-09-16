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

from __future__ import annotations

import builtins
import io
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.evaluation import standalone_math as evaluation
from verl.utils.evaluation.math_benchmark import EvalSample, build_math_chat_messages
from verl.utils.reward_score import default_compute_score
from verl.utils.tokenizer import normalize_token_ids

REPO_ROOT = Path(__file__).resolve().parents[3]
QWEN_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] "
    "+ '<|im_end|>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}"
    "{% if enable_thinking is defined and not enable_thinking %}{{ '<think>\n\n</think>\n\n' }}"
    "{% endif %}{% endif %}"
)
DEEPSEEK_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}{{ '<｜User｜>' + message['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<｜Assistant｜><think>\n' }}{% endif %}"
)


@pytest.fixture(autouse=True)
def _restore_runtime_settings():
    names = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "VLLM_NO_USAGE_STATS",
        "PYTHONDONTWRITEBYTECODE",
    )
    previous = {name: os.environ.get(name) for name in names}
    previous_bytecode = sys.dont_write_bytecode
    yield
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    sys.dont_write_bytecode = previous_bytecode


def _save_tokenizer(path: Path, template=QWEN_TEMPLATE):
    vocabulary = ["[UNK]", "[BOS]", "[PAD]", "user", "assistant", "Question", "0", "1", r"\boxed{4}"]
    backend = Tokenizer(models.WordLevel(dict(zip(vocabulary, range(len(vocabulary)), strict=True)), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        bos_token="[BOS]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<think>", "</think>", "<｜User｜>", "<｜Assistant｜>"],
        chat_template=template,
    )
    tokenizer.save_pretrained(path)
    return tokenizer


@pytest.fixture
def inputs(tmp_path):
    model_path = tmp_path / "HF checkpoint"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"test weights; the inference engine is mocked")
    tokenizer = _save_tokenizer(model_path)
    for benchmark in evaluation.BENCHMARKS:
        data_path = tmp_path / "dataset" / benchmark / "data"
        data_path.mkdir(parents=True)
        rows = [{"problem_idx": i + 1, "problem": f"Question {i}", "answer": "4"} for i in range(2)]
        if benchmark == "hmmt_feb_2026":
            for row in rows:
                row["problem_type"] = ["Algebra"]
        pd.DataFrame(rows).to_parquet(data_path / "train-00000-of-00001.parquet")
    config = evaluation.EvaluationConfig(str(model_path), str(tmp_path / "dataset"))
    return config, tokenizer


def test_legacy_defaults_match_training_profile_on_cpu():
    config = evaluation.EvaluationConfig.from_environment({"MODEL_PATH": "/local/checkpoint", "ROLLOUT_N": "4"})
    legacy = OmegaConf.load(REPO_ROOT / "verl/trainer/config/rollout/rollout.yaml").benchmark_sampling_profiles.legacy
    for key in ("temperature", "top_p", "top_k", "do_sample"):
        assert getattr(config, key) == legacy[key]
    assert config.rollout_n == legacy.n == 16
    assert (config.prompt_length, config.response_length, config.generation_max_tokens) == (1024, 23552, 23552)
    assert config.max_model_len == 24577
    assert config.dataset_root == str(REPO_ROOT / "dataset")
    assert config.seed == 0
    assert not config.force_merge


def test_existing_environment_overrides_and_context_precedence_on_cpu():
    env = {
        "MODEL_PATH": "/local/checkpoint",
        "TEST_TEMPERATURE": "0.9",
        "TEST_TOP_P": "0.8",
        "ROLLOUT_TOP_K": "20",
        "TEST_N": "3",
        "TEST_BENCHMARK_N": "5",
        "TEST_DO_SAMPLE": "false",
        "ENABLE_THINKING": "0",
        "MAX_PROMPT_LENGTH": "512",
        "MAX_RESPONSE_LENGTH": "512",
        "TEST_BENCHMARKS_MAX_RESPONSE_LENGTH": "128",
        "MAX_NUM_TOKENS": "4096",
        "MAX_MODEL_LEN": "3072",
        "STUDENT_MAX_MODEL_LEN": "2048",
        "ROLLOUT_MAX_MODEL_LEN": "1024",
        "STUDENT_TP": "1",
        "STUDENT_GPU_MEM_UTIL": "0.8",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS": "256",
        "ROLLOUT_MAX_NUM_SEQS": "4",
        "TEST_BENCHMARK_BATCH_SIZE": "2",
        "TEST_BENCHMARK_MAX_SAMPLES": "1",
        "EVAL_SEED": "17",
        "ROLLOUT_ENFORCE_EAGER": "True",
        "EVAL_FORCE_MERGE": "True",
    }
    config = evaluation.EvaluationConfig.from_environment(env)
    assert (config.temperature, config.top_p, config.top_k, config.rollout_n) == (0.9, 0.8, 20, 5)
    assert not config.do_sample and not config.enable_thinking
    assert (config.max_num_batched_tokens, config.max_num_seqs, config.batch_size, config.max_samples) == (256, 4, 2, 1)
    assert (config.tensor_parallel_size, config.gpu_memory_utilization, config.seed) == (1, 0.8, 17)
    assert config.enforce_eager
    assert config.force_merge
    for name, expected in [
        ("ROLLOUT_MAX_MODEL_LEN", 1024),
        ("STUDENT_MAX_MODEL_LEN", 2048),
        ("MAX_MODEL_LEN", 3072),
        ("MAX_NUM_TOKENS", 4096),
    ]:
        assert evaluation.EvaluationConfig.from_environment(env).max_model_len == expected
        env.pop(name)
    assert evaluation.EvaluationConfig.from_environment(env).max_model_len == 1025
    env["TEST_BENCHMARK_N"] = ""
    assert evaluation.EvaluationConfig.from_environment(env).rollout_n == 3


@pytest.mark.parametrize(
    "override",
    [
        {"TEST_TEMPERATURE": "nan"},
        {"TEST_TOP_P": "0"},
        {"TEST_TOP_K": "0"},
        {"TEST_BENCHMARK_N": "0"},
        {"TEST_DO_SAMPLE": "sometimes"},
        {"ENABLE_THINKING": "sometimes"},
        {"EVAL_FORCE_MERGE": "sometimes"},
        {"STUDENT_TP": "0"},
        {"STUDENT_GPU_MEM_UTIL": "2"},
        {"TEST_BENCHMARK_MAX_SAMPLES": "0"},
        {"EVAL_SEED": "-1"},
        {"MAX_RESPONSE_LENGTH": "32", "TEST_BENCHMARKS_MAX_RESPONSE_LENGTH": "64"},
        {"ENABLE_TOOL_CALL": "True"},
        {"TEST_BENCHMARK_REWARD_DATA_SOURCE": "math_dapo"},
    ],
)
def test_invalid_configuration_fails_on_cpu(override):
    with pytest.raises(ValueError):
        evaluation.EvaluationConfig.from_environment({"MODEL_PATH": "/local/checkpoint", **override})


def test_missing_model_environment_fails_on_cpu():
    with pytest.raises(ValueError, match="MODEL_PATH"):
        evaluation.EvaluationConfig.from_environment({})


def test_checkpoint_validation_checks_every_indexed_shard_on_cpu(inputs):
    config, _ = inputs
    root = Path(config.model_path)
    evaluation.validate_checkpoint(config.model_path)
    (root / "model.safetensors").unlink()
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "part1.safetensors", "y": "part2.safetensors"}}))
    (root / "part1.safetensors").write_bytes(b"1")
    with pytest.raises(ValueError, match="part2.safetensors"):
        evaluation.validate_checkpoint(config.model_path)
    (root / "part2.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        evaluation.validate_checkpoint(config.model_path)
    (root / "part2.safetensors").write_bytes(b"2")
    evaluation.validate_checkpoint(config.model_path)
    index.write_text(json.dumps({"weight_map": {"x": ["bad shard"]}}))
    with pytest.raises(ValueError, match="Invalid local shard"):
        evaluation.validate_checkpoint(config.model_path)


def _fsdp_layout(inputs, world_size=32):
    config, tokenizer = inputs
    step = Path(config.model_path).parent / "experiment with spaces" / "global_step_217"
    actor = step / "actor"
    actor.mkdir(parents=True)
    hf_dir = actor / "huggingface"
    Path(config.model_path).rename(hf_dir)
    (hf_dir / "model.safetensors").unlink()
    (actor / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 2, "world_size": world_size}))
    for rank in range(world_size):
        (actor / f"model_world_size_{world_size}_rank_{rank}.pt").write_bytes(b"mock model shard")
    return replace(config, model_path=str(step)), tokenizer, actor, hf_dir


def _snapshot(path):
    return {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()}


@pytest.mark.parametrize("path_kind", ["step", "actor", "huggingface", "relative"])
def test_fsdp_input_paths_merge_once_then_reuse_without_source_shards_on_cpu(inputs, monkeypatch, path_kind, capsys):
    config, _, actor, hf_dir = _fsdp_layout(inputs)
    paths = {"step": actor.parent, "actor": actor, "huggingface": hf_dir}
    if path_kind == "relative":
        monkeypatch.chdir(actor.parent.parent)
        path = "global_step_217/actor/huggingface"
    else:
        path = str(paths[path_kind])
    calls = []
    source_before = {file.name: file.read_bytes() for file in actor.iterdir() if file.is_file()}

    def merge(command, **kwargs):
        calls.append(command)
        assert command == [
            sys.executable,
            "-B",
            "-u",
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(actor),
            "--target_dir",
            str(hf_dir),
        ]
        assert set(kwargs) == {"check", "env"} and kwargs["check"] is True
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONDONTWRITEBYTECODE"):
            assert kwargs["env"][name] == "1"
        (hf_dir / "model.safetensors").write_bytes(b"merged mock weights")

    monkeypatch.setattr(evaluation.subprocess, "run", merge)
    assert evaluation.prepare_checkpoint(path) == str(hf_dir)
    assert source_before == {file.name: file.read_bytes() for file in actor.iterdir() if file.is_file()}
    for shard in actor.glob("model_*.pt"):
        shard.unlink()
    before = _snapshot(actor)
    assert evaluation.prepare_checkpoint(path) == str(hf_dir)
    assert before == _snapshot(actor) and len(calls) == 1
    assert "Reusing complete HF weights" in capsys.readouterr().out


def test_force_merge_removes_only_standard_hf_weights_on_cpu(inputs, monkeypatch):
    _, _, actor, hf_dir = _fsdp_layout(inputs)
    weight_names = list(evaluation.HF_WEIGHT_FILES) + [
        "model-00001-of-00002.safetensors",
        "pytorch_model-00001-of-00002.bin",
    ]
    for name in weight_names:
        (hf_dir / name).write_bytes(b"old weights")
    (hf_dir / "model_notes.txt").write_text("keep unrelated files")
    (hf_dir / "adapter_model.safetensors").write_bytes(b"keep unrelated adapter")
    (actor / "optim_world_size_32_rank_0.pt").write_bytes(b"keep optimizer")
    before = _snapshot(actor)

    def merge(*args, **kwargs):
        assert all(not (hf_dir / name).exists() for name in weight_names)
        current = _snapshot(actor)
        for name, contents in before.items():
            if name not in {f"huggingface/{weight}" for weight in weight_names}:
                assert current[name] == contents
        (hf_dir / "model.safetensors").write_bytes(b"new weights")

    monkeypatch.setattr(evaluation.subprocess, "run", merge)
    assert evaluation.prepare_checkpoint(str(actor), force_merge=True) == str(hf_dir)
    assert (hf_dir / "model.safetensors").read_bytes() == b"new weights"


def test_plain_hf_force_merge_and_experiment_root_are_rejected_on_cpu(inputs, monkeypatch):
    config, _ = inputs
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected merger"))
    before = _snapshot(Path(config.model_path))
    with pytest.raises(ValueError, match="corresponding FSDP actor"):
        evaluation.prepare_checkpoint(config.model_path, force_merge=True)
    assert _snapshot(Path(config.model_path)) == before
    _, _, actor, _ = _fsdp_layout(inputs)
    with pytest.raises(ValueError, match="specific global_step"):
        evaluation.prepare_checkpoint(str(actor.parent.parent))


def test_incomplete_hf_index_triggers_merge_without_force_on_cpu(inputs, monkeypatch):
    config, _, _, hf_dir = _fsdp_layout(inputs)
    index = hf_dir / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "model-00001-of-00002.safetensors"}}))
    calls = []

    def merge(*args, **kwargs):
        calls.append(args)
        assert not index.exists()
        (hf_dir / "model.safetensors").write_bytes(b"complete mock weights")

    monkeypatch.setattr(evaluation.subprocess, "run", merge)
    assert evaluation.prepare_checkpoint(config.model_path) == str(hf_dir)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "missing_shard",
        "empty_shard",
        "missing_fsdp_config",
        "bad_json",
        "bad_world_size",
        "bool_world_size",
        "missing_hf_config",
        "missing_tokenizer",
        "readonly",
    ],
)
def test_bad_merge_source_preserves_existing_weights_and_stops_evaluation_on_cpu(inputs, monkeypatch, capsys, fault):
    config, _, actor, hf_dir = _fsdp_layout(inputs)
    (hf_dir / "model.safetensors").write_bytes(b"old complete weights")
    if fault == "missing_shard":
        (actor / "model_world_size_32_rank_31.pt").unlink()
    elif fault == "empty_shard":
        (actor / "model_world_size_32_rank_31.pt").write_bytes(b"")
    elif fault == "missing_fsdp_config":
        (actor / "fsdp_config.json").unlink()
    elif fault == "bad_json":
        (actor / "fsdp_config.json").write_text("{")
    elif fault in {"bad_world_size", "bool_world_size"}:
        (actor / "fsdp_config.json").write_text(json.dumps({"world_size": 0 if fault == "bad_world_size" else True}))
    elif fault == "missing_hf_config":
        (hf_dir / "config.json").unlink()
    elif fault == "missing_tokenizer":
        (hf_dir / "tokenizer.json").unlink()
    elif fault == "readonly":
        monkeypatch.setattr(evaluation.os, "access", lambda *a: False)
    config = replace(config, force_merge=True)
    monkeypatch.setattr(evaluation.EvaluationConfig, "from_environment", lambda: config)
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected merger"))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=lambda **k: pytest.fail("Unexpected inference")))
    before = _snapshot(actor)
    assert evaluation.main() == 1
    assert _snapshot(actor) == before
    output = capsys.readouterr()
    assert "[eval] ERROR:" in output.err
    assert "EVAL_RESULT_JSON=" not in output.out and "[result]" not in output.out


@pytest.mark.parametrize("failure", ["exit", "incomplete_weights", "invalid_index"])
def test_failed_conversion_removes_partial_weights_and_can_retry_on_cpu(inputs, monkeypatch, capsys, failure):
    config, _, actor, hf_dir = _fsdp_layout(inputs)
    source_before = _snapshot(actor)

    def fail(command, **kwargs):
        (hf_dir / "model-00001-of-00002.safetensors").write_bytes(b"partial weights")
        index = {"weight_map": {"x": "model-00001-of-00002.safetensors", "y": "model-00002-of-00002.safetensors"}}
        (hf_dir / "model.safetensors.index.json").write_text(json.dumps([] if failure == "invalid_index" else index))
        if failure == "exit":
            raise subprocess.CalledProcessError(37, command)

    monkeypatch.setattr(evaluation.subprocess, "run", fail)
    monkeypatch.setattr(evaluation.EvaluationConfig, "from_environment", lambda: config)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=lambda **k: pytest.fail("Unexpected inference")))
    assert evaluation.main() == 1
    assert _snapshot(actor) == source_before
    assert "EVAL_RESULT_JSON=" not in capsys.readouterr().out

    def succeed(*args, **kwargs):
        (hf_dir / "model.safetensors").write_bytes(b"complete mock weights")

    monkeypatch.setattr(evaluation.subprocess, "run", succeed)
    assert evaluation.prepare_checkpoint(config.model_path) == str(hf_dir)


@pytest.mark.parametrize("template", [QWEN_TEMPLATE, DEEPSEEK_TEMPLATE], ids=["qwen3", "deepseek"])
def test_real_qwen_conversion_preserves_weights_templates_and_eos_on_cpu(inputs, template):
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig, Qwen3Config, Qwen3ForCausalLM

    config, _, actor, hf_dir = _fsdp_layout(inputs, world_size=1)
    tokenizer = _save_tokenizer(hf_dir, template)
    model_config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=False,
        architectures=["Qwen3ForCausalLM"],
    )
    model = Qwen3ForCausalLM(model_config).to(torch.bfloat16)
    expected = model.state_dict()
    model.config.save_pretrained(hf_dir)
    model.generation_config.eos_token_id = [tokenizer.eos_token_id, tokenizer.bos_token_id]
    model.generation_config.save_pretrained(hf_dir)
    torch.save(expected, actor / "model_world_size_1_rank_0.pt")
    before = {file.name: file.read_bytes() for file in actor.iterdir() if file.is_file()}
    assert evaluation.prepare_checkpoint(config.model_path) == str(hf_dir)
    assert before == {file.name: file.read_bytes() for file in actor.iterdir() if file.is_file()}
    restored = AutoModelForCausalLM.from_pretrained(hf_dir, local_files_only=True, torch_dtype=torch.bfloat16)
    assert restored.state_dict().keys() == expected.keys()
    for name, weight in restored.state_dict().items():
        torch.testing.assert_close(weight, expected[name], rtol=0, atol=0)
    merged_tokenizer = evaluation.load_tokenizer(replace(config, model_path=str(hf_dir)))
    assert merged_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert merged_tokenizer.chat_template == tokenizer.chat_template
    assert merged_tokenizer.special_tokens_map == tokenizer.special_tokens_map
    assert GenerationConfig.from_pretrained(hf_dir).eos_token_id == model.generation_config.eos_token_id
    for thinking in (True, False):
        prepared = evaluation.prepare_samples(
            replace(config, model_path=str(hf_dir), enable_thinking=thinking), merged_tokenizer
        )
        for rows in prepared.values():
            for row in rows:
                original_ids = tokenizer.apply_chat_template(
                    build_math_chat_messages(row.sample.problem),
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
                assert row.prompt_ids == normalize_token_ids(original_ids)


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_prompt_token_ids_match_training_and_hmmt_schemas_on_cpu(inputs, enable_thinking):
    config, tokenizer = inputs
    config = replace(config, enable_thinking=enable_thinking)
    prepared = evaluation.prepare_samples(config, tokenizer)
    assert [len(rows) for rows in prepared.values()] == [2, 2]
    for benchmark, rows in prepared.items():
        for row in rows:
            expected = tokenizer.apply_chat_template(
                build_math_chat_messages(row.sample.problem),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            assert row.prompt_ids == normalize_token_ids(expected)
            assert row.uid == f"{benchmark}:{row.sample.sample_id}"
            assert row.max_tokens == config.generation_max_tokens
        assert (tokenizer.convert_tokens_to_ids("</think>") in rows[0].prompt_ids) is (not enable_thinking)


def test_external_deepseek_template_changes_only_template_on_cpu(inputs, tmp_path):
    config, original = inputs
    template_path = tmp_path / "template"
    _save_tokenizer(template_path, DEEPSEEK_TEMPLATE)
    tokenizer = evaluation.load_tokenizer(replace(config, chat_template_model=str(template_path)))
    assert tokenizer.get_vocab() == original.get_vocab()
    assert tokenizer.eos_token_id == original.eos_token_id
    rendered = tokenizer.apply_chat_template(
        build_math_chat_messages("Question 0"), tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    assert rendered.startswith("[BOS]<｜User｜>")
    assert rendered.endswith("<｜Assistant｜><think>\n")
    assert evaluation.load_tokenizer(config).chat_template == QWEN_TEMPLATE


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_cached_qwen3_template_has_exact_input_tokens_on_cpu(inputs, enable_thinking):
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3-0.6B tokenizer is not cached locally.")
    config, _ = inputs
    config = replace(config, enable_thinking=enable_thinking)
    row = evaluation.prepare_samples(config, tokenizer)[evaluation.BENCHMARKS[0]][0]
    rendered = tokenizer.apply_chat_template(
        build_math_chat_messages(row.sample.problem),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    assert row.prompt_ids == tokenizer.encode(rendered, add_special_tokens=False)
    assert rendered.endswith("<|im_start|>assistant\n" if enable_thinking else "<think>\n\n</think>\n\n")
    assert tokenizer.decode(row.prompt_ids, skip_special_tokens=False) == rendered


def test_prompt_budget_errors_and_generation_context_clamp_on_cpu(inputs):
    config, tokenizer = inputs
    rows = evaluation.prepare_samples(config, tokenizer)[evaluation.BENCHMARKS[0]]
    prompt_length = len(rows[0].prompt_ids)
    with pytest.raises(ValueError, match="prompt tokens"):
        evaluation.prepare_samples(replace(config, prompt_length=1), tokenizer)
    with pytest.raises(ValueError, match="No generation space"):
        evaluation.prepare_samples(replace(config, max_model_len=prompt_length), tokenizer)
    clamped = evaluation.prepare_samples(replace(config, max_model_len=prompt_length + 5), tokenizer)
    assert clamped[evaluation.BENCHMARKS[0]][0].max_tokens == 5


class _Engine:
    def __init__(self, tokenizer, failure=None):
        self.calls = []
        self.tokenizer = tokenizer
        self.failure = failure

    def generate(self, prompts, sampling_params, use_tqdm):
        self.calls.append((prompts, sampling_params))
        assert use_tqdm is False
        tokens = self.tokenizer.encode(r"\boxed{4}", add_special_tokens=False) + [self.tokenizer.eos_token_id]
        outputs = [
            SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                finished=True,
                outputs=[SimpleNamespace(token_ids=tokens, finish_reason="stop")],
            )
            for prompt in prompts
        ]
        if self.failure == "missing":
            outputs.pop()
        elif self.failure == "abort":
            outputs[0].outputs[0].finish_reason = "abort"
        elif self.failure == "mismatched":
            outputs[0].prompt_token_ids = [-1]
        elif self.failure == "unfinished":
            outputs[0].finished = False
        elif self.failure == "multiple":
            outputs[0].outputs *= 2
        return outputs


def _score(**kwargs):
    assert kwargs["solution_str"] == r"\boxed{4}"
    assert kwargs["extra_info"]["reward_data_source"] == "math_verify"
    return {"score": 1.0, "acc": True, "pred": "4"}


def test_requests_repeat_sixteen_times_and_preserve_eos_on_cpu(inputs, monkeypatch):
    config, tokenizer = inputs
    config = replace(config, batch_size=1)
    rows = evaluation.prepare_samples(config, tokenizer)[evaluation.BENCHMARKS[0]]
    engine = _Engine(tokenizer)
    monkeypatch.setattr(evaluation, "default_compute_score", _score)
    uids, infos, summary = evaluation.evaluate_benchmark(config, tokenizer, engine, SimpleNamespace, rows)
    assert len(engine.calls) == 2
    assert [len(prompts) for prompts, _ in engine.calls] == [16, 16]
    assert uids == [row.uid for row in rows for _ in range(16)]
    assert len(infos) == summary["responses"] == 32
    assert summary["response_tokens_mean"] == 2  # New tokens include EOS; scoring decode removes it.
    for prompts, params in engine.calls:
        assert all(param.n == 1 and param.ignore_eos is False and param.repetition_penalty == 1.0 for param in params)
        assert all(set(prompt) == {"prompt_token_ids"} for prompt in prompts)
    greedy = evaluation.sampling_kwargs(replace(config, do_sample=False), 64)
    assert (greedy["temperature"], greedy["top_p"], greedy["top_k"], greedy["n"]) == (0.0, 1.0, -1, 1)
    kwargs = evaluation.engine_kwargs(config)
    assert kwargs["load_format"] == "auto" and kwargs["distributed_executor_backend"] == "mp"
    assert kwargs["generation_config"] == "auto" and "eos_token_id" not in kwargs["override_generation_config"]


@pytest.mark.parametrize("failure", ["missing", "abort", "mismatched", "unfinished", "multiple"])
def test_incomplete_engine_output_is_rejected_on_cpu(inputs, monkeypatch, failure):
    config, tokenizer = inputs
    monkeypatch.setattr(evaluation, "default_compute_score", _score)
    rows = evaluation.prepare_samples(config, tokenizer)[evaluation.BENCHMARKS[0]]
    with pytest.raises(RuntimeError):
        evaluation.evaluate_benchmark(config, tokenizer, _Engine(tokenizer, failure), SimpleNamespace, rows)


def test_real_scoring_and_aggregation_match_training_on_cpu():
    cases = [
        (r"<think>\boxed{0}</think> Final answer: \boxed{4}", "4", 1.0),
        (r"<think>\boxed{4}</think> No final content", "4", 1.0),
        (r"\boxed{-1/21}", r"-\frac{1}{21}", 1.0),
        (r"\boxed{5}", "4", 0.0),
        ("No answer.", "4", 0.0),
        ("", "4", 0.0),
    ]
    benchmark = evaluation.BENCHMARKS[0]
    uids, infos = [], []
    for index, (text, ground_truth, expected_score) in enumerate(cases):

        def decode(ids, skip_special_tokens, text=text, index=index):
            assert ids == [index] and skip_special_tokens is True
            return text

        sample = EvalSample(benchmark, str(index // 2), "Question", ground_truth)
        result = evaluation.score_response(SimpleNamespace(decode=decode), [index], sample)
        reference = default_compute_score(
            benchmark, text, ground_truth, extra_info={"reward_data_source": "math_verify"}
        )
        assert result == {**reference, "reward": float(reference["score"])}
        assert result["score"] == expected_score
        uids.append(f"{benchmark}:{index // 2}")
        infos.append(result)
    trainer = object.__new__(RayPPOTrainer)
    columns = {key: [row[key] for row in infos] for key in infos[0]}
    reference_metrics = trainer._val_metrics_update(
        [benchmark] * len(infos), uids, columns, [], metric_prefix="test-legacy"
    )
    assert evaluation.aggregate_metrics(uids, benchmark, infos) == reference_metrics


def test_full_evaluation_is_stdout_only_with_no_input_mutation_on_cpu(inputs, monkeypatch, capsys, tmp_path):
    config, tokenizer = inputs
    config = replace(config, rollout_n=2)
    engine = _Engine(tokenizer)
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=lambda **kwargs: engine, SamplingParams=SimpleNamespace)
    )
    monkeypatch.setattr(evaluation, "default_compute_score", _score)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    original_open, original_io_open, original_os_open = builtins.open, io.open, os.open

    def readonly_open(file, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+"), f"Unexpected evaluation write: {file}"
        return original_open(file, mode, *args, **kwargs)

    def readonly_io_open(file, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+"), f"Unexpected evaluation write: {file}"
        return original_io_open(file, mode, *args, **kwargs)

    def readonly_os_open(file, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC), f"Unexpected evaluation write: {file}"
        return original_os_open(file, flags, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", readonly_open)
        guard.setattr(io, "open", readonly_io_open)
        guard.setattr(os, "open", readonly_os_open)
        result = evaluation.run_evaluation(config)
    output = capsys.readouterr().out
    final_lines = [line for line in output.splitlines() if line.startswith("EVAL_RESULT_JSON=")]
    assert len(final_lines) == 1 and json.loads(final_lines[0].split("=", 1)[1]) == result
    assert len(engine.calls) == 2
    assert result["config"]["bootstrap_seed"] == 42
    for benchmark in evaluation.BENCHMARKS:
        assert result["benchmarks"][benchmark]["questions"] == 2
        assert result["benchmarks"][benchmark]["responses"] == 4
        assert result["metrics"][f"test-legacy-core/{benchmark}/acc/mean@2"] == 1.0
        assert f"test-legacy-core/{benchmark}/acc/maj@2/mean" in result["metrics"]
    after = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after


def test_fsdp_evaluation_loads_and_reports_merged_path_on_cpu(inputs, monkeypatch, capsys):
    config, tokenizer, _, hf_dir = _fsdp_layout(inputs)
    engine = _Engine(tokenizer)
    stages = []

    def merge(*args, **kwargs):
        stages.append("merge")
        (hf_dir / "model.safetensors").write_bytes(b"complete mock weights")

    def create_engine(**kwargs):
        assert stages == ["merge"]
        assert kwargs["model"] == str(hf_dir) and kwargs["tokenizer"] == str(hf_dir)
        stages.append("inference")
        return engine

    monkeypatch.setattr(evaluation.subprocess, "run", merge)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=create_engine, SamplingParams=SimpleNamespace))
    monkeypatch.setattr(evaluation, "default_compute_score", _score)
    result = evaluation.run_evaluation(config)
    assert stages == ["merge", "inference"]
    assert result["config"]["model_path"] == str(hf_dir)
    assert result["config"]["force_merge"] is False
    assert len(engine.calls) == 2
    for benchmark in evaluation.BENCHMARKS:
        assert result["benchmarks"][benchmark]["responses"] == 32
        assert result["metrics"][f"test-legacy-core/{benchmark}/acc/mean@16"] == 1.0
    output = capsys.readouterr().out
    assert output.index("[checkpoint] HF conversion complete") < output.index("[eval] config=")
    final = next(line for line in output.splitlines() if line.startswith("EVAL_RESULT_JSON="))
    assert json.loads(final.split("=", 1)[1]) == result


@pytest.mark.parametrize("rank_key", ["NODE_RANK", "VC_TASK_INDEX"])
def test_python_entrypoint_skips_other_nodes_before_loading_config_on_cpu(monkeypatch, rank_key, capsys):
    monkeypatch.delenv("NODE_RANK", raising=False)
    monkeypatch.delenv("VC_TASK_INDEX", raising=False)
    monkeypatch.setenv(rank_key, "1")
    monkeypatch.setattr(evaluation.EvaluationConfig, "from_environment", lambda: pytest.fail("Unexpected config load"))
    assert evaluation.main() == 0
    assert "Skipping standalone evaluation" in capsys.readouterr().out


def test_failure_in_second_benchmark_prints_no_formal_results_on_cpu(inputs, monkeypatch, capsys):
    config, tokenizer = inputs
    engine = _Engine(tokenizer)
    generate = engine.generate

    def fail_second(*args, **kwargs):
        if engine.calls:
            raise RuntimeError("second benchmark failed")
        return generate(*args, **kwargs)

    monkeypatch.setattr(engine, "generate", fail_second)
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=lambda **kwargs: engine, SamplingParams=SimpleNamespace)
    )
    monkeypatch.setattr(evaluation, "default_compute_score", _score)
    monkeypatch.setattr(evaluation.EvaluationConfig, "from_environment", lambda: config)
    assert evaluation.main() == 1
    captured = capsys.readouterr()
    assert "second benchmark failed" in captured.err
    assert "[result]" not in captured.out and "EVAL_RESULT_JSON=" not in captured.out
