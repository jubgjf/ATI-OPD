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
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from verl.utils.evaluation.checkpoint_evaluation import write_evaluation_spec, write_evaluation_state, write_json_atomic
from verl.utils.evaluation.checkpoint_watcher import (
    acquire_claim,
    checkpoint_claim_path,
    discover_committed_checkpoints,
    discover_experiment_roots,
    discover_search_root_checkpoints,
    heartbeat_claim,
    inspect_claim,
    list_claims,
    load_endpoint_descriptor,
    pending_checkpoints,
    release_claim,
    release_stale_claim,
    render_queue_snapshot,
    role_claim_path,
    wandb_claim_path,
    write_endpoint_descriptor,
)
from verl.utils.evaluation.model_identity import (
    capture_stable_model_manifest,
    load_model_fingerprint_aliases,
    register_model_fingerprint_aliases,
)


def _config(*, metric_logging_mode: str = "full", experiment_name: str = "watcher-test"):
    return OmegaConf.create(
        {
            "trainer": {
                "experiment_name": experiment_name,
                "project_name": "DMPD",
                "metric_logging": {"mode": metric_logging_mode},
                "save_freq": 20,
                "max_actor_ckpt_to_keep": None,
            },
            "actor_rollout_ref": {
                "model": {"path": "/student", "tokenizer_path": None, "chat_template_from_model": "/teacher"},
                "rollout": {"test_kwargs": {"n": 16}, "max_model_len": 4096},
            },
            "data": {
                "max_prompt_length": 1024,
                "max_response_length": 3072,
                "seed": 1,
                "test_benchmarks": {"names": ["math-ai/aime24"], "reward_data_source": "math_verify"},
            },
            "reward": {"reward_manager": {"name": "naive"}},
            "distillation": {
                "enabled": True,
                "teacher_key": "data_source",
                "teacher_models": {"default": {"model_path": "/teacher"}},
                "distillation_loss": {"log_prob_min_clamp": -10.0},
                "intervention": {"enabled": True, "gap_metric": "k3"},
            },
        }
    )


def _write_fsdp_checkpoint(root, step: int, spec: dict, *, complete: bool = True, balanced: bool = False):
    checkpoint = root / f"global_step_{step}"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    write_evaluation_state(
        checkpoint,
        checkpoint_step=step,
        spec_fingerprint=spec["spec_fingerprint"],
        teacher_nll_calibration=None,
    )
    (actor / "model_world_size_2_rank_0.pt").write_bytes(b"rank-0")
    if complete:
        (actor / "model_world_size_2_rank_1.pt").write_bytes(b"rank-1")
    if balanced:
        (checkpoint / "metrics_commit.json").write_text(
            json.dumps({"checkpoint_step": step}),
            encoding="utf-8",
        )
    return checkpoint


def test_watcher_discovers_only_committed_complete_checkpoints_in_step_order(tmp_path):
    spec = write_evaluation_spec(tmp_path, _config(), training_run_id="run")
    _write_fsdp_checkpoint(tmp_path, 20, spec)
    _write_fsdp_checkpoint(tmp_path, 0, spec)
    _write_fsdp_checkpoint(tmp_path, 30, spec, complete=False)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("30", encoding="utf-8")

    candidates = discover_committed_checkpoints(tmp_path)
    assert [candidate.step for candidate in candidates] == [0, 20]
    assert all(candidate.needs_merge for candidate in candidates)
    assert [candidate.step for candidate in pending_checkpoints(tmp_path, completed_steps=[0])] == [20]


def test_watcher_does_not_synthesize_global_step_zero(tmp_path):
    spec = write_evaluation_spec(tmp_path, _config(), training_run_id="run")
    _write_fsdp_checkpoint(tmp_path, 20, spec)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")
    assert [candidate.step for candidate in discover_committed_checkpoints(tmp_path)] == [20]


def test_search_root_watcher_discovers_checkpoints_across_experiments(tmp_path):
    outputs = tmp_path / "outputs"
    run_a = outputs / "experiment-a"
    run_b = outputs / "nested-project" / "experiment-b"
    spec_a = write_evaluation_spec(run_a, _config(experiment_name="experiment-a"), training_run_id="run-a")
    spec_b = write_evaluation_spec(run_b, _config(experiment_name="experiment-b"), training_run_id="run-b")
    checkpoint_a = _write_fsdp_checkpoint(run_a, 20, spec_a)
    checkpoint_b = _write_fsdp_checkpoint(run_b, 10, spec_b)
    (run_a / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")
    (run_b / "latest_checkpointed_iteration.txt").write_text("10", encoding="utf-8")

    assert discover_experiment_roots(outputs) == [run_a.resolve(), run_b.resolve()]
    candidates = discover_search_root_checkpoints(outputs)
    by_experiment = {candidate.experiment_name: candidate for candidate in candidates}
    assert set(by_experiment) == {"experiment-a", "experiment-b"}
    assert by_experiment["experiment-a"].checkpoint_root == str(run_a.resolve())
    assert by_experiment["experiment-a"].checkpoint_dir == str(checkpoint_a.resolve())
    assert by_experiment["experiment-b"].checkpoint_root == str(run_b.resolve())
    assert by_experiment["experiment-b"].checkpoint_dir == str(checkpoint_b.resolve())
    assert len(by_experiment["experiment-a"].source_checkpoint_identity) == 64
    assert (
        by_experiment["experiment-a"].source_checkpoint_identity
        != by_experiment["experiment-b"].source_checkpoint_identity
    )

    original_identity = by_experiment["experiment-a"].source_checkpoint_identity
    os.utime(checkpoint_a, None)
    rediscovered = {candidate.experiment_name: candidate for candidate in discover_search_root_checkpoints(outputs)}
    assert rediscovered["experiment-a"].source_checkpoint_identity == original_identity

    source_shard = checkpoint_a / "actor" / "model_world_size_2_rank_0.pt"
    source_mtime = source_shard.stat().st_mtime_ns
    os.utime(source_shard, ns=(source_mtime + 1_000_000_000, source_mtime + 1_000_000_000))
    changed_source = {candidate.experiment_name: candidate for candidate in discover_search_root_checkpoints(outputs)}
    assert changed_source["experiment-a"].source_checkpoint_identity != original_identity


def test_balanced_watcher_requires_checkpoint_local_metric_commit(tmp_path):
    spec = write_evaluation_spec(tmp_path, _config(metric_logging_mode="balanced"), training_run_id="run")
    checkpoint = _write_fsdp_checkpoint(tmp_path, 20, spec, balanced=False)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")
    assert discover_committed_checkpoints(tmp_path) == []
    (checkpoint / "metrics_commit.json").write_text(json.dumps({"checkpoint_step": 20}), encoding="utf-8")
    assert [candidate.step for candidate in discover_committed_checkpoints(tmp_path)] == [20]


def test_watcher_waits_for_every_indexed_huggingface_weight_shard(tmp_path):
    spec = write_evaluation_spec(tmp_path, _config(), training_run_id="run")
    checkpoint = _write_fsdp_checkpoint(tmp_path, 20, spec, complete=False)
    hf = checkpoint / "actor" / "huggingface"
    hf.mkdir()
    (hf / "config.json").write_text("{}", encoding="utf-8")
    (hf / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (hf / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first": "model-00001-of-00002.safetensors",
                    "second": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (hf / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")

    assert discover_committed_checkpoints(tmp_path) == []
    (hf / "model-00002-of-00002.safetensors").write_bytes(b"second")
    candidates = discover_committed_checkpoints(tmp_path)
    assert [candidate.step for candidate in candidates] == [20]
    assert candidates[0].needs_merge is False


def test_endpoint_descriptor_round_trip(tmp_path):
    model = tmp_path / "teacher"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    endpoint_path = tmp_path / "teacher-endpoint.json"
    written = write_endpoint_descriptor(
        endpoint_path,
        base_url="http://teacher:8001/v1/",
        served_model_name="teacher",
        model_path=model,
        request_id="request-a",
        model_load_seconds=12.5,
        manifest_cache_root=tmp_path / "manifest-cache",
    )
    assert load_endpoint_descriptor(endpoint_path) == written
    assert written["base_url"] == "http://teacher:8001/v1"
    assert written["request_id"] == "request-a"
    assert written["model_load_seconds"] == 12.5
    assert written["model_fingerprint"] == written["model_manifest"]["fingerprint"]
    assert "legacy_stat_model_fingerprint" in written

    (model / "model.safetensors").write_bytes(b"weights")
    changed = write_endpoint_descriptor(
        endpoint_path,
        base_url="http://teacher:8001/v1",
        served_model_name="teacher",
        model_path=model,
        manifest_cache_root=tmp_path / "manifest-cache",
    )
    assert changed["model_fingerprint"] != written["model_fingerprint"]

    previous_legacy = changed["legacy_stat_model_fingerprint"]
    previous_mtime = (model / "model.safetensors").stat().st_mtime_ns
    os.utime(
        model / "model.safetensors",
        ns=(previous_mtime + 1_000_000_000, previous_mtime + 1_000_000_000),
    )
    touched = write_endpoint_descriptor(
        endpoint_path,
        base_url="http://teacher:8001/v1",
        served_model_name="teacher-relocated",
        model_path=model,
        manifest_cache_root=tmp_path / "manifest-cache",
    )
    assert touched["model_fingerprint"] == changed["model_fingerprint"]
    assert touched["legacy_stat_model_fingerprint"] != previous_legacy


def test_stable_model_manifest_is_path_and_mtime_independent_but_content_addressed(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        (root / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"same-weight-content")

    first_manifest = capture_stable_model_manifest(first, cache_path=tmp_path / "first-cache.json")
    second_manifest = capture_stable_model_manifest(second, cache_path=tmp_path / "second-cache.json")
    assert first_manifest == second_manifest

    original_mtime = (first / "model.safetensors").stat().st_mtime_ns
    os.utime(
        first / "model.safetensors",
        ns=(original_mtime + 2_000_000_000, original_mtime + 2_000_000_000),
    )
    assert capture_stable_model_manifest(first, cache_path=tmp_path / "first-cache.json") == first_manifest

    (first / "model.safetensors").write_bytes(b"diff-weight-content")
    assert capture_stable_model_manifest(first, cache_path=tmp_path / "first-cache.json") != first_manifest


def test_register_model_fingerprint_alias_requires_expected_verified_content(tmp_path):
    model = tmp_path / "teacher"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    manifest = capture_stable_model_manifest(model)
    alias_file = tmp_path / "aliases.json"
    legacy = "1" * 64

    register_model_fingerprint_aliases(
        alias_file,
        legacy_fingerprints=[legacy],
        model_path=model,
        expected_stable_fingerprint=manifest["fingerprint"],
        note="verified against archived checkpoint",
    )
    aliases = load_model_fingerprint_aliases(alias_file)
    assert aliases[legacy]["stable_model_fingerprint"] == manifest["fingerprint"]
    assert aliases[legacy]["model_manifest"] == manifest

    with pytest.raises(RuntimeError, match="does not match"):
        register_model_fingerprint_aliases(
            alias_file,
            legacy_fingerprints=["2" * 64],
            model_path=model,
            expected_stable_fingerprint="3" * 64,
        )


def test_atomic_checkpoint_claim_has_one_winner_across_processes_on_cpu(tmp_path):
    runtime_root = tmp_path / "runtime"
    checkpoint_root = tmp_path / "outputs" / "experiment"
    command = [
        sys.executable,
        "-m",
        "verl.utils.evaluation.checkpoint_watcher",
        "acquire-claim",
        "--runtime-root",
        str(runtime_root),
        "--claim-type",
        "checkpoint",
        "--instance-id",
        "worker",
        "--owner-pid",
        str(os.getpid()),
        "--checkpoint-root",
        str(checkpoint_root),
        "--global-step",
        "20",
        "--experiment-name",
        "experiment",
    ]
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, text=True) for _ in range(8)]
    results = [(process.wait(), json.loads(process.stdout.read())) for process in processes]

    winners = [payload for return_code, payload in results if return_code == 0]
    assert len(winners) == 1
    assert sum(return_code == 3 for return_code, _ in results) == 7
    winner = winners[0]
    release_claim(winner["claim_path"], winner["claim_token"])


def test_different_checkpoint_claims_can_be_owned_together_on_cpu(tmp_path):
    runtime_root = tmp_path / "runtime"
    root = tmp_path / "outputs" / "experiment"
    first = acquire_claim(
        checkpoint_claim_path(runtime_root, root, 20),
        claim_type="checkpoint",
        instance_id="worker-a",
        owner_pid=1,
    )
    second = acquire_claim(
        checkpoint_claim_path(runtime_root, root, 40),
        claim_type="checkpoint",
        instance_id="worker-b",
        owner_pid=2,
    )

    assert first["acquired"] is True
    assert second["acquired"] is True
    assert first["claim_path"] != second["claim_path"]
    release_claim(first["claim_path"], first["claim_token"])
    release_claim(second["claim_path"], second["claim_token"])


def test_claim_heartbeat_token_and_manual_stale_release_on_cpu(tmp_path):
    runtime_root = tmp_path / "runtime"
    path = wandb_claim_path(runtime_root, tmp_path / "results" / "experiment")
    claim = acquire_claim(
        path,
        claim_type="wandb",
        instance_id="worker-a",
        owner_pid=1,
        phase="wandb:publish",
    )
    heartbeat = heartbeat_claim(path, claim["claim_token"], phase="wandb:artifact")
    assert heartbeat["phase"] == "wandb:artifact"
    heartbeat_claim(path, claim["claim_token"])
    assert inspect_claim(path, stale_after_seconds=300)["owner"]["phase"] == "wandb:artifact"
    with pytest.raises(RuntimeError, match="token does not match"):
        heartbeat_claim(path, "wrong-token")
    with pytest.raises(RuntimeError, match="token does not match"):
        release_claim(path, "wrong-token")

    heartbeat["updated_at_unix"] = time.time() - 600
    write_json_atomic(path / "heartbeat.json", heartbeat)
    assert inspect_claim(path, stale_after_seconds=300)["state"] == "STALE"
    with pytest.raises(RuntimeError, match="token does not match"):
        release_stale_claim(
            runtime_root,
            path,
            claim_token="wrong-token",
            stale_after_seconds=300,
            allow_ownerless=False,
        )
    released = release_stale_claim(
        runtime_root,
        path,
        claim_token=claim["claim_token"],
        stale_after_seconds=300,
        allow_ownerless=False,
    )
    assert released["state"] == "STALE"
    assert not path.exists()


def test_ownerless_claim_requires_explicit_stale_release_on_cpu(tmp_path):
    runtime_root = tmp_path / "runtime"
    path = checkpoint_claim_path(runtime_root, tmp_path / "experiment", 20)
    path.mkdir(parents=True)
    old = time.time() - 600
    os.utime(path, (old, old))

    state = inspect_claim(path, stale_after_seconds=300)
    assert state["state"] == "STALE"
    assert state["owner"] is None
    with pytest.raises(RuntimeError, match="--allow-ownerless"):
        release_stale_claim(
            runtime_root,
            path,
            claim_token=None,
            stale_after_seconds=300,
            allow_ownerless=False,
        )
    release_stale_claim(
        runtime_root,
        path,
        claim_token=None,
        stale_after_seconds=300,
        allow_ownerless=True,
    )
    assert list_claims(runtime_root, stale_after_seconds=300) == []

    outside = tmp_path / "outside.claim"
    outside.mkdir()
    os.utime(outside, (old, old))
    with pytest.raises(ValueError, match="outside managed claim root"):
        release_stale_claim(
            runtime_root,
            outside,
            claim_token=None,
            stale_after_seconds=300,
            allow_ownerless=True,
        )


def test_role_and_wandb_claim_paths_are_resource_scoped_on_cpu(tmp_path):
    runtime_root = tmp_path / "runtime"
    controller_path = role_claim_path(runtime_root, "job-a", "controller")
    teacher_path = role_claim_path(runtime_root, "job-a", "teacher")
    assert controller_path != role_claim_path(runtime_root, "job-b", "controller")
    assert controller_path != teacher_path
    first_wandb_path = wandb_claim_path(runtime_root, tmp_path / "results-a")
    second_wandb_path = wandb_claim_path(runtime_root, tmp_path / "results-b")
    assert first_wandb_path != second_wandb_path
    controller = acquire_claim(
        controller_path,
        claim_type="role",
        instance_id="job-a",
        owner_pid=1,
        metadata={"role": "controller"},
    )
    duplicate = acquire_claim(
        controller_path,
        claim_type="role",
        instance_id="job-a",
        owner_pid=2,
        metadata={"role": "controller"},
    )
    teacher = acquire_claim(
        teacher_path,
        claim_type="role",
        instance_id="job-a",
        owner_pid=3,
        metadata={"role": "teacher"},
    )
    assert controller["acquired"] is True
    assert duplicate["acquired"] is False
    assert teacher["acquired"] is True
    first_wandb = acquire_claim(
        first_wandb_path,
        claim_type="wandb",
        instance_id="job-a",
        owner_pid=1,
    )
    duplicate_wandb = acquire_claim(
        first_wandb_path,
        claim_type="wandb",
        instance_id="job-b",
        owner_pid=2,
    )
    second_wandb = acquire_claim(
        second_wandb_path,
        claim_type="wandb",
        instance_id="job-b",
        owner_pid=2,
    )
    assert first_wandb["acquired"] is True
    assert duplicate_wandb["acquired"] is False
    assert second_wandb["acquired"] is True
    release_claim(controller["claim_path"], controller["claim_token"])
    release_claim(teacher["claim_path"], teacher["claim_token"])
    release_claim(first_wandb["claim_path"], first_wandb["claim_token"])
    release_claim(second_wandb["claim_path"], second_wandb["claim_token"])


def test_queue_snapshot_reports_current_queued_claimed_and_stale_on_cpu(tmp_path):
    runtime_root = tmp_path / "runtime"
    roots = [tmp_path / f"experiment-{index}" for index in range(4)]
    candidates = [
        {"checkpoint_root": str(root), "experiment_name": root.name, "step": index * 20}
        for index, root in enumerate(roots, start=1)
    ]
    current = acquire_claim(
        checkpoint_claim_path(runtime_root, roots[0], 20),
        claim_type="checkpoint",
        instance_id="worker-a",
        owner_pid=1,
        phase="raw:aime24",
    )
    claimed = acquire_claim(
        checkpoint_claim_path(runtime_root, roots[2], 60),
        claim_type="checkpoint",
        instance_id="worker-b",
        owner_pid=2,
        phase="load_student",
    )
    stale = acquire_claim(
        checkpoint_claim_path(runtime_root, roots[3], 80),
        claim_type="checkpoint",
        instance_id="worker-c",
        owner_pid=3,
        phase="repair:amc23",
    )
    stale_heartbeat = json.loads((Path(stale["claim_path"]) / "heartbeat.json").read_text(encoding="utf-8"))
    stale_heartbeat["updated_at_unix"] = time.time() - 600
    write_json_atomic(Path(stale["claim_path"]) / "heartbeat.json", stale_heartbeat)

    table, signature = render_queue_snapshot(
        candidates,
        runtime_root=runtime_root,
        instance_id="worker-a",
        stale_after_seconds=300,
        current_claim_path=current["claim_path"],
    )

    assert "current=experiment-0/global_step_20" in table
    assert "CURRENT" in table
    assert "QUEUED" in table
    assert "CLAIMED" in table
    assert "STALE" in table
    assert "worker-b" in table
    assert len(signature) == 64
    limited_table, repeated_signature = render_queue_snapshot(
        candidates,
        runtime_root=runtime_root,
        instance_id="worker-a",
        stale_after_seconds=300,
        current_claim_path=current["claim_path"],
        max_rows=2,
    )
    assert "2 additional row(s) hidden" in limited_table
    assert repeated_signature == signature
    release_claim(current["claim_path"], current["claim_token"])
    release_claim(claimed["claim_path"], claimed["claim_token"])
    release_claim(stale["claim_path"], stale["claim_token"])


def test_search_root_rejects_duplicate_result_identity_on_cpu(tmp_path):
    outputs = tmp_path / "outputs"
    first = outputs / "first-root"
    second = outputs / "second-root"
    write_evaluation_spec(first, _config(experiment_name="duplicate"), training_run_id="first")
    write_evaluation_spec(second, _config(experiment_name="duplicate"), training_run_id="second")

    with pytest.raises(RuntimeError, match="same standalone result identity"):
        discover_search_root_checkpoints(outputs)
