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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from run import opd_ray
from verl.trainer import main_ppo
from verl.utils import ray_utils


def _checkpoint(root, step):
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))
    checkpoint = root / f"global_step_{step}"
    (checkpoint / "actor").mkdir(parents=True, exist_ok=True)
    return checkpoint


@pytest.mark.parametrize(
    "case,expected_status,expected_modules",
    [
        ("success", 0, ["verl.trainer.main_ppo", "verl.utils.evaluation.standalone_math"]),
        ("eval-failed", 2, ["verl.trainer.main_ppo", "verl.utils.evaluation.standalone_math"]),
        ("train-failed", 3, ["verl.trainer.main_ppo"]),
        ("train-only", 0, ["verl.trainer.main_ppo"]),
        ("missing-handoff", None, ["verl.trainer.main_ppo"]),
        ("stale-tracker", None, ["verl.trainer.main_ppo"]),
    ],
)
def test_post_train_evaluation_requires_a_committed_checkpoint(
    tmp_path, monkeypatch, case, expected_status, expected_modules
):
    attempt = tmp_path / ".post-train-test"
    attempt.mkdir()
    checkpoint = _checkpoint(tmp_path, 20)
    args = SimpleNamespace(
        train_runtime_env='{"working_dir": "./", "env_vars": {"SWANLAB_MODE": "offline"}}',
        eval_env=None if case == "train-only" else ["ENABLE_THINKING=False", "CHAT_TEMPLATE_MODEL=/local/template"],
        overrides=["trainer.save_freq=20"],
        checkpoint_root=tmp_path,
    )
    cluster = SimpleNamespace(node_ip="127.0.0.1", gcs_port=6167, nnodes=4, npus=8, ready_timeout=600)
    jobs = []

    def submit(_cluster, _head, _workers, _name, runtime_env, module, arguments):
        jobs.append(module)
        env = runtime_env["env_vars"]
        if module == "verl.trainer.main_ppo":
            assert env["VERL_OPD_MANAGED_TRAINING"] == "1"
            assert env["SWANLAB_MODE"] == "offline"
            if case not in {"train-only", "missing-handoff"}:
                assert env["VERL_OPD_CHECKPOINT_HANDOFF"] == str(attempt / "checkpoint.json")
                step = 10 if case == "stale-tracker" else 20
                (attempt / "checkpoint.json").write_text(
                    json.dumps({"step": step, "checkpoint": str(tmp_path / f"global_step_{step}")})
                )
            return 3 if case == "train-failed" else 0
        assert env["MODEL_PATH"] == str(checkpoint)
        assert env["EVAL_SUITE"] == "math-all"
        assert env["EVAL_RECORDS_PATH"] == str(tmp_path / "evaluations" / "math-all-step_20.json")
        assert env["ENABLE_THINKING"] == "False"
        assert env["CHAT_TEMPLATE_MODEL"] == "/local/template"
        assert arguments[:6] == ["--ray-address", "127.0.0.1:6167", "--nodes", "4", "--npus-per-node", "8"]
        return 2 if case == "eval-failed" else 0

    monkeypatch.setattr(opd_ray, "_submit_job", submit)
    if expected_status is None:
        with pytest.raises(RuntimeError, match="final checkpoint"):
            opd_ray._run_jobs(cluster, args, None, {}, attempt)
    else:
        assert opd_ray._run_jobs(cluster, args, None, {}, attempt) == expected_status
    assert jobs == expected_modules
    assert (checkpoint / "actor").is_dir()


def test_worker_remains_registered_until_head_requests_shutdown(monkeypatch):
    from socket import socketpair

    head_conn, worker_conn = socketpair()
    sockets = iter([SimpleNamespace(close=lambda: None), worker_conn])
    monkeypatch.setattr(opd_ray, "_connect", lambda *_: next(sockets))
    events = []

    @contextmanager
    def local_ray(_cluster):
        events.append("started")
        try:
            yield SimpleNamespace(poll=lambda: None)
        finally:
            events.append("stopped")

    monkeypatch.setattr(opd_ray, "_local_ray", local_ray)
    cluster = SimpleNamespace(
        master="head", identity="head:6167", gcs_port=6167, lifecycle_port=6168, rank=1, connect_timeout=1
    )
    with ThreadPoolExecutor(max_workers=1) as executor, head_conn:
        worker = executor.submit(opd_ray._run_worker, cluster)
        head_conn.settimeout(1)
        with head_conn.makefile("rb") as registration:
            assert registration.readline() == b"head:6167 1\n"
        assert events == ["started"]
        opd_ray._stop_workers({1: head_conn}, 1)
        assert worker.result(timeout=1) == 0
    assert events == ["started", "stopped"]


@pytest.mark.parametrize("failed_stage", [None, "training", "cleanup"])
def test_task_runner_finishes_before_publishing_handoff(tmp_path, monkeypatch, failed_stage):
    attempt = tmp_path / "handoff.json"
    monkeypatch.setenv("VERL_OPD_MANAGED_TRAINING", "1")
    monkeypatch.setenv("VERL_OPD_CHECKPOINT_HANDOFF", str(attempt))
    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(main_ppo.ray, "get", lambda task, **_: task())
    events = []
    checkpoint = _checkpoint(tmp_path, 20)
    (checkpoint / "metrics_commit.json").write_text("{}")
    config = OmegaConf.create(
        {"trainer": {"default_local_dir": str(tmp_path), "metric_logging": {"mode": "balanced"}}, "ray_kwargs": {}}
    )

    def train(_config):
        events.append("fit")
        if failed_stage == "training":
            raise ValueError("training failed")
        return 20

    def cleanup(interrupted):
        assert not attempt.exists()
        events.append(("cleanup", interrupted))
        if failed_stage is not None:
            raise RuntimeError("cleanup failed")

    def task(func: Callable):
        return SimpleNamespace(remote=lambda *args: lambda: func(*args))

    runner = SimpleNamespace(
        remote=lambda: SimpleNamespace(
            run=task(train),
            finish_training=task(cleanup),
            __ray_terminate__=task(lambda: events.append("runner exit")),
        )
    )
    if failed_stage:
        error = ValueError if failed_stage == "training" else RuntimeError
        with pytest.raises(error, match=f"{failed_stage} failed"):
            main_ppo.run_ppo(config, task_runner_class=runner)
        assert not attempt.exists()
    else:
        main_ppo.run_ppo(config, task_runner_class=runner)
        assert json.loads(attempt.read_text()) == {"step": 20, "checkpoint": str(checkpoint)}
    assert events == ["fit", ("cleanup", failed_stage == "training")] + ([] if failed_stage else ["runner exit"])


def test_partial_startup_releases_engines_workers_and_placement_groups(monkeypatch):
    resource = ray_utils.RayJobResources()
    calls = []

    def task(name):
        return SimpleNamespace(remote=lambda *args: calls.append((name, args)) or name)

    class Actor:
        drain_for_shutdown = task("drain")
        shutdown_engine = task("shutdown")
        __ray_call__ = task("distributed worker")
        __ray_terminate__ = task("exit")

    def get(ref):
        if ref == "drain":
            raise RuntimeError("engine init failed")

    monkeypatch.setattr(ray_utils.ray, "get", get)
    monkeypatch.setattr(ray_utils.ray, "wait", lambda refs, **_: (refs, []))
    monkeypatch.setattr(ray_utils.ray.util, "remove_placement_group", lambda group: calls.append(("remove", group)))
    with resource.track(), pytest.raises(ValueError):
        ray_utils.track_ray_actor(Actor(), vllm=True, distributed=True)
        ray_utils.track_ray_placement_group("pg")
        raise ValueError("startup failed")
    with pytest.raises(RuntimeError, match="vLLM request drain"):
        resource.close(interrupted=True, timeout=1)
    assert [name for name, _ in calls] == ["drain", "shutdown", "distributed worker", "exit", "remove"]
