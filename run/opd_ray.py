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

"""Supervise the Ray cluster for an OPD training job and its post-training evaluation."""

import argparse
import json
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cluster:
    """Addresses, topology and time limits shared by the managed nodes."""

    rank: int
    nnodes: int
    npus: int
    node_ip: str
    master: str
    gcs_port: int
    dashboard_port: int
    lifecycle_port: int
    connect_timeout: int
    ready_timeout: int
    stop_grace: int

    @classmethod
    def from_environment(cls):
        """Read the platform and launcher settings."""
        gcs_port = int(os.environ["MASTER_PORT"])
        cluster = cls(
            rank=int(os.environ["NODE_RANK"]),
            nnodes=int(os.environ["NNODES"]),
            npus=int(os.environ["NPUS_PER_NODE"]),
            node_ip=os.environ.get("MA_CURRENT_IP") or os.environ["CURRENT_IP"],
            master=os.environ["MASTER_ADDR"],
            gcs_port=gcs_port,
            dashboard_port=int(os.environ["DASHBOARD_PORT"]),
            lifecycle_port=int(os.environ.get("EXPERIMENT_SYNC_PORT", gcs_port + 1)),
            connect_timeout=int(os.environ.get("RAY_WORKER_CONNECT_TIMEOUT_SECONDS", "600")),
            ready_timeout=int(os.environ.get("RAY_CLUSTER_READY_TIMEOUT_SECONDS", "600")),
            stop_grace=int(os.environ.get("RAY_STOP_GRACE_SECONDS", "30")),
        )
        ports = {cluster.lifecycle_port, cluster.gcs_port, cluster.dashboard_port}
        if len(ports) != 3 or any(not 0 < port <= 65535 for port in ports):
            raise ValueError("Ray GCS, dashboard and lifecycle require three distinct valid TCP ports.")
        if min(cluster.connect_timeout, cluster.ready_timeout, cluster.stop_grace) <= 0:
            raise ValueError("Ray startup and cleanup time limits must be positive.")
        return cluster

    @property
    def identity(self) -> str:
        return f"{self.master}:{self.gcs_port}"

    @property
    def dashboard(self) -> str:
        return f"http://127.0.0.1:{self.dashboard_port}"


def _connect(host: str, port: int, timeout: int) -> socket.socket:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out connecting to {host}:{port}.")
        try:
            return socket.create_connection((host, port), timeout=min(2.0, remaining))
        except OSError:
            time.sleep(min(0.5, remaining))


def _ensure_running(process: subprocess.Popen, label: str) -> None:
    if process.poll() is not None:
        raise RuntimeError(f"{label} exited unexpectedly with status {process.returncode}.")


def _stop_local_ray(cluster: Cluster, process: subprocess.Popen) -> None:
    try:
        subprocess.run(
            ["ray", "stop", f"--grace-period={cluster.stop_grace}"],
            check=True,
            timeout=cluster.stop_grace + 15,
        )
        process.wait(timeout=10)
    except (subprocess.SubprocessError, OSError):
        try:
            subprocess.run(["ray", "stop", "--force"], check=False, timeout=10)
        finally:
            _stop_process(process)
        raise


@contextmanager
def _local_ray(cluster: Cluster):
    args = ["ray", "start", f"--node-ip-address={cluster.node_ip}", f'--resources={{"NPU": {cluster.npus}}}', "--block"]
    if cluster.rank == 0:
        args.extend(
            [
                "--head",
                f"--port={cluster.gcs_port}",
                "--dashboard-host=0.0.0.0",
                f"--dashboard-port={cluster.dashboard_port}",
            ]
        )
    else:
        args.append(f"--address={cluster.identity}")
    process = subprocess.Popen(args, start_new_session=True)
    try:
        yield process
    except BaseException:
        try:
            _stop_local_ray(cluster, process)
        except Exception as error:
            print(f"[opd] Ray rank {cluster.rank} also failed to shut down: {error}", file=sys.stderr)
        raise
    else:
        _stop_local_ray(cluster, process)


def _register_workers(cluster: Cluster, listener: socket.socket, process: subprocess.Popen, workers: dict) -> None:
    deadline = time.monotonic() + cluster.connect_timeout
    while len(workers) < cluster.nnodes - 1:
        _ensure_running(process, "Ray head")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            missing = sorted(set(range(1, cluster.nnodes)) - workers.keys())
            raise TimeoutError(f"Timed out waiting for Ray workers at ranks {missing}.")
        listener.settimeout(min(0.5, remaining))
        try:
            conn, _ = listener.accept()
        except TimeoutError:
            continue
        try:
            conn.settimeout(min(5.0, remaining))
            with conn.makefile("rb") as registration:
                identity, rank = registration.readline(1024).decode().split()
            rank = int(rank)
            if identity != cluster.identity or not 0 < rank < cluster.nnodes or rank in workers:
                raise RuntimeError(f"Invalid or duplicate Ray worker registration: {identity} rank {rank}.")
        except BaseException:
            conn.close()
            raise
        workers[rank] = conn
        print(f"[opd] Ray worker registered at rank {rank}.")


def _stop_workers(workers: dict[int, socket.socket], timeout: int) -> None:
    pending = {}
    failures = []
    for rank, conn in workers.items():
        try:
            conn.sendall(b"S")
            pending[rank] = conn
        except OSError:
            failures.append(rank)

    deadline = time.monotonic() + timeout
    for rank, conn in pending.items():
        try:
            conn.settimeout(max(0.01, deadline - time.monotonic()))
            if conn.recv(1) != b"0":
                failures.append(rank)
        except OSError:
            failures.append(rank)
    if failures:
        raise RuntimeError(f"Ray cleanup failed at worker ranks {sorted(failures)}.")


def _run_worker(cluster: Cluster) -> int:
    _connect(cluster.master, cluster.gcs_port, cluster.connect_timeout).close()
    with _connect(cluster.master, cluster.lifecycle_port, cluster.connect_timeout) as conn:
        try:
            with _local_ray(cluster) as process:
                conn.sendall(f"{cluster.identity} {cluster.rank}\n".encode())
                while not select.select([conn], [], [], 0.5)[0]:
                    _ensure_running(process, f"Ray worker rank {cluster.rank}")
                if conn.recv(1) != b"S":
                    raise RuntimeError("Ray head disconnected before requesting shutdown.")
        except BaseException:
            try:
                conn.sendall(b"1")
            except OSError:
                pass
            raise
        conn.sendall(b"0")
    return 0


def _wait_for_topology(cluster: Cluster) -> int:
    import ray

    ray.init(address=f"{cluster.node_ip}:{cluster.gcs_port}", log_to_driver=False)
    try:
        while True:
            counts = [node["Resources"].get("NPU", 0) for node in ray.nodes() if node["Alive"]]
            if counts == [cluster.npus] * cluster.nnodes:
                print(f"[opd] Ray is ready with {cluster.nnodes} nodes and {cluster.npus} NPUs per node.")
                return 0
            time.sleep(1)
    finally:
        ray.shutdown()


def _wait_for_stage(process, ray_head, workers, *, timeout=None):
    deadline = time.monotonic() + timeout if timeout is not None else float("inf")
    while process.poll() is None:
        _ensure_running(ray_head, "Ray head")
        if select.select(list(workers.values()), [], [], 0.5)[0]:
            raise RuntimeError("A Ray worker disconnected while the job was active.")
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the configured Ray node and NPU topology.")
    return process.returncode


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _submit_job(cluster, ray_head, workers, name, runtime_env, module, arguments):
    job_options = ["ray", "job"]
    submission = [f"--address={cluster.dashboard}", f"--submission-id={name}"]
    command = [
        *job_options,
        "submit",
        *submission,
        "--runtime-env-json=" + json.dumps(runtime_env),
        "--",
        sys.executable,
        "-m",
        module,
        *arguments,
    ]
    print(f"[opd] Submitting {name}.")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        status = _wait_for_stage(process, ray_head, workers)
    except BaseException:
        try:
            subprocess.run(
                [*job_options, "stop", f"--address={cluster.dashboard}", name],
                check=False,
                timeout=cluster.stop_grace,
            )
        except (subprocess.SubprocessError, OSError) as error:
            print(f"[opd] Could not stop {name}: {error}", file=sys.stderr)
        finally:
            _stop_process(process)
        raise
    print(f"[opd] {name} finished with status {status}.")
    return status


def _read_checkpoint_handoff(handoff: Path, checkpoint_root: Path) -> tuple[int, Path]:
    if not handoff.is_file():
        raise RuntimeError("Training did not commit a final checkpoint for this attempt.")
    record = json.loads(handoff.read_text())
    step = record["step"]
    if type(step) is not int or step <= 0:
        raise ValueError(f"Invalid final checkpoint step: {step!r}.")
    checkpoint = checkpoint_root / f"global_step_{step}"
    tracker = (checkpoint_root / "latest_checkpointed_iteration.txt").read_text().strip()
    if Path(record["checkpoint"]) != checkpoint or tracker != str(step) or not (checkpoint / "actor").is_dir():
        raise RuntimeError("The final checkpoint handoff disagrees with the committed checkpoint tracker or payload.")
    return step, checkpoint


def _run_jobs(cluster, args, ray_head, workers, attempt_dir):
    training_env = json.loads(args.train_runtime_env)
    training_env["env_vars"]["VERL_OPD_MANAGED_TRAINING"] = "1"
    attempt_id = attempt_dir.name.removeprefix(".post-train-")
    handoff = attempt_dir / "checkpoint.json"
    if args.eval_env:
        training_env["env_vars"]["VERL_OPD_CHECKPOINT_HANDOFF"] = str(handoff)
    status = _submit_job(
        cluster,
        ray_head,
        workers,
        f"opd-train-{attempt_id}",
        training_env,
        "verl.trainer.main_ppo",
        args.overrides,
    )
    if status != 0 or not args.eval_env:
        return status

    step, checkpoint = _read_checkpoint_handoff(handoff, args.checkpoint_root)
    records_path = args.checkpoint_root / "evaluations" / f"math-all-step_{step}.json"
    evaluation_env = {
        **dict(setting.split("=", 1) for setting in args.eval_env),
        "MODEL_PATH": str(checkpoint),
        "EVAL_SUITE": "math-all",
        "EVAL_ALL": "0",
        "EVAL_RECORDS_PATH": str(records_path),
        "TEST_BENCHMARK_REWARD_DATA_SOURCE": "math_verify",
        "ENABLE_TOOL_CALL": "False",
        "SOD_STYLE": "False",
        "NODE_RANK": "0",
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
    }
    return _submit_job(
        cluster,
        ray_head,
        workers,
        f"opd-eval-{attempt_id}",
        {**training_env, "env_vars": evaluation_env},
        "verl.utils.evaluation.standalone_math",
        [
            "--ray-address",
            f"{cluster.node_ip}:{cluster.gcs_port}",
            "--nodes",
            str(cluster.nnodes),
            "--npus-per-node",
            str(cluster.npus),
            "--startup-timeout",
            str(cluster.ready_timeout),
        ],
    )


def _run_head(cluster: Cluster, args) -> int:
    workers = {}
    status = None
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    try:
        with (
            tempfile.TemporaryDirectory(prefix=".post-train-", dir=args.checkpoint_root) as attempt_dir,
            socket.create_server((cluster.node_ip, cluster.lifecycle_port), backlog=cluster.nnodes) as listener,
            _local_ray(cluster) as ray_head,
        ):
            try:
                _register_workers(cluster, listener, ray_head, workers)
                topology = subprocess.Popen([sys.executable, __file__, "topology"], start_new_session=True)
                try:
                    if _wait_for_stage(topology, ray_head, workers, timeout=cluster.ready_timeout) != 0:
                        raise RuntimeError("Ray topology check failed.")
                finally:
                    _stop_process(topology)
                status = _run_jobs(cluster, args, ray_head, workers, Path(attempt_dir))
            finally:
                try:
                    _stop_workers(workers, cluster.stop_grace + 55)
                except RuntimeError as error:
                    print(f"[opd] {error}", file=sys.stderr)
                    if status == 0:
                        status = 1
                finally:
                    for conn in workers.values():
                        conn.close()
    except (subprocess.SubprocessError, OSError) as error:
        if not status:
            raise
        print(f"[opd] Ray head also failed to shut down: {error}", file=sys.stderr)
    assert status is not None
    return status


def _exit_on_signal(signum, _frame):
    raise SystemExit(128 + signum)


def main() -> int:
    """Dispatch the head, worker or topology check process."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="role", required=True)
    head = commands.add_parser("head")
    head.add_argument("--checkpoint-root", required=True, type=Path)
    head.add_argument("--train-runtime-env", required=True)
    head.add_argument("--eval-env", nargs="*")
    head.add_argument("overrides", nargs=argparse.REMAINDER)
    commands.add_parser("worker")
    commands.add_parser("topology")
    args = parser.parse_args()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, _exit_on_signal)
    cluster = Cluster.from_environment()
    if args.role == "head":
        if args.overrides[:1] == ["--"]:
            args.overrides.pop(0)
        return _run_head(cluster, args)
    if args.role == "worker":
        return _run_worker(cluster)
    return _wait_for_topology(cluster)


if __name__ == "__main__":
    sys.exit(main())
