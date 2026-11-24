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

"""Ray inference workers for the standalone mathematics evaluator."""

import logging
import math
import os
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.utils.evaluation.standalone_math import DrawRequest, EvaluationConfig, engine_kwargs, generate_draws
from verl.utils.ray_utils import RayJobResources, track_ray_actor

logger = logging.getLogger(__name__)
_SHUTDOWN_TIMEOUT = 30


@dataclass(frozen=True)
class RayEvaluationTopology:
    """Resources expected to be available after the training job exits."""

    address: str
    nodes: int
    npus_per_node: int
    startup_timeout: int

    def __post_init__(self):
        if not self.nodes or not self.npus_per_node or min(self.nodes, self.npus_per_node, self.startup_timeout) <= 0:
            raise ValueError("Ray evaluation requires positive --nodes, --npus-per-node and --startup-timeout.")


class _MathReplica:
    def __init__(self, engine_options):
        context = ray.get_runtime_context()
        self.node_id = context.get_node_id()
        self.devices = tuple(map(str, context.get_accelerator_ids().get("NPU", [])))
        if len(set(self.devices)) != engine_options["tensor_parallel_size"]:
            raise RuntimeError(f"vLLM replica received unexpected NPU IDs: {self.devices}.")
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(self.devices)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM, SamplingParams

        self.engine = LLM(**engine_options)
        self.sampling_params = SamplingParams

    def ready(self):
        return self.node_id, self.devices

    def generate(self, requests):
        return generate_draws(self.engine, self.sampling_params, requests)

    def close(self):
        if self.engine is not None:
            engine, self.engine = self.engine, None
            if callable(shutdown := getattr(engine, "shutdown", None)):
                shutdown()
            else:
                engine.llm_engine.engine_core.shutdown()


def _available_nodes(topology: RayEvaluationTopology, deadline: float):
    npu_count = topology.nodes * topology.npus_per_node
    while True:
        live = {node["NodeID"]: node["Resources"].get("NPU", 0) for node in ray.nodes() if node["Alive"]}
        if (
            len(live) == topology.nodes
            and all(count == topology.npus_per_node for count in live.values())
            and ray.available_resources().get("NPU", 0) >= npu_count
        ):
            return sorted(live)
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Expected {npu_count} idle NPUs on {topology.nodes} Ray nodes, found {live}.")
        time.sleep(1)


def _check_replica_devices(placements, nodes, npus_per_node: int, tp: int):
    devices: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for node, ids in placements:
        counts[node] += 1
        assignments = {(node, device_id) for device_id in ids}
        if len(assignments) != tp or devices.intersection(assignments):
            raise RuntimeError(f"Invalid or overlapping NPU assignment on Ray node {node}: {ids}.")
        devices.update(assignments)
        print(f"[eval] Replica ready on Ray node {node}, NPUs {','.join(ids)}.", flush=True)
    if counts != {node: npus_per_node // tp for node in nodes}:
        raise RuntimeError(f"Unexpected vLLM replica distribution: {dict(counts)}.")


def _dispatch_draws(replicas, requests, batch_size):
    batches = iter(requests[start : start + batch_size] for start in range(0, len(requests), batch_size))
    pending = {}

    def submit_next(replica):
        batch = next(batches, None)
        if batch is not None:
            pending[replica.generate.remote(batch)] = (replica, batch)

    for replica in replicas:
        submit_next(replica)
    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1)
        ref = ready[0]
        replica, batch = pending.pop(ref)
        completions = ray.get(ref)
        submit_next(replica)
        yield batch, completions


@contextmanager
def ray_draw_batches(config: EvaluationConfig, topology: RayEvaluationTopology, requests: list[DrawRequest]):
    """Reserve one node-local NPU group per engine and refill replicas as batches finish."""
    tp = config.tensor_parallel_size
    if topology.npus_per_node % tp != 0:
        raise ValueError(f"STUDENT_TP={tp} must divide the {topology.npus_per_node} NPUs on each evaluation node.")
    ray.init(address=topology.address)
    resources = RayJobResources()
    replicas = []
    interrupted = True
    try:
        deadline = time.monotonic() + topology.startup_timeout
        nodes = _available_nodes(topology, deadline)
        options = engine_kwargs(config)
        replica_cls = ray.remote(num_cpus=1, resources={"NPU": tp})(_MathReplica)
        runtime_env = {
            "env_vars": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "VLLM_NO_USAGE_STATS": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "0",
            }
        }
        with resources.track():
            for node in nodes:
                for _ in range(topology.npus_per_node // tp):
                    replica = replica_cls.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False),
                        runtime_env=runtime_env,
                    ).remote(options)
                    replicas.append(track_ray_actor(replica))
        placements = ray.get(
            [replica.ready.remote() for replica in replicas], timeout=max(0, deadline - time.monotonic())
        )
        _check_replica_devices(placements, nodes, topology.npus_per_node, tp)
        batch_size = min(
            config.max_num_seqs,
            config.batch_size * config.rollout_n,
            max(1, math.ceil(len(requests) / (4 * len(replicas)))),
        )
        yield _dispatch_draws(replicas, requests, batch_size)
        interrupted = False
    finally:
        try:
            try:
                ray.get([replica.close.remote() for replica in replicas], timeout=_SHUTDOWN_TIMEOUT)
            finally:
                resources.close(interrupted=interrupted, timeout=_SHUTDOWN_TIMEOUT)
        except Exception:
            if not interrupted:
                raise
            logger.exception("Additional failure releasing distributed evaluation replicas")
        finally:
            ray.shutdown()
