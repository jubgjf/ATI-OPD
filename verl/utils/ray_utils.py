# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Contains commonly used utilities for ray
"""

import asyncio
import concurrent.futures
import functools
import inspect
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from multiprocessing.process import BaseProcess
from typing import Any, Optional

import ray


class RayJobResources:
    """Track and release the Ray resources created by a managed training task."""

    def __init__(self):
        self.actors: set[ray.actor.ActorHandle] = set()
        self.distributed_workers: set[ray.actor.ActorHandle] = set()
        self.vllm_servers: set[ray.actor.ActorHandle] = set()
        self.placement_groups: list = []
        self.processes: list[BaseProcess] = []

    @contextmanager
    def track(self):
        """Capture resources as soon as they are created, including on partial startup."""
        token = _job_resources.set(self)
        try:
            yield
        finally:
            _job_resources.reset(token)

    def close(self, *, interrupted: bool, timeout: int):
        """Close engines and workers before removing their placement groups."""
        log = logging.getLogger(__name__)
        errors = []

        def wait(stage, calls, *, terminating=False):
            if not calls:
                return
            ready, pending = ray.wait(list(calls), num_returns=len(calls), timeout=timeout)
            for ref in ready:
                try:
                    ray.get(ref)
                except ray.exceptions.RayActorError as exc:
                    if not terminating:
                        errors.append((stage, exc))
                except Exception as exc:
                    errors.append((stage, exc))
            if pending:
                errors.append((stage, TimeoutError(f"{len(pending)} RPCs exceeded {timeout}s")))
                if terminating:
                    for ref in pending:
                        ray.kill(calls[ref], no_restart=True)

        try:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            for process in self.processes:
                process.join(timeout=timeout)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                    errors.append(("local process shutdown", TimeoutError(f"PID {process.pid} did not exit")))

            wait(
                "vLLM request drain",
                {server.drain_for_shutdown.remote(interrupted): server for server in self.vllm_servers},
            )
            wait("vLLM engine shutdown", {server.shutdown_engine.remote(): server for server in self.vllm_servers})
            wait(
                "distributed worker shutdown",
                {
                    worker.__ray_call__.remote(_destroy_worker_process_group): worker
                    for worker in self.distributed_workers
                },
            )
            wait(
                "Ray actor exit",
                {actor.__ray_terminate__.remote(): actor for actor in self.actors},
                terminating=True,
            )
        finally:
            for group in self.placement_groups:
                ray.util.remove_placement_group(group)
            self.actors.clear()
            self.distributed_workers.clear()
            self.vllm_servers.clear()
            self.placement_groups.clear()
            self.processes.clear()

        if errors:
            for stage, error in errors:
                log.error("%s failed: %s", stage, error)
            stage, error = errors[0]
            raise RuntimeError(f"{stage} failed during training cleanup") from error


_job_resources: ContextVar[RayJobResources | None] = ContextVar("ray_job_resources", default=None)


def track_ray_actor(actor, *, distributed: bool = False, vllm: bool = False):
    """Register a newly created actor with the current managed job."""
    resources = _job_resources.get()
    if resources is not None:
        resources.actors.add(actor)
        if distributed:
            resources.distributed_workers.add(actor)
        if vllm:
            resources.vllm_servers.add(actor)
    return actor


def track_ray_placement_group(group):
    """Register a newly created placement group with the current managed job."""
    resources = _job_resources.get()
    if resources is not None:
        resources.placement_groups.append(group)
    return group


def track_ray_local_process(process: BaseProcess) -> None:
    """Register a subprocess owned by the current managed Ray task."""
    resources = _job_resources.get()
    if resources is not None:
        resources.processes.append(process)


def _destroy_worker_process_group(_worker):
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/3b9e729f6a669ffd85190f901f5e262af79771b0/python/ray/_private/accelerators/amd_gpu.py#L114-L115
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def parallel_put(data_list: list[Any], max_workers: Optional[int] = None):
    """
    Puts a list of data into the Ray object store in parallel using a thread pool.

    Args:
        data_list (List[Any]): A list of Python objects to be put into the Ray object store.
        max_workers (int, optional): The maximum number of worker threads to use.
                                     Defaults to min(len(data_list), 16).

    Returns:
        List[ray.ObjectRef]: A list of Ray object references corresponding to the input data_list,
                             maintaining the original order.
    """
    assert len(data_list) > 0, "data_list must not be empty"

    def put_data(index, data):
        return index, ray.put(data)

    if max_workers is None:
        max_workers = min(len(data_list), 16)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        data_list_f = [executor.submit(put_data, i, data) for i, data in enumerate(data_list)]
        res_lst = []
        for future in concurrent.futures.as_completed(data_list_f):
            res_lst.append(future.result())

        # reorder based on index
        output = [None for _ in range(len(data_list))]
        for res in res_lst:
            index, data_ref = res
            output[index] = data_ref

    return output


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop


def auto_await(func):
    """Auto await a coroutine function.

    Handles three cases:
    1. When the decorated function is called with await: returns the coroutine
       so the caller can await it.
    2. When called directly and there is no running event loop: runs the
       coroutine with asyncio.run() and returns the result.
    3. When called directly and the event loop is already running: runs the
       coroutine (e.g. in a thread pool to avoid deadlock) and returns the result.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        coro = func(*args, **kwargs)

        if not inspect.iscoroutine(coro):
            return coro

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        # Case 1: No running loop -> run with asyncio.run()
        if loop is None:
            return asyncio.run(coro)

        # Case 2: Running loop -> return coro if caller will await
        caller_frame = inspect.currentframe()
        if caller_frame is not None:
            caller_frame = caller_frame.f_back
        caller_is_async = caller_frame is not None and (caller_frame.f_code.co_flags & inspect.CO_COROUTINE) != 0
        if caller_is_async:
            return coro

        # Case 3: Running loop -> run coro in thread pool
        # (cannot block the loop thread without deadlock)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(copy_context().run, asyncio.run, coro)
            return future.result()

    return wrapper
