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

from typing import Any

import pytest
import ray
from omegaconf import OmegaConf

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput


class EngineDeadError(RuntimeError):
    """Test double matching vLLM's terminal EngineCore exception name."""


class _AsyncRemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class _FakeServer:
    def __init__(self, *, output: TokenOutput | None = None, error: BaseException | None = None):
        self.output = output
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.generate = _AsyncRemoteMethod(self._generate)

    async def _generate(self, **kwargs) -> TokenOutput:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.output is not None
        return self.output


class _FakeLoadBalancer:
    def __init__(self, servers: dict[str, _FakeServer]):
        self.servers = dict(servers)
        self.removed: list[str] = []
        self.released: list[str] = []
        self.acquire_server = _AsyncRemoteMethod(self._acquire_server)
        self.release_server = _AsyncRemoteMethod(self._release_server)
        self.remove_servers = _AsyncRemoteMethod(self._remove_servers)
        self.get_all_servers = _AsyncRemoteMethod(self._get_all_servers)

    async def _acquire_server(self, request_id: str) -> tuple[str, _FakeServer]:
        del request_id
        if not self.servers:
            raise RuntimeError("No available servers in load balancer")
        server_id = next(iter(self.servers))
        return server_id, self.servers[server_id]

    def _release_server(self, server_id: str) -> None:
        self.released.append(server_id)

    async def _remove_servers(self, server_ids: list[str]) -> None:
        for server_id in server_ids:
            self.removed.append(server_id)
            self.servers.pop(server_id, None)

    async def _get_all_servers(self) -> list[str]:
        return list(self.servers)


def _wrapped_engine_dead_error() -> ray.exceptions.RayTaskError:
    cause = EngineDeadError("EngineCore encountered an issue")
    return ray.exceptions.RayTaskError(
        function_name="vLLMHttpServer.generate",
        traceback_str="Traceback (most recent call last):\nEngineDeadError: EngineCore encountered an issue",
        cause=cause,
    )


@pytest.mark.asyncio
async def test_llm_server_client_evicts_dead_engine_and_retries_on_healthy_replica_on_cpu():
    dead_server = _FakeServer(error=_wrapped_engine_dead_error())
    expected = TokenOutput(token_ids=[7], stop_reason="completed")
    healthy_server = _FakeServer(output=expected)
    load_balancer = _FakeLoadBalancer({"dead": dead_server, "healthy": healthy_server})
    client = LLMServerClient(config=OmegaConf.create({}), load_balancer_handle=load_balancer)

    output = await client.generate(
        request_id="logical-request",
        prompt_ids=[1, 2],
        sampling_params={"max_tokens": 1},
    )

    assert output == expected
    assert load_balancer.removed == ["dead"]
    assert load_balancer.released == ["dead", "healthy"]
    assert len(dead_server.calls) == 1
    assert len(healthy_server.calls) == 1


@pytest.mark.asyncio
async def test_llm_server_client_does_not_evict_or_retry_nonterminal_request_error_on_cpu():
    invalid_request_server = _FakeServer(error=ValueError("invalid sampling params"))
    untouched_server = _FakeServer(output=TokenOutput(token_ids=[9]))
    load_balancer = _FakeLoadBalancer({"invalid": invalid_request_server, "untouched": untouched_server})
    client = LLMServerClient(config=OmegaConf.create({}), load_balancer_handle=load_balancer)

    with pytest.raises(ValueError, match="invalid sampling params"):
        await client.generate(
            request_id="logical-request",
            prompt_ids=[1, 2],
            sampling_params={"max_tokens": 1},
        )

    assert load_balancer.removed == []
    assert load_balancer.released == ["invalid"]
    assert len(invalid_request_server.calls) == 1
    assert untouched_server.calls == []
