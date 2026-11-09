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

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestTokenUsage:
    """Lengths reported for a single backend request, before rollout truncation."""

    input_tokens: int
    output_tokens: int | None
    cached_input_tokens: int | None


class RequestUsage:
    """Task-local counter. Child scoring tasks share the active collector."""

    def __init__(self):
        self.metrics: Counter[str] = Counter()

    def record(self, kind: str, usage: RequestTokenUsage | None, *, failed: bool = False) -> None:
        prefix = f"opd_cost/{kind}/"
        self.metrics[prefix + "requests"] += 1
        self.metrics[prefix + "failed_requests"] += int(failed)
        self.metrics[prefix + "input_known_requests"] += int(usage is not None)
        self.metrics[prefix + "output_known_requests"] += int(usage is not None and usage.output_tokens is not None)
        self.metrics[prefix + "cache_known_requests"] += int(
            usage is not None and usage.cached_input_tokens is not None
        )
        if usage is None:
            return
        self.metrics[prefix + "input_tokens"] += usage.input_tokens
        if usage.output_tokens is not None:
            self.metrics[prefix + "output_tokens"] += usage.output_tokens
        if usage.cached_input_tokens is not None:
            self.metrics[prefix + "cached_input_tokens"] += usage.cached_input_tokens
            self.metrics[prefix + "uncached_input_tokens"] += usage.input_tokens - usage.cached_input_tokens


active_request_usage: ContextVar[RequestUsage | None] = ContextVar("active_request_usage", default=None)


@contextmanager
def collect_request_usage(enabled: bool) -> Iterator[RequestUsage | None]:
    if not enabled:
        yield None
        return
    usage = RequestUsage()
    token = active_request_usage.set(usage)
    try:
        yield usage
    finally:
        active_request_usage.reset(token)
