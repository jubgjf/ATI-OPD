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

import nltk
import pytest

from verl.third_party import ifbench, ifevalg


@pytest.mark.parametrize("resource_module", [ifbench, ifevalg])
def test_instruction_verifiers_fail_without_resources_and_never_download_on_cpu(monkeypatch, resource_module):
    downloads = []

    def missing_resource(_resource):
        raise LookupError("not staged")

    monkeypatch.setattr(nltk.data, "find", missing_resource)
    monkeypatch.setattr(nltk, "download", lambda *args, **kwargs: downloads.append((args, kwargs)))

    with pytest.raises(RuntimeError, match="pre-provisioned"):
        resource_module.require_nltk_resources()

    assert downloads == []
