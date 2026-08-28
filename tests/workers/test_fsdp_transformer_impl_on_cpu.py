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

from types import SimpleNamespace

import pytest
import torch

from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def _engine_with_vocab_size(tokenizer_vocab_size: int) -> FSDPEngineWithLMHead:
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.model_config = SimpleNamespace(tokenizer=range(tokenizer_vocab_size), processor=None)
    return engine


def test_fsdp_lm_head_masks_padded_vocabulary_logits_and_gradients_on_cpu():
    engine = _engine_with_vocab_size(5)
    raw_logits = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 100.0, 200.0]],
        requires_grad=True,
    )
    logits = raw_logits + 0.0

    engine._mask_tokenizer_invalid_logits_(logits)

    torch.testing.assert_close(logits[..., :5], raw_logits[..., :5])
    assert torch.isneginf(logits[..., 5:]).all()
    assert int(torch.argmax(logits, dim=-1).item()) == 4

    torch.logsumexp(logits.float(), dim=-1).sum().backward()
    torch.testing.assert_close(raw_logits.grad[..., 5:], torch.zeros_like(raw_logits.grad[..., 5:]))
    assert torch.isfinite(raw_logits.grad[..., :5]).all()


def test_fsdp_lm_head_uses_processor_tokenizer_when_direct_tokenizer_is_missing_on_cpu():
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.model_config = SimpleNamespace(tokenizer=None, processor=SimpleNamespace(tokenizer=range(3)))
    logits = torch.zeros(2, 5)

    engine._mask_tokenizer_invalid_logits_(logits)

    assert torch.isfinite(logits[..., :3]).all()
    assert torch.isneginf(logits[..., 3:]).all()


@pytest.mark.parametrize("tokenizer_vocab_size", [0, 6])
def test_fsdp_lm_head_rejects_invalid_tokenizer_vocab_size_on_cpu(tokenizer_vocab_size):
    engine = _engine_with_vocab_size(tokenizer_vocab_size)

    with pytest.raises(
        RuntimeError,
        match=rf"tokenizer_vocab_size={tokenizer_vocab_size}, model_vocab_size=5",
    ):
        engine._mask_tokenizer_invalid_logits_(torch.zeros(2, 5))
