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

import torch

from verl.experimental.teacher_loop.teacher_manager import TeacherLogprobScoringResult

# Relay-OPD public implementation eab21451f99e1a40fbb244f556de766d153c88f5.
RELAY_TRIGGER_TOPK = 5
RELAY_PARAGRAPHS = 3
RELAY_MAX_ADDITIONAL_TOKENS = 256
_REFLECTION_MARKERS = (
    "Wait, Wait,wait, wait,But, But,but, but,"
    "Hmm, Hmm, hmm,Actually, Actually,actually, actually,"
    "Hold, Hold,hold, hold,However, However,however, however,"
    "Yet, Yet,yet, yet,Oh, Oh,oh, oh,"
    "Alternatively, Alternatively,No, No,no, no,"
    "Ah, Ah,ah, ah,Oops, Oops,Well, Well"
).split(",")


def reflection_token_ids(tokenizer) -> set[int]:
    ids = set()
    for marker in _REFLECTION_MARKERS:
        encoded = tokenizer.encode(marker, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    if not ids:
        raise ValueError("Relay found no single-token reflection markers in the student tokenizer.")
    return ids


def first_reflection_trigger(
    *,
    teacher: TeacherLogprobScoringResult,
    student: TeacherLogprobScoringResult,
    prefix_len: int,
    candidate: list[int],
    reflection_ids: set[int],
) -> tuple[int, int] | None:
    """Find the first public-implementation handoff on the shared causal prefix."""
    start = prefix_len - 1
    end = start + len(candidate)
    if start < 0 or teacher.ids.shape != (end + 1, 1) or student.ids.shape != (end + 1, RELAY_TRIGGER_TOPK):
        raise ValueError("Relay expected full causal prompt scores with one final dummy row.")
    actual = torch.tensor(candidate, dtype=teacher.ids.dtype)
    valid = torch.ones_like(actual, dtype=torch.bool)
    for scoring in (teacher, student):
        if scoring.sampled_ids is None or not torch.equal(scoring.sampled_ids[start:end, 0], actual):
            raise ValueError("Relay prompt scoring does not align with the emitted candidate token IDs.")
        for index in scoring.nonfinite_sequence_positions:
            if start <= index < end:
                valid[index - start] = False
    top1 = teacher.ids[start:end, 0]
    reflect = torch.isin(top1, top1.new_tensor(sorted(reflection_ids)))
    proposed = student.ids[start:end]
    positions = torch.where(valid & reflect & (top1 != actual) & ~(proposed == top1[:, None]).any(dim=1))[0]
    if not positions.numel():
        return None
    index = int(positions[0])
    return index, int(top1[index])


def teacher_paragraph_end(tokenizer, token_ids: list[int], cache: dict[int, bool], *, starter: int) -> int:
    """Return an exclusive token boundary using the released per-token paragraph rule."""
    paragraphs = 0
    for index, token_id in enumerate([starter, *token_ids]):
        if token_id not in cache:
            cache[token_id] = "\n\n" in tokenizer.decode([token_id], skip_special_tokens=False)
        paragraphs += cache[token_id]
        if paragraphs == RELAY_PARAGRAPHS:
            return index
    return len(token_ids)
