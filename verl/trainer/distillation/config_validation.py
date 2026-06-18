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

from typing import Any

from verl.trainer.distillation.losses import is_distillation_enabled


def _get_nested(config: Any, *keys: str, default: Any = None) -> Any:
    current = config
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def validate_distillation_intervention_training_config(config: Any) -> None:
    """Validate cross-feature combinations that cannot be represented in one dataclass."""
    distillation_config = _get_nested(config, "distillation")
    if not is_distillation_enabled(distillation_config):
        return
    intervention_enabled = bool(_get_nested(distillation_config, "intervention", "enabled", default=False))
    train_on_teacher_segments = bool(
        _get_nested(distillation_config, "intervention", "train_on_teacher_segments", default=False)
    )
    bypass_mode = bool(_get_nested(config, "algorithm", "rollout_correction", "bypass_mode", default=False))
    if intervention_enabled and train_on_teacher_segments and bypass_mode:
        raise ValueError(
            "distillation.intervention.train_on_teacher_segments=true is incompatible with "
            "algorithm.rollout_correction.bypass_mode=true. Teacher intervention drops rollout_log_probs so the "
            "actor can recompute logprobs on repaired teacher tokens, while bypass mode requires rollout_log_probs."
        )
