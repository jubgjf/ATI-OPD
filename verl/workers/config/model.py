# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from transformers import AutoConfig

from verl.base_config import BaseConfig
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.model import get_generation_config, update_model_config

__all__ = ["HFModelConfig", "MtpConfig"]

logger = logging.getLogger(__name__)


@dataclass
class MtpConfig(BaseConfig):
    """
    Configuration for MTP model.

    enable: Enable loading and saving of MTP parameters, but do not use them

    enable_train: Whether to enable using MTP parameters during training
    enable_rollout: Whether to enable using MTP parameters during rollout

    Training parameters:
        detach_encoder: Whether to detach encoder parameters during MTP training
        mtp_loss_scaling_factor: Loss scaling factor during MTP training

    vLLM rollout parameters:
        method: "mtp"
        num-speculative-tokens: 1

    SGLang rollout parameters:
        speculative-algorithm: EAGLE
        speculative-num-steps: 3
        speculative-eagle-topk: 1
        speculative-num-draft-tokens: 4
    """

    enable: bool = False
    enable_train: bool = False
    enable_rollout: bool = False

    detach_encoder: bool = False
    mtp_loss_scaling_factor: float = 0.1

    speculative_algorithm: str = "EAGLE"
    speculative_num_steps: int = 3
    speculative_eagle_topk: int = 1
    speculative_num_draft_tokens: int = 4

    method: str = "mtp"
    num_speculative_tokens: int = 1


@dataclass
class HFModelConfig(BaseConfig):
    # note that we separate model_path, model_config_path and tokenizer_path in case they are different
    _mutable_fields = {
        "model_type",
        "hf_config_path",
        "tokenizer_path",
        "hf_config",
        "generation_config",
        "tokenizer",
        "processor",
        "local_path",
        "architectures",
        "local_hf_config_path",
        "local_tokenizer_path",
        "mtp",
    }

    path: str = MISSING
    local_path: Optional[str] = None
    hf_config_path: Optional[str] = None
    local_hf_config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None

    # model type, e.g., "language_model", "value_model"
    model_type: str = "language_model"

    # whether to load tokenizer. This is useful when we only want to load model config
    load_tokenizer: bool = True

    hf_config: Any = None
    generation_config: Any = None
    tokenizer: Any = None
    processor: Any = None

    # whether to use shared memory
    use_shm: bool = False
    trust_remote_code: bool = False

    # custom chat template for the model
    custom_chat_template: Optional[str] = None

    # copy chat template from another model or tokenizer path
    chat_template_from_model: Optional[str] = None

    external_lib: Optional[str] = None

    override_config: dict = field(default_factory=dict)

    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False

    use_remove_padding: bool = True

    # TODO: unify fsdp and megatron lora config
    # fsdp lora related. We may setup a separate config later
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: Optional[Any] = "all-linear"  # allow both "all-linear" and ["q_proj","k_proj"]
    target_parameters: Optional[list[str]] = None  # for lora adapter on nn.Parameter

    exclude_modules: Optional[str] = None

    # megatron lora config
    lora: dict[str, Any] = field(default_factory=dict)

    # path to pre-trained LoRA adapter to load for continued training
    lora_adapter_path: Optional[str] = None
    use_liger: bool = False

    use_fused_kernels: bool = False
    fused_kernel_options: dict = field(default_factory=dict)

    # TiledMLP configuration for memory-efficient MLP computation
    tiled_mlp: dict = field(default_factory=lambda: {"enabled": False, "num_shards": 4})

    architectures: Optional[list[str]] = None

    mtp: MtpConfig = field(default_factory=MtpConfig)

    def __post_init__(self):
        self._validate_target_modules()
        import_external_libs(self.external_lib)

        if self.hf_config_path is None:
            self.hf_config_path = self.path
        if self.tokenizer_path is None:
            self.tokenizer_path = self.path

        self.local_path = copy_to_local(self.path, use_shm=self.use_shm)

        # construct tokenizer
        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            self.tokenizer = hf_tokenizer(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)
            self.processor = hf_processor(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)

        # For base models (e.g. Qwen3.5-2b-Base), the processor may not have a chat_template
        # while the tokenizer does. Sync it so that processor.apply_chat_template() works.
        if (
            self.processor is not None
            and not getattr(self.processor, "chat_template", None)
            and getattr(self.tokenizer, "chat_template", None)
        ):
            self.processor.chat_template = self.tokenizer.chat_template

        if self.custom_chat_template is not None and self.chat_template_from_model is not None:
            raise ValueError("Only one of custom_chat_template and chat_template_from_model can be set.")

        if self.chat_template_from_model is not None:
            self._load_chat_template_from_model(self.chat_template_from_model)
        elif self.custom_chat_template is not None:
            self._set_chat_template(self.custom_chat_template, source="custom_chat_template")

        self.local_hf_config_path = copy_to_local(self.hf_config_path, use_shm=self.use_shm)
        self.generation_config = get_generation_config(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code
        )

        # construct hf_config
        attn_implementation = self.override_config.get("attn_implementation", "flash_attention_2")
        self.hf_config = AutoConfig.from_pretrained(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code, attn_implementation=attn_implementation
        )

        override_config_kwargs = {}

        if self.tokenizer is not None:
            override_config_kwargs.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )

        # TODO: (vermouth1992). self.config.model in megatron differs from that of fsdp in the override_config.
        override_config = (
            self.override_config["model_config"] if "model_config" in self.override_config else self.override_config
        )
        override_config_kwargs.update(override_config)
        update_model_config(self.hf_config, override_config_kwargs=override_config_kwargs)

        self.share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)

        # get model architectures
        self.architectures = getattr(self.hf_config, "architectures", None)
        assert self.architectures is not None and len(self.architectures) == 1, (
            "Expect only one architecture, got {}".format(self.architectures)
        )

        # per model patch
        if getattr(self.hf_config, "model_type", None) == "kimi_vl":
            self.hf_config.text_config.topk_method = "greedy"

        # When MTP is disabled, zero out MTP layer counts from hf_config so that
        # downstream engine/worker code does not need to handle each MTP field format
        # individually. Supports both DeepSeek-style (num_nextn_predict_layers) and
        # Qwen3.5-style (mtp_num_hidden_layers, possibly nested under text_config).
        if not self.mtp.enable:
            if hasattr(self.hf_config, "num_nextn_predict_layers"):
                self.hf_config.num_nextn_predict_layers = 0
            if hasattr(self.hf_config, "mtp_num_hidden_layers"):
                self.hf_config.mtp_num_hidden_layers = 0
            if hasattr(self.hf_config, "text_config") and hasattr(self.hf_config.text_config, "mtp_num_hidden_layers"):
                self.hf_config.text_config.mtp_num_hidden_layers = 0

    def _validate_target_modules(self) -> None:
        # Ensure target_modules is a str or list[str] (only if not None)
        if self.target_modules is not None:
            if not isinstance(self.target_modules, (str | list)):
                raise TypeError(
                    "target_modules must be a string or a list of strings, "
                    f"but got {type(self.target_modules).__name__}"
                )
            if isinstance(self.target_modules, list):
                for x in self.target_modules:
                    if not isinstance(x, str):
                        raise TypeError(
                            f"All elements in target_modules list must be strings, but found {type(x).__name__}"
                        )

    def get_processor(self):
        return self.processor if self.processor is not None else self.tokenizer

    def _set_chat_template(self, chat_template: str, *, source: str) -> None:
        if self.tokenizer is None and self.processor is None:
            raise ValueError(f"{source} requires load_tokenizer=True so a tokenizer or processor is available.")
        if self.tokenizer is not None:
            self.tokenizer.chat_template = chat_template
        if self.processor is not None:
            self.processor.chat_template = chat_template

    def _load_chat_template_from_model(self, source_model_path: str) -> None:
        local_source_model_path = copy_to_local(source_model_path, use_shm=self.use_shm)
        source_tokenizer = hf_tokenizer(local_source_model_path, trust_remote_code=self.trust_remote_code)
        chat_template = getattr(source_tokenizer, "chat_template", None)
        if not chat_template:
            raise ValueError(f"Tokenizer at {source_model_path!r} does not define chat_template.")

        self._set_chat_template(chat_template, source=f"chat_template_from_model={source_model_path!r}")
        self._log_chat_template_probe(source_model_path=source_model_path, chat_template=chat_template)

    def _log_chat_template_probe(self, *, source_model_path: str, chat_template: str) -> None:
        if self.tokenizer is None:
            logger.info(
                "Loaded chat template from %s: template_len=%d; tokenizer probe skipped because tokenizer is absent.",
                source_model_path,
                len(chat_template),
            )
            return
        try:
            probe = [{"role": "user", "content": "Solve 1+1. Put the final answer in \\boxed{}."}]
            rendered = self.tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
            think_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
            end_think_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        except Exception as exc:
            logger.warning(
                "Loaded chat template from %s, but failed to render tokenizer probe: %s",
                source_model_path,
                exc,
            )
            return

        logger.info(
            "Loaded chat template from %s: template_len=%d, contains_think=%s, contains_end_think=%s, "
            "think_ids=%s, end_think_ids=%s, rendered_prefix=%r",
            source_model_path,
            len(chat_template),
            "<think>" in rendered,
            "</think>" in rendered,
            think_ids,
            end_think_ids,
            rendered[:240],
        )
