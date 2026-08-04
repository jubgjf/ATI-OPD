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
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import json
import logging
import os
from contextlib import contextmanager, nullcontext
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Literal

import orjson

logger = logging.getLogger(__name__)

MLFLOW_MAX_ATTEMPTS = 3
MLFLOW_SLEEP_SECONDS = 5

_WANDB_OFFLINE_MODES = {"offline", "dryrun"}
_WANDB_NON_NETWORKED_MODES = _WANDB_OFFLINE_MODES | {"disabled"}


@dataclasses.dataclass(frozen=True)
class WandbResumeSpec:
    """Explicit W&B run identity and online resume behavior.

    W&B cannot resume an offline process. In offline/disabled mode, ``run_id`` is
    still reused to preserve the logical run identity, while the resume directive
    is suppressed. Each process therefore creates a separate local W&B segment.
    """

    run_id: str
    resume: bool | Literal["allow", "never", "must", "auto"] | None = None
    resume_from: str | None = None

    def __post_init__(self):
        if not self.run_id:
            raise ValueError("WandbResumeSpec.run_id must be non-empty.")
        if self.resume is not None and self.resume_from is not None:
            raise ValueError("W&B resume and resume_from are mutually exclusive.")
        if self.resume_from is not None:
            resume_run_id = self.resume_from.split("?", maxsplit=1)[0]
            if resume_run_id != self.run_id:
                raise ValueError(
                    "WandbResumeSpec.resume_from must reference the same run_id: "
                    f"run_id={self.run_id!r}, resume_from={self.resume_from!r}."
                )

    def init_kwargs(self, mode: str) -> dict[str, Any]:
        """Return conflict-free keyword arguments for ``wandb.init``."""
        kwargs: dict[str, Any] = {"id": self.run_id}
        if _normalize_wandb_mode(mode) in _WANDB_NON_NETWORKED_MODES:
            # An explicit value is required to override a process-level
            # WANDB_RESUME=must inherited by Ray workers.
            kwargs["resume"] = "never"
        elif self.resume_from is not None:
            kwargs["resume_from"] = self.resume_from
        elif self.resume is not None:
            kwargs["resume"] = self.resume
        return kwargs


@dataclasses.dataclass(frozen=True)
class WandbRunInfo:
    """Stable W&B session information needed by checkpoint/archive code."""

    mode: str
    run_id: str | None
    sync_dir: Path | None
    sync_file: Path | None

    @property
    def is_online(self) -> bool:
        return self.mode not in _WANDB_NON_NETWORKED_MODES

    @property
    def is_offline(self) -> bool:
        return self.mode in _WANDB_OFFLINE_MODES


@dataclasses.dataclass(frozen=True)
class WandbArtifactSubmission:
    """Result of staging an artifact in the current W&B session."""

    artifact_name: str
    artifact_type: str
    aliases: tuple[str, ...]
    run_info: WandbRunInfo
    wait_handle: Any | None = dataclasses.field(default=None, repr=False)

    @property
    def requires_wait(self) -> bool:
        """Whether the caller may wait for a remote upload to finish."""
        return self.wait_handle is not None


def _normalize_wandb_mode(mode: str | None) -> str:
    normalized = str(getattr(mode, "value", mode) or "online").lower()
    return "offline" if normalized == "dryrun" else normalized


def _path_or_none(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser().resolve()


@contextmanager
def _without_environment_variable(name: str):
    previous = os.environ.pop(name, None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ[name] = previous


def _is_wandb_media_value(value: Any) -> bool:
    value_type = type(value)
    module = getattr(value_type, "__module__", "")
    return module.startswith("wandb.") or module.startswith("wandb.sdk.")


def _filter_wandb_media_for_backend(data: Any, backend: str) -> Any:
    if backend in {"wandb", "vemlp_wandb"} or not isinstance(data, dict):
        return data
    return {key: value for key, value in data.items() if not _is_wandb_media_value(value)}


def _as_integral_step(value: Any, *, name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer step, got {value!r}.") from exc
    if normalized != value:
        raise ValueError(f"{name} must be an integer step, got {value!r}.")
    return normalized


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def _env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    value_lower = value.lower()
    if value_lower in {"1", "true", "yes", "y", "on"}:
        return True
    if value_lower in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _trainer_config_value(config: Any, key: str) -> Any:
    if not config:
        return None
    try:
        trainer_config = config["trainer"]
    except (KeyError, TypeError):
        return None
    if hasattr(trainer_config, "get"):
        return trainer_config.get(key, None)
    return getattr(trainer_config, key, None)


def _config_path_value(config: Any, path: str) -> Any:
    current = config
    for key in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        elif hasattr(current, "get"):
            current = current.get(key, None)
        else:
            current = getattr(current, key, None)
    return current


def _prepare_wandb_config(config: Any) -> Any:
    config_mode = os.environ.get("WANDB_CONFIG_MODE", "full").lower()
    if config_mode in {"full", ""}:
        return config
    if config_mode in {"none", "off", "disabled", "false", "0"}:
        return None
    if config_mode not in {"minimal", "summary"}:
        raise ValueError(f"WANDB_CONFIG_MODE must be one of full, minimal, summary, none; got {config_mode!r}")

    keys = [
        "trainer.project_name",
        "trainer.experiment_name",
        "trainer.default_local_dir",
        "trainer.total_training_steps",
        "trainer.save_freq",
        "trainer.test_freq",
        "trainer.test_benchmarks_freq",
        "trainer.test_before_train",
        "trainer.logger",
        "data.train_files",
        "data.val_files",
        "data.train_batch_size",
        "data.val_batch_size",
        "data.test_benchmarks",
        "data.max_prompt_length",
        "data.max_response_length",
        "actor_rollout_ref.model.path",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.rollout.test_kwargs",
        "actor_rollout_ref.rollout.tensor_model_parallel_size",
        "actor_rollout_ref.rollout.max_model_len",
        "distillation.enabled",
        "distillation.use_actor_loss",
        "distillation.teacher_models.teacher_model.model_path",
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size",
        "distillation.distillation_loss.loss_mode",
        "distillation.distillation_loss.topk",
        "distillation.distillation_loss.teacher_topk",
        "distillation.distillation_loss.top_k_strategy",
        "distillation.distillation_loss.reward_weight_mode",
        "distillation.intervention.enabled",
        "distillation.diagnostics.enabled",
        "distillation.diagnostics.async_metrics",
        "distillation.diagnostics.log_interval",
        "distillation.diagnostics.heatmap_interval",
    ]
    return {"_wandb_config_mode": config_mode} | {
        key: value for key in keys if (value := _config_path_value(config, key)) is not None
    }


def _build_wandb_settings(wandb: Any, config: Any) -> Any | None:
    settings_kwargs = {}

    wandb_mode = os.environ.get("WANDB_MODE")
    if wandb_mode:
        settings_kwargs["mode"] = wandb_mode

    wandb_dir = os.environ.get("WANDB_DIR")
    if wandb_dir:
        settings_kwargs["root_dir"] = wandb_dir

    wandb_start_method = os.environ.get("WANDB_START_METHOD")
    if wandb_start_method:
        settings_kwargs["start_method"] = wandb_start_method

    wandb_init_timeout = _env_float("WANDB_INIT_TIMEOUT")
    if wandb_init_timeout is not None:
        settings_kwargs["init_timeout"] = wandb_init_timeout

    wandb_service_wait = _env_float("WANDB_SERVICE_WAIT") or _env_float("WANDB__SERVICE_WAIT")
    if wandb_service_wait is not None:
        settings_kwargs["x_service_wait"] = wandb_service_wait

    wandb_proxy = _trainer_config_value(config, "wandb_proxy")
    if wandb_proxy:
        settings_kwargs["https_proxy"] = wandb_proxy

    if settings_kwargs.get("mode") in {"offline", "dryrun", "disabled"}:
        # In Ray/vLLM jobs, W&B is initialized after CUDA-heavy workers are created.
        # Keep offline initialization light so it does not stall on metadata/code scans.
        settings_kwargs.setdefault("disable_code", True)
        settings_kwargs.setdefault("disable_git", True)
        settings_kwargs.setdefault("x_disable_stats", True)
        settings_kwargs.setdefault("x_disable_machine_info", True)
        settings_kwargs.setdefault("x_save_requirements", False)

    for env_name, setting_name in {
        "WANDB_DISABLE_CODE": "disable_code",
        "WANDB_DISABLE_GIT": "disable_git",
        "WANDB_DISABLE_STATS": "x_disable_stats",
        "WANDB_DISABLE_MACHINE_INFO": "x_disable_machine_info",
        "WANDB_SAVE_REQUIREMENTS": "x_save_requirements",
    }.items():
        env_bool = _env_bool(env_name)
        if env_bool is not None:
            settings_kwargs[setting_name] = env_bool

    if not settings_kwargs:
        return None
    return wandb.Settings(**settings_kwargs)


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
        "mlflow",
        "swanlab",
        "vemlp_wandb",
        "tensorboard",
        "console",
        "clearml",
        "trackio",
        "file",
    ]

    def __init__(
        self,
        project_name,
        experiment_name,
        default_backend: str | list[str] = "console",
        config=None,
        *,
        wandb_resume: WandbResumeSpec | None = None,
    ):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}
        self._finished = False
        self._last_step: int | None = None
        self._last_steps_by_backend: dict[str, int] = {}
        self._wandb_run = None

        if "tracking" in default_backend or "wandb" in default_backend:
            import os

            import wandb

            settings = _build_wandb_settings(wandb, config)
            entity = os.environ.get("WANDB_ENTITY", None)
            init_kwargs: dict[str, Any] = {}
            if wandb_resume is not None:
                wandb_mode = _normalize_wandb_mode(os.environ.get("WANDB_MODE"))
                init_kwargs.update(wandb_resume.init_kwargs(wandb_mode))
                if wandb_mode in _WANDB_NON_NETWORKED_MODES and (
                    wandb_resume.resume is not None or wandb_resume.resume_from is not None
                ):
                    logger.warning(
                        "W&B does not resume local %s runs; reusing run_id=%s in a new local session.",
                        wandb_mode,
                        wandb_resume.run_id,
                    )
            # Make an explicit resume spec authoritative. In particular,
            # WANDB_RESUME=must must not collide with resume_from during rewind.
            resume_env_context = (
                _without_environment_variable("WANDB_RESUME") if wandb_resume is not None else nullcontext()
            )
            with resume_env_context:
                self._wandb_run = wandb.init(
                    project=project_name,
                    name=experiment_name,
                    entity=entity,
                    config=_prepare_wandb_config(config),
                    settings=settings,
                    **init_kwargs,
                )
            self.logger["wandb"] = wandb

        if "trackio" in default_backend:
            import trackio

            trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = trackio

        if "mlflow" in default_backend:
            import os
            import time

            import mlflow

            for _mlflow_attempt in range(1, MLFLOW_MAX_ATTEMPTS + 1):
                try:
                    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
                    logger.info("Using MLFlow tracking URI: %s", MLFLOW_TRACKING_URI)
                    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

                    # Some cloud providers like Azure ML or Databricks automatically set MLFLOW_RUN_ID
                    # If set, attach to the existing run instead of creating a new one
                    run_id = os.environ.get("MLFLOW_RUN_ID")
                    if run_id:
                        mlflow.start_run(run_id=run_id)
                    else:
                        # Project_name is actually experiment_name in MLFlow
                        # If experiment does not exist, will create a new experiment
                        experiment = mlflow.set_experiment(project_name)
                        mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)

                    mlflow.log_params(_compute_mlflow_params_from_objects(config))
                    self.logger["mlflow"] = _MlflowLoggingAdapter()
                    break  # Success
                except Exception as e:
                    logger.warning(
                        "MLflow initialization attempt %d/%d failed: %s", _mlflow_attempt, MLFLOW_MAX_ATTEMPTS, e
                    )
                    if _mlflow_attempt < MLFLOW_MAX_ATTEMPTS:
                        time.sleep(MLFLOW_SLEEP_SECONDS)
                    else:
                        logger.warning("All MLflow initialization attempts failed. Proceeding without MLflow tracking.")

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)

        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "file" in default_backend:
            self.logger["file"] = FileLogger(project_name, experiment_name)

    @property
    def wandb_run_info(self) -> WandbRunInfo | None:
        """Return the current W&B run identity and local sync paths."""
        wandb_module = self.logger.get("wandb")
        if wandb_module is None:
            return None
        run = getattr(self, "_wandb_run", None) or getattr(wandb_module, "run", None)
        if run is None:
            return None

        settings = getattr(run, "settings", None)
        raw_mode = getattr(settings, "mode", None) or os.environ.get("WANDB_MODE")
        mode = _normalize_wandb_mode(raw_mode)
        if getattr(run, "offline", False):
            mode = "offline"

        run_id = getattr(run, "id", None) or getattr(settings, "run_id", None)
        return WandbRunInfo(
            mode=mode,
            run_id=run_id,
            sync_dir=_path_or_none(getattr(settings, "sync_dir", None)),
            sync_file=_path_or_none(getattr(settings, "sync_file", None)),
        )

    def log_wandb_artifact(
        self,
        local_path: str | os.PathLike,
        *,
        artifact_name: str,
        artifact_type: str,
        metadata: dict[str, Any] | None = None,
        aliases: tuple[str, ...] | list[str] = (),
        artifact_file_name: str | None = None,
        additional_files: dict[str, str | os.PathLike] | None = None,
    ) -> WandbArtifactSubmission:
        """Stage one immutable local file as a W&B artifact.

        Online submissions return the SDK handle through ``wait_handle`` so the
        caller can decide when to wait. Offline submissions are recorded in the
        local W&B sync file and deliberately return no wait handle.
        """
        if self._finished:
            raise RuntimeError("Cannot log a W&B artifact after Tracking.finish().")
        if not artifact_name:
            raise ValueError("artifact_name must be non-empty.")
        if not artifact_type:
            raise ValueError("artifact_type must be non-empty.")

        source_path = Path(local_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"W&B artifact source must be an existing file: {source_path}")

        wandb_module = self.logger.get("wandb")
        run_info = self.wandb_run_info
        run = (
            getattr(self, "_wandb_run", None) or getattr(wandb_module, "run", None)
            if wandb_module is not None
            else None
        )
        if wandb_module is None or run is None or run_info is None:
            raise RuntimeError("The wandb backend must be initialized before logging an artifact.")
        if run_info.mode == "disabled":
            raise RuntimeError("Cannot stage a W&B artifact while WANDB_MODE=disabled.")

        normalized_aliases = tuple(dict.fromkeys(aliases))
        artifact = wandb_module.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=dict(metadata or {}),
        )
        artifact.add_file(str(source_path), name=artifact_file_name or source_path.name)
        for artifact_path, local_file in (additional_files or {}).items():
            additional_path = Path(local_file).expanduser().resolve()
            if not additional_path.is_file():
                raise FileNotFoundError(f"W&B artifact source must be an existing file: {additional_path}")
            artifact.add_file(str(additional_path), name=str(artifact_path))
        submitted_handle = run.log_artifact(artifact, aliases=list(normalized_aliases))

        return WandbArtifactSubmission(
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            aliases=normalized_aliases,
            run_info=run_info,
            wait_handle=submitted_handle if run_info.is_online else None,
        )

    def log(self, data, step, backend=None):
        normalized_step = _as_integral_step(step, name="Tracking.log(step)")
        if isinstance(data, dict) and (source_step := data.get("opd/diagnostics_async/source_step")) is not None:
            normalized_source_step = _as_integral_step(source_step, name="opd/diagnostics_async/source_step")
            if normalized_source_step != normalized_step:
                raise ValueError(
                    "opd/diagnostics_async/source_step="
                    f"{normalized_source_step} does not match Tracking.log(step={normalized_step})."
                )

        selected_backends = [
            (default_backend, logger_instance)
            for default_backend, logger_instance in self.logger.items()
            if backend is None or default_backend in backend
        ]
        last_steps = getattr(self, "_last_steps_by_backend", None)
        if last_steps is None:
            last_steps = {}
            self._last_steps_by_backend = last_steps
        for default_backend, _logger_instance in selected_backends:
            backend_last_step = last_steps.get(default_backend)
            if backend_last_step is not None and normalized_step < backend_last_step:
                raise ValueError(
                    "Tracking steps must be monotonically non-decreasing per backend: "
                    f"backend={default_backend!r}, last_step={backend_last_step}, "
                    f"requested_step={normalized_step}."
                )

        for default_backend, logger_instance in selected_backends:
            backend_data = _filter_wandb_media_for_backend(data, default_backend)
            if isinstance(backend_data, dict) and not backend_data:
                continue
            logger_instance.log(data=backend_data, step=step)
            last_steps[default_backend] = normalized_step
        self._last_step = normalized_step if self._last_step is None else max(self._last_step, normalized_step)

    def finish(self, exit_code=0):
        if getattr(self, "_finished", False):
            return
        self._finished = True
        for backend, logger_instance in list(getattr(self, "logger", {}).items()):
            try:
                if backend in {"wandb", "vemlp_wandb"}:
                    logger_instance.finish(exit_code=exit_code)
                elif backend != "mlflow" and hasattr(logger_instance, "finish"):
                    logger_instance.finish()
            except Exception as exc:
                logger.warning("Ignoring tracking backend shutdown error for %s: %s", backend, exc)

    def __del__(self):
        try:
            self.finish()
        except Exception:
            pass


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        self._task.close()


class FileLogger:
    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.filepath = os.getenv("VERL_FILE_LOGGER_PATH", None)
        if self.filepath is None:
            root_path = os.path.expanduser(os.getenv("VERL_FILE_LOGGER_ROOT", "."))
            directory = os.path.join(root_path, self.project_name)
            os.makedirs(directory, exist_ok=True)
            self.filepath = os.path.join(directory, f"{self.experiment_name}.jsonl")
        print(f"Creating file logger at {os.path.abspath(self.filepath)}")
        self.fp = open(self.filepath, "wb", buffering=0)

    def log(self, data, step):
        data = {"step": step, "data": data}
        self.fp.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n")

    def finish(self):
        self.fp.close()


class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def __init__(self):
        import logging
        import re

        self.logger = logging.getLogger(__name__)
        # Suppress noisy "Found credentials from IAM Role" on every MLflow request
        logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
        # MLflow metric key validation logic:
        # https://github.com/mlflow/mlflow/blob/master/mlflow/utils/validation.py#L157C12-L157C44
        # Only characters allowed: slashes, alphanumerics, underscores, periods, dashes, colons,
        # and spaces.
        self._invalid_chars_pattern = re.compile(
            r"[^/\w.\- :]"
        )  # Allowed: slashes, alphanumerics, underscores, periods, dashes, colons, and spaces.
        self._consecutive_slashes_pattern = re.compile(r"/+")
        self._sanitized_key_cache = {}

    def _sanitize_key(self, key):
        if key in self._sanitized_key_cache:
            return self._sanitized_key_cache[key] or key
        # First replace @ with _at_ for backward compatibility
        sanitized = key.replace("@", "_at_")
        # Replace consecutive slashes with a single slash (MLflow treats them as file paths)
        sanitized = self._consecutive_slashes_pattern.sub("/", sanitized)
        # Then replace any other invalid characters with _
        sanitized = self._invalid_chars_pattern.sub("_", sanitized)
        if sanitized == key:
            self._sanitized_key_cache[key] = None
        else:
            self.logger.warning("[MLflow] Metric key '%s' sanitized to '%s' due to invalid characters.", key, sanitized)
            self._sanitized_key_cache[key] = sanitized
        return sanitized

    def log(self, data, step):
        import mlflow

        results = {self._sanitize_key(k): v for k, v in data.items()}
        for _attempt in range(MLFLOW_MAX_ATTEMPTS):
            try:
                mlflow.log_metrics(metrics=results, step=step)
                return
            except Exception as error:
                # No sleep between retries — this runs per training step, so we avoid blocking.
                msg = "mlflow.log_metrics failed (attempt %d/%d): %s"
                args = (_attempt + 1, MLFLOW_MAX_ATTEMPTS, error)
                if _attempt < MLFLOW_MAX_ATTEMPTS - 1:
                    self.logger.info(msg, *args)
                else:
                    self.logger.warning(msg, *args)


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    project_name: str = None
    experiment_name: str = None
    table_name: str = "val/generations"
    cumulative_wandb_table: bool = True

    def log(self, loggers, samples, step, *, defer_wandb: bool = False) -> dict[str, Any]:
        deferred_media: dict[str, Any] = {}
        if "wandb" in loggers:
            deferred_media.update(self.log_generations_to_wandb(samples, step, defer=defer_wandb))
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(samples, step)

        return deferred_media

    def log_generations_to_vemlp_wandb(self, samples, step):
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step, *, defer: bool = False) -> dict[str, Any]:
        import wandb

        return self._log_generations_to_wandb(samples, step, wandb, defer=defer)

    def _log_generations_to_wandb(self, samples, step, wandb, *, defer: bool = False) -> dict[str, Any]:
        """Log samples to wandb as a table"""

        fields = self._sample_fields(samples)
        if self._samples_are_records(samples):
            columns = ["step", "sample_index", *fields]
            previous_data = (
                self.validation_table.data
                if self.cumulative_wandb_table and hasattr(self, "validation_table")
                else None
            )
            new_table = wandb.Table(columns=columns, data=previous_data)
            for sample_index, sample in enumerate(samples):
                new_table.add_data(step, sample_index, *self._sample_to_values(sample, fields))

            if wandb.run is not None and not defer:
                wandb.log({self.table_name: new_table}, step=step)
            if self.cumulative_wandb_table:
                self.validation_table = new_table
            return {self.table_name: new_table} if wandb.run is not None and defer else {}

        # Create column names for all samples
        columns = ["step"] + sum([[f"{field}_{i + 1}" for field in fields] for i in range(len(samples))], [])

        previous_data = (
            self.validation_table.data if self.cumulative_wandb_table and hasattr(self, "validation_table") else None
        )
        new_table = wandb.Table(columns=columns, data=previous_data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(self._sample_to_values(sample, fields))

        new_table.add_data(*row_data)

        # Update reference and log
        if wandb.run is not None and not defer:
            wandb.log({self.table_name: new_table}, step=step)
        if self.cumulative_wandb_table:
            self.validation_table = new_table
        return {self.table_name: new_table} if wandb.run is not None and defer else {}

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_table = swanlab.echarts.Table()

        # Create column names
        fields = self._sample_fields(samples)
        headers = ["step", *fields]

        swanlab_row_list = [[step, *self._sample_to_values(sample, fields)] for sample in samples]
        swanlab_table.add(headers=headers, rows=swanlab_row_list)

        # Log to swanlab
        swanlab.log({self.table_name: swanlab_table}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                artifact_stem = self.table_name.replace("/", "_")
                validation_gen_step_file = Path(tmp_dir, f"{artifact_stem}_step{step}.json")
                fields = self._sample_fields(samples)
                row_data = []
                for sample in samples:
                    data = dict(zip(fields, self._sample_to_values(sample, fields), strict=True))
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        fields = self._sample_fields(samples)
        table = [
            {"step": step, **dict(zip(fields, self._sample_to_values(sample, fields), strict=True))}
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series=self.table_name,
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            fields = self._sample_fields(samples)
            for field, value in zip(fields, self._sample_to_values(sample, fields), strict=True):
                text_content += f"**{field}:** {value}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text(self.table_name, text_content, step)
        # Flush to ensure data is written
        self.writer.flush()

    @staticmethod
    def _sample_fields(samples):
        if not samples:
            return ["input", "output", "score"]
        first_sample = samples[0]
        if isinstance(first_sample, dict):
            return list(first_sample.keys())
        if len(first_sample) == 3:
            return ["input", "output", "score"]
        return [f"field_{i + 1}" for i in range(len(first_sample))]

    @staticmethod
    def _sample_to_values(sample, fields):
        if isinstance(sample, dict):
            return [sample.get(field) for field in fields]
        return list(sample)

    @staticmethod
    def _samples_are_records(samples):
        return bool(samples) and isinstance(samples[0], dict)
