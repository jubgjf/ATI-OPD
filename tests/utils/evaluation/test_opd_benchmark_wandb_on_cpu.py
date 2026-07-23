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

from verl.utils.evaluation.opd_benchmark_wandb import (
    _is_not_found_error,
    artifact_fingerprint,
    artifact_name,
    resolve_run_id,
    stable_run_id,
    unit_metric_payload,
    unit_sync_key,
)


def test_wandb_identity_is_stable_and_scoped_on_cpu():
    first = stable_run_id("jubgjf", "DMPD", "experiment")

    assert first == stable_run_id("jubgjf", "DMPD", "experiment")
    assert first != stable_run_id("jubgjf", "other", "experiment")
    assert artifact_name("experiment", "math-ai__aime24--abc", 20) != artifact_name(
        "experiment", "math-ai__aime24--abc", 40
    )


def test_wandb_metric_payload_uses_global_step_axis_and_unit_fingerprint_on_cpu():
    summary = {
        "global_step": 20,
        "benchmark": {"id": "aime24--abc", "path": "math-ai/aime24"},
        "metrics": {
            "acc": 0.5,
            "question_count": 30,
            "rollout_count": 480,
            "response_tokens": {"mean": 1000, "p50": 900, "p90": 2000, "p99": 3000},
            "gap_regions": {"last_window": {"abs_mean_token_weighted": 0.3}},
        },
        "question_cluster_bootstrap_ci95": {"acc": {"ci95_low": 0.4, "ci95_high": 0.6}},
    }

    payload = unit_metric_payload(summary, "result-hash")

    assert payload["eval/global_step"] == 20
    assert payload["sync/unit_key"] == unit_sync_key("math-ai/aime24", 20)
    assert payload["sync/result_fingerprint"] == "result-hash"
    assert payload["benchmark/aime24--abc/acc"] == 0.5
    assert payload["benchmark/aime24--abc/acc_ci95_low"] == 0.4


def test_artifact_fingerprint_tracks_logs_and_rollout_policy_on_cpu(tmp_path):
    unit_dir = tmp_path / "unit"
    unit_dir.mkdir()
    log_path = unit_dir / "generate.stdout.log"
    rollout_path = unit_dir / "rollouts.jsonl.gz"
    log_path.write_text("first\n", encoding="utf-8")
    rollout_path.write_bytes(b"rollout-one")

    with_rollouts = artifact_fingerprint(unit_dir, include_rollouts=True)
    without_rollouts = artifact_fingerprint(unit_dir, include_rollouts=False)
    log_path.write_text("first\nsecond\n", encoding="utf-8")

    assert artifact_fingerprint(unit_dir, include_rollouts=True) != with_rollouts
    assert artifact_fingerprint(unit_dir, include_rollouts=False) != without_rollouts
    without_rollouts = artifact_fingerprint(unit_dir, include_rollouts=False)
    rollout_path.write_bytes(b"rollout-two")
    assert artifact_fingerprint(unit_dir, include_rollouts=False) == without_rollouts


def test_wandb_not_found_detection_does_not_swallow_other_api_failures_on_cpu():
    class Response:
        status_code = 404

    missing = RuntimeError("wrapped")
    missing.response = Response()

    assert _is_not_found_error(missing) is True
    assert _is_not_found_error(RuntimeError("connection timed out")) is False


def test_run_id_resolution_adopts_one_legacy_name_and_rejects_ambiguity_on_cpu():
    class Response:
        status_code = 404

    class FakeApi:
        def __init__(self, matches):
            self.matches = matches

        def run(self, path):
            del path
            error = RuntimeError("missing")
            error.response = Response()
            raise error

        def runs(self, **kwargs):
            assert kwargs["filters"] == {"display_name": "experiment"}
            return self.matches

    assert (
        resolve_run_id(
            FakeApi([SimpleNamespace(id="legacy-id")]),
            "jubgjf",
            "DMPD",
            "experiment",
            adopt_existing_name=True,
        )
        == "legacy-id"
    )
    with pytest.raises(RuntimeError, match="multiple W&B runs"):
        resolve_run_id(
            FakeApi([SimpleNamespace(id="first"), SimpleNamespace(id="second")]),
            "jubgjf",
            "DMPD",
            "experiment",
            adopt_existing_name=True,
        )
