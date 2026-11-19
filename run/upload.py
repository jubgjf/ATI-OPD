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

"""Sync completed OPD sessions. SwanLab receives scalars and HF stores the full log archive."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
import tempfile
import time
from importlib import import_module
from inspect import signature
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

if __package__:
    from . import migrate_experiment_logs as migration
else:
    import migrate_experiment_logs as migration

DEFAULT_HF_REPO = "jubgjf/ATI-OPD-logs"
DEFAULT_SWANLAB_PROJECT = "jubgjf/ATI-OPD"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def _run_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", value):
        raise ValueError(f"Invalid experiment run ID {value!r}.")
    return value


def _is_log_file(path: PurePosixPath) -> bool:
    if not path.parts or any(part.startswith(".") or part.endswith(".tmp") for part in path.parts):
        return False
    if path.as_posix() in {"diagnostics/manifest.json", "latest_checkpointed_iteration.txt"}:
        return True
    if len(path.parts) >= 3 and path.parts[0] == "diagnostics":
        return path.parts[1] in {"shards", "sessions", "attachments"}
    if path.parts[0] == "evaluations":
        return len(path.parts) > 1 and path.suffix == ".json"
    if path.parts[0] == "swanlog":
        return len(path.parts) >= 3
    return len(path.parts) == 2 and bool(re.fullmatch(r"global_step_\d+/metrics_commit\.json", path.as_posix()))


def _local_files(run_dir: Path) -> dict[str, Path]:
    paths = []
    for directory in ("diagnostics", "evaluations", "swanlog"):
        root = run_dir / directory
        if not root.is_symlink():
            paths.extend(root.rglob("*"))
    paths.extend(run_dir.glob("global_step_*/metrics_commit.json"))
    paths.append(run_dir / "latest_checkpointed_iteration.txt")
    files = {}
    for path in sorted(paths):
        relative = PurePosixPath(path.relative_to(run_dir).as_posix())
        if _is_log_file(relative) and path.is_file() and not path.is_symlink():
            if not path.resolve().is_relative_to(run_dir):
                raise ValueError(f"Archive file is outside the experiment directory at {path}.")
            files[relative.as_posix()] = path
    return files


def _remote_files(api: HfApi, repo: str, revision: str, prefix: str) -> dict[str, Any]:
    try:
        return {
            file.path: file
            for file in api.list_repo_tree(repo, prefix, repo_type="dataset", revision=revision, recursive=True)
            if hasattr(file, "blob_id")
        }
    except EntryNotFoundError:
        return {}


def _matches_remote(path: Path, remote: Any) -> bool:
    if not path.is_file() or path.stat().st_size != remote.size:
        return False
    if remote.lfs is None:
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {remote.size}\0".encode())
        expected = remote.blob_id
    else:
        digest = hashlib.sha256()
        expected = remote.lfs.sha256
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def upload_archive(api: HfApi, repo: str, run_dir: Path, run_id: str, *, dry_run: bool) -> int:
    """Upload only new or modified files against a single Hub commit."""
    info = api.repo_info(repo, repo_type="dataset")
    if info.private is not True:
        raise ValueError(f"HF experiment archive must be a private dataset, got {repo!r}.")
    revision = info.sha
    if not revision:
        raise RuntimeError(f"Cannot resolve the current revision for {repo!r}.")
    prefix = f"runs/{_run_id(run_id)}"
    remote = _remote_files(api, repo, revision, prefix)
    changes = []
    for name, source in _local_files(run_dir).items():
        destination = f"{prefix}/{name}"
        if destination not in remote or not _matches_remote(source, remote[destination]):
            changes.append(CommitOperationAdd(path_in_repo=destination, path_or_fileobj=source))
    if changes and not dry_run:
        api.create_commit(
            repo,
            changes,
            repo_type="dataset",
            parent_commit=revision,
            commit_message=f"Sync experiment logs for {run_id}",
        )
    return len(changes)


def download_archives(api: HfApi, repo: str, destination: Path, run_ids: list[str], *, dry_run: bool) -> int:
    """Download changed files from an immutable revision and publish the manifest last."""
    revision = api.repo_info(repo, repo_type="dataset").sha
    if not revision:
        raise RuntimeError(f"Cannot resolve the current revision for {repo!r}.")
    prefixes = [f"runs/{_run_id(run_id)}" for run_id in dict.fromkeys(run_ids)] or ["runs"]
    changes = []
    destination = destination.expanduser().resolve()
    for prefix in prefixes:
        files = _remote_files(api, repo, revision, prefix)
        if run_ids and not files:
            raise ValueError(f"No archived run at {prefix!r} in {repo!r}.")
        for name, remote in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("runs",):
                raise ValueError(f"Unsafe experiment archive path {name!r}.")
            if len(relative.parts) < 3:
                continue
            run_file = PurePosixPath(*relative.parts[2:])
            if run_file.parts[0] != "wandb_mirror" and not _is_log_file(run_file):
                continue
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Archive destination is outside the download directory at {target}.")
            if not _matches_remote(target, remote):
                changes.append((name, target, remote))
    if not changes or dry_run:
        return len(changes)

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".experiment-log-download-", dir=destination) as staging:
        fetched = []
        for name, target, remote in changes:
            source = Path(
                hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=staging, token=api.token)
            )
            if not _matches_remote(source, remote):
                raise RuntimeError(f"Downloaded content does not match the Hub revision for {name!r}.")
            fetched.append((source, target))
        for source, target in sorted(fetched, key=lambda item: item[1].name == "manifest.json"):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
    return len(changes)


def _swanlab_delta(history: migration.History, summary: dict[str, Any]) -> migration.History:
    delta = {}
    for step, row in sorted(history.items()):
        fresh = {}
        for key, value in row.items():
            latest = summary.get(key)
            if latest is None or step > latest["step"]:
                fresh[key] = value
            elif step == latest["step"] and not math.isclose(value, latest["value"], rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(f"SwanLab already has a different value for {key!r} at step {step}.")
        if fresh:
            delta[step] = fresh
    return delta


def _dashboard_history(run_dir: Path, manifest: dict[str, Any]) -> migration.History:
    tracker = run_dir / "latest_checkpointed_iteration.txt"
    last_checkpoint = int(tracker.read_text(encoding="utf-8")) if tracker.is_file() else None
    sessions = manifest.get("sessions", {})
    shards = []
    for shard in manifest.get("shards", []):
        if shard["state"] in {"OPEN", "ORPHANED"}:
            continue
        checkpoint = shard.get("checkpoint_step")
        closed = sessions.get(shard["session_id"], {}).get("status") == "CLOSED"
        if last_checkpoint is None and (checkpoint is not None or not closed):
            continue
        if last_checkpoint is not None and (checkpoint is None or checkpoint > last_checkpoint):
            continue
        shards.append(run_dir / "diagnostics" / "shards" / Path(shard["path"]).name)
    return migration._parquet_history(shards)


def _swanlab_summary(api: Any, run_path: str, keys: list[str]) -> tuple[bool, dict[str, Any]]:
    remote = api.run(run_path)
    if not remote.run_id:
        return False, {}
    # Bound the request size for runs with many dashboard metrics.
    summary = {}
    for offset in range(0, len(keys), 100):
        summary.update(remote.summary(keys=keys[offset : offset + 100]))
    return True, summary


def upload_swanlab(
    swanlab: Any, project: str, experiment: migration.Experiment, history: migration.History, *, dry_run: bool
) -> int:
    """Append only committed scalar points beyond each remote series' latest step."""
    if not history:
        return 0
    workspace, project_name = project.split("/")
    run_id = _run_id(experiment.run_id)
    keys = sorted({key for row in history.values() for key in row})
    api = swanlab.Api()
    run_path = f"{project}/{run_id}"
    exists, summary = _swanlab_summary(api, run_path, keys)
    delta = _swanlab_delta(history, summary)
    count = sum(map(len, delta.values()))
    if not count or dry_run:
        return count

    settings = swanlab.Settings(
        interactive=False,
        probe=swanlab.Settings.Probe(hardware=False, monitor=False, runtime=False, requirements=False, git=False),
        terminal=swanlab.Settings.Terminal(proxy_type="none"),
    )
    # SwanLab 0.10.3 already raises on online initialization failure. Newer
    # releases expose a policy to disable automatic fallback to offline mode.
    options = {"on_online_error": "raise"} if "on_online_error" in signature(swanlab.init).parameters else {}
    with tempfile.TemporaryDirectory(prefix="swanlab-log-sync-") as log_dir:
        run = swanlab.init(
            workspace=workspace,
            project=project_name,
            name=experiment.name,
            id=run_id,
            resume="must" if exists else "allow",
            config={"logical_run_id": run_id},
            mode="online",
            public=False,
            log_dir=log_dir,
            settings=settings,
            **options,
        )
        try:
            for step, row in delta.items():
                run.log(row, step=step)
        finally:
            run.finish()
    changed_keys = sorted({key for row in delta.values() for key in row})
    for delay in (0, 1, 2, 4):
        if delay:
            time.sleep(delay)
        _, confirmed = _swanlab_summary(api, run_path, changed_keys)
        if not _swanlab_delta(delta, confirmed):
            break
    else:
        raise RuntimeError(f"SwanLab did not confirm the latest values for {run_path}. Retry the sync to continue.")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="With no subcommand, upload all experiments under the workspace's outputs directory.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("upload", help="Upload local archives and append SwanLab scalars")
    download = commands.add_parser("download", help="Download the full archive and offline logs from HF")
    for command in (upload, download):
        command.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
        command.add_argument("--dry-run", action="store_true", help="Report the delta without transferring it")
    upload.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Workspace containing outputs/")
    upload.add_argument("--run", action="append", type=Path, help="Limit upload to an experiment directory, repeatable")
    upload.add_argument("--target", choices=("both", "hf", "swanlab"), default="both")
    upload.add_argument("--swanlab-project", default=DEFAULT_SWANLAB_PROJECT, help="SwanLab workspace/project")
    download.add_argument("--dest", required=True, type=Path)
    download.add_argument("--run", action="append", default=[], help="Logical run ID, repeatable")
    argv = sys.argv[1:]
    if not argv or argv[0] not in {"upload", "download", "-h", "--help"}:
        argv = ["upload", *argv]
    args = parser.parse_args(argv)

    if args.command == "download":
        count = download_archives(HfApi(), args.hf_repo, args.dest, args.run, dry_run=args.dry_run)
        print(f"HF archive files {'to download' if args.dry_run else 'downloaded'} {count}")
        return 0

    if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_.+-]+", args.swanlab_project):
        parser.error("--swanlab-project requires workspace/project.")
    outputs = args.root.expanduser() / "outputs"
    run_dirs = args.run or [path.parents[1] for path in sorted(outputs.glob("*/diagnostics/manifest.json"))]
    if not run_dirs:
        parser.error(
            f"No experiment archives found under {outputs}. Use --root or --run to select a different location."
        )
    hf_api = HfApi() if args.target in {"both", "hf"} else None
    swanlab = import_module("swanlab") if args.target in {"both", "swanlab"} else None

    failures = False
    seen = set()
    for run_dir in dict.fromkeys(path.expanduser().resolve() for path in run_dirs):
        try:
            with migration._snapshot_lock(run_dir):
                manifest = migration._read_json(run_dir / "diagnostics" / "manifest.json")
                run_id = _run_id(manifest["logical_run_id"])
                if run_id in seen:
                    raise ValueError(f"Multiple output directories have the same run ID {run_id!r}.")
                seen.add(run_id)
                experiment = migration.Experiment(run_id, manifest["experiment"], output=run_dir)
                print(f"{experiment.name} ({run_id})")
                if hf_api is not None:
                    count = upload_archive(hf_api, args.hf_repo, run_dir, run_id, dry_run=args.dry_run)
                    print(f"  HF archive files {'to upload' if args.dry_run else 'uploaded'} {count}")
                if swanlab is not None:
                    history = _dashboard_history(run_dir, manifest)
                    count = upload_swanlab(swanlab, args.swanlab_project, experiment, history, dry_run=args.dry_run)
                    print(f"  SwanLab points {'to upload' if args.dry_run else 'uploaded'} {count}")
        except migration.ArchiveBusyError:
            print(f"  Archive is being written at {run_dir}. Retry after the training session exits.")
            failures = True
        except (OSError, ValueError, RuntimeError) as error:
            print(f"  Sync failed for {run_dir}. {error}")
            failures = True
    return int(failures)


if __name__ == "__main__":
    raise SystemExit(main())
