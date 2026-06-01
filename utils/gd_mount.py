from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_DRIVE_PROJECT = "/content/drive/MyDrive/VLM-Lecture2"
DEFAULT_WORK_PROJECT = "/content/VLM-Lecture2"
DEFAULT_RSYNC_EXCLUDES = (
    ".git",
    ".venv",
    "__pycache__",
    ".ipynb_checkpoints",
    "outputs",
)


def is_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def build_rsync_command(
    *,
    drive_project: str = DEFAULT_DRIVE_PROJECT,
    work_project: str = DEFAULT_WORK_PROJECT,
    excludes: Sequence[str] = DEFAULT_RSYNC_EXCLUDES,
) -> list[str]:
    command = ["rsync", "-ah", "--delete"]
    for excluded in excludes:
        command.extend(["--exclude", excluded])
    command.extend([drive_project.rstrip("/") + "/", work_project.rstrip("/") + "/"])
    return command


def setup_colab_workdir(
    *,
    drive_project: str = DEFAULT_DRIVE_PROJECT,
    work_project: str = DEFAULT_WORK_PROJECT,
    mount_drive: bool = True,
    excludes: Sequence[str] = DEFAULT_RSYNC_EXCLUDES,
) -> Path | None:
    if not is_colab():
        print("Colab 환경이 아니므로 이 셀은 건너뜁니다. 로컬 환경 설정 셀을 실행하세요.")
        return None

    if mount_drive:
        from google.colab import drive

        drive.mount("/content/drive")

    print("Copying files from Drive to /content for better I/O performance...")
    subprocess.run(
        build_rsync_command(
            drive_project=drive_project,
            work_project=work_project,
            excludes=excludes,
        ),
        check=True,
    )

    os.chdir(work_project)
    if work_project not in sys.path:
        sys.path.insert(0, work_project)

    working_root = Path.cwd()
    print(f"Colab working root: {working_root}")
    return working_root


__all__ = [
    "DEFAULT_DRIVE_PROJECT",
    "DEFAULT_RSYNC_EXCLUDES",
    "DEFAULT_WORK_PROJECT",
    "build_rsync_command",
    "is_colab",
    "setup_colab_workdir",
]
