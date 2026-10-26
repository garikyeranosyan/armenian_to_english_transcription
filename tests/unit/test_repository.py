"""Deliverable rules (PLAN.md §8, risk 14): no model weights, audio recordings or
secrets are committed, and .gitignore keeps them out.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parents[2]
WEIGHT_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".onnx", ".ckpt"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SECRET_FILES = {".env"}
# Anything bigger is probably a model or a recording, not code or results.
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024
IGNORED_PATTERNS = ["models/", "*.onnx", "*.bin", "*.wav", "*.mp3", ".env"]


@pytest.fixture(scope="module")
def tracked() -> list[Path]:
    if shutil.which("git") is None or not (REPO_DIR / ".git").exists():
        pytest.skip("not a git checkout")
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_DIR,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    return [Path(name) for name in output.split("\0") if name]


def test_no_model_weights_are_committed(tracked):
    assert [p for p in tracked if p.suffix.lower() in WEIGHT_SUFFIXES] == []


def test_no_audio_recordings_are_committed(tracked):
    assert [p for p in tracked if p.suffix.lower() in AUDIO_SUFFIXES] == []


def test_no_secrets_are_committed(tracked):
    assert [p for p in tracked if p.name in SECRET_FILES] == []


def test_no_large_files_are_committed(tracked):
    large = [
        p
        for p in tracked
        if (REPO_DIR / p).is_file()
        and (REPO_DIR / p).stat().st_size > MAX_TRACKED_FILE_BYTES
    ]
    assert large == []


def test_gitignore_keeps_weights_audio_and_secrets_out():
    patterns = (REPO_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert [p for p in IGNORED_PATTERNS if p not in patterns] == []
