"""Tests for the entry point (PLAN.md §5.1: device list; clear errors; one command)."""

import pytest

import live_hy_en.__main__ as entry_point
from live_hy_en.config import AudioDevice, DeviceError, DeviceList, Settings
from live_hy_en.transcriber import ModelLoadError

MIC = AudioDevice(1, "Headset Microphone", 1, 48_000)
SPEAKERS = AudioDevice(5, "Speakers", 2, 48_000)
LOOPBACK = AudioDevice(7, "Speakers [Loopback]", 2, 48_000)


class FakePyAudio:
    instances: list["FakePyAudio"] = []

    def __init__(self) -> None:
        self.terminated = False
        FakePyAudio.instances.append(self)

    def terminate(self) -> None:
        self.terminated = True


@pytest.fixture
def fakes(monkeypatch):
    FakePyAudio.instances = []
    calls = {}
    monkeypatch.setattr(entry_point.pyaudio, "PyAudio", FakePyAudio)
    monkeypatch.setattr(entry_point, "configure_terminal", lambda: None)
    monkeypatch.setattr(
        entry_point, "list_devices", lambda audio: DeviceList([MIC], [SPEAKERS])
    )
    monkeypatch.setattr(
        entry_point, "resolve_settings", lambda args, audio: Settings(MIC, LOOPBACK)
    )
    monkeypatch.setattr(
        entry_point, "run", lambda settings, audio: calls.update(run=settings)
    )
    return calls


def test_list_devices_prints_microphones_and_outputs(fakes, capsys):
    assert entry_point.main(["--list-devices"]) == 0
    output = capsys.readouterr().out
    assert "[1] Headset Microphone" in output
    assert "[5] Speakers" in output
    assert "run" not in fakes


def test_without_options_a_session_runs_with_the_resolved_devices(fakes):
    assert entry_point.main([]) == 0
    assert fakes["run"] == Settings(MIC, LOOPBACK)
    assert FakePyAudio.instances[0].terminated


@pytest.mark.parametrize(
    "error",
    [
        DeviceError("No microphone matches 'usb'."),
        ModelLoadError("Speaker model not found at models/x.onnx."),
    ],
)
def test_setup_errors_are_shown_clearly_and_exit_with_1(
    fakes, monkeypatch, capsys, error
):
    def failing_run(settings, audio):
        raise error

    monkeypatch.setattr(entry_point, "run", failing_run)
    assert entry_point.main([]) == 1
    assert capsys.readouterr().err == f"Error: {error}\n"
    assert FakePyAudio.instances[0].terminated
