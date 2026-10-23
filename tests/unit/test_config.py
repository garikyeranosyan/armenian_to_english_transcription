"""Tests for device listing and selection (PLAN.md §5.1: user picks the devices)."""

import pytest

from live_hy_en.config import (
    DeviceError,
    format_device_list,
    list_devices,
    parse_args,
    resolve_settings,
)

MIC_INDEX = 1
HEADSET_MIC_INDEX = 2
SPEAKERS_INDEX = 3
HEADPHONES_INDEX = 4
SPEAKERS_LOOPBACK_INDEX = 5
HEADPHONES_LOOPBACK_INDEX = 6


def _info(
    index: int,
    name: str,
    *,
    inputs: int = 0,
    outputs: int = 0,
    is_loopback: bool = False,
) -> dict:
    return {
        "index": index,
        "name": name,
        "maxInputChannels": inputs,
        "maxOutputChannels": outputs,
        "defaultSampleRate": 48000.0,
        "isLoopbackDevice": is_loopback,
    }


DEVICE_INFOS = [
    _info(MIC_INDEX, "Microphone Array (Realtek)", inputs=2),
    _info(HEADSET_MIC_INDEX, "Headset Microphone (USB)", inputs=1),
    _info(SPEAKERS_INDEX, "Speakers (Realtek)", outputs=2),
    _info(HEADPHONES_INDEX, "Headphones (USB)", outputs=2),
    _info(
        SPEAKERS_LOOPBACK_INDEX,
        "Speakers (Realtek) [Loopback]",
        inputs=2,
        is_loopback=True,
    ),
    _info(
        HEADPHONES_LOOPBACK_INDEX,
        "Headphones (USB) [Loopback]",
        inputs=2,
        is_loopback=True,
    ),
]


class FakeAudio:
    """Stands in for pyaudiowpatch.PyAudio with the methods config uses."""

    def __init__(
        self,
        infos: list[dict] = DEVICE_INFOS,
        *,
        default_mic_index: int | None = MIC_INDEX,
        default_output_index: int | None = SPEAKERS_INDEX,
        has_wasapi: bool = True,
    ) -> None:
        self._infos = infos
        self._default_mic_index = default_mic_index
        self._default_output_index = default_output_index
        self._has_wasapi = has_wasapi

    def get_device_info_generator_by_host_api(self, *, host_api_type):
        if not self._has_wasapi:
            raise OSError("host API not found")
        yield from self._infos

    def get_default_wasapi_device(self, *, d_out=False, d_in=False):
        index = self._default_output_index if d_out else self._default_mic_index
        if index is None:
            raise OSError("invalid device")
        return self._get_info(index)

    def get_wasapi_loopback_analogue_by_index(self, index):
        name = self._get_info(index)["name"]
        for info in self._infos:
            if info["isLoopbackDevice"] and name in info["name"]:
                return info
        raise LookupError("no analogue")

    def _get_info(self, index):
        return next(info for info in self._infos if info["index"] == index)


def _resolve(*argv: str, audio: FakeAudio | None = None):
    return resolve_settings(parse_args(list(argv)), audio or FakeAudio())


def test_defaults_are_used_when_no_device_is_given():
    settings = _resolve()
    assert settings.mic.index == MIC_INDEX
    assert settings.loopback.index == SPEAKERS_LOOPBACK_INDEX


def test_mic_can_be_chosen_by_index():
    assert _resolve("--mic", str(HEADSET_MIC_INDEX)).mic.index == HEADSET_MIC_INDEX


def test_mic_can_be_chosen_by_part_of_its_name_ignoring_case():
    assert _resolve("--mic", "headset").mic.index == HEADSET_MIC_INDEX


def test_output_choice_resolves_to_its_loopback():
    settings = _resolve("--output", "Headphones")
    assert settings.loopback.index == HEADPHONES_LOOPBACK_INDEX
    assert settings.loopback.channels == 2


def test_exact_name_wins_over_partial_matches():
    infos = DEVICE_INFOS + [_info(7, "Microphone Array (Realtek) 2", inputs=2)]
    settings = _resolve("--mic", "Microphone Array (Realtek)", audio=FakeAudio(infos))
    assert settings.mic.index == MIC_INDEX


def test_unknown_device_gives_clear_error():
    with pytest.raises(DeviceError, match="No microphone matches 'Webcam'.*--list"):
        _resolve("--mic", "Webcam")


def test_ambiguous_name_gives_clear_error():
    with pytest.raises(DeviceError, match="matches several microphones"):
        _resolve("--mic", "microphone")


def test_output_index_cannot_select_a_microphone():
    with pytest.raises(DeviceError, match="No output device matches"):
        _resolve("--output", str(MIC_INDEX))


def test_loopback_device_is_not_offered_as_microphone():
    with pytest.raises(DeviceError, match="No microphone matches"):
        _resolve("--mic", str(SPEAKERS_LOOPBACK_INDEX))


def test_missing_default_mic_gives_clear_error():
    with pytest.raises(DeviceError, match="No default microphone.*--mic"):
        _resolve(audio=FakeAudio(default_mic_index=None))


def test_output_without_loopback_gives_clear_error():
    infos = [info for info in DEVICE_INFOS if not info["isLoopbackDevice"]]
    with pytest.raises(DeviceError, match="cannot be captured"):
        _resolve(audio=FakeAudio(infos))


def test_missing_wasapi_gives_clear_error():
    with pytest.raises(DeviceError, match="WASAPI audio is not available"):
        _resolve(audio=FakeAudio(has_wasapi=False))


def test_device_list_shows_microphones_and_outputs():
    text = format_device_list(list_devices(FakeAudio()))
    assert "Microphones (--mic):\n  [1] Microphone Array (Realtek)\n" in text
    assert "Output devices (--output):\n  [3] Speakers (Realtek)\n" in text
    assert "[Loopback]" not in text


def test_list_devices_flag_is_parsed():
    assert parse_args(["--list-devices"]).list_devices
    assert not parse_args([]).list_devices
