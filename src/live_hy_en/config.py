"""Command-line options, audio device listing and device selection."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pyaudiowpatch as pyaudio

DeviceInfo = dict[str, Any]

LIST_DEVICES_HINT = "Run with --list-devices to see the choices."


class DeviceError(Exception):
    """An audio device could not be found or resolved."""


@dataclass(frozen=True)
class AudioDevice:
    """A WASAPI device; `channels` counts channels in its own direction."""

    index: int
    name: str
    channels: int
    sample_rate_hz: int


@dataclass(frozen=True)
class DeviceList:
    """WASAPI devices the user can choose from."""

    microphones: list[AudioDevice]
    outputs: list[AudioDevice]


@dataclass(frozen=True)
class Settings:
    """Validated settings for one session."""

    mic: AudioDevice
    loopback: AudioDevice


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line options."""
    parser = argparse.ArgumentParser(
        prog="live_hy_en",
        description="Live Armenian transcription with English translation.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list microphones and output devices, then exit",
    )
    parser.add_argument(
        "--mic",
        help="microphone index or name (default: the Windows default microphone)",
    )
    parser.add_argument(
        "--output",
        help="output device index or name whose sound is captured "
        "(default: the Windows default output)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show each phrase's latency, broken down by pipeline step",
    )
    return parser.parse_args(argv)


def list_devices(audio: pyaudio.PyAudio) -> DeviceList:
    """Return the WASAPI microphones and output devices."""
    try:
        infos = list(
            audio.get_device_info_generator_by_host_api(host_api_type=pyaudio.paWASAPI)
        )
    except OSError as error:
        raise DeviceError(f"Windows WASAPI audio is not available: {error}") from error
    microphones = [
        _to_device(info, is_output=False)
        for info in infos
        if info["maxInputChannels"] > 0 and not info["isLoopbackDevice"]
    ]
    outputs = [
        _to_device(info, is_output=True)
        for info in infos
        if info["maxOutputChannels"] > 0
    ]
    return DeviceList(microphones=microphones, outputs=outputs)


def format_device_list(devices: DeviceList) -> str:
    """Return the device list as text for the terminal."""
    lines = ["Microphones (--mic):"]
    lines += _format_section(devices.microphones)
    lines.append("Output devices (--output):")
    lines += _format_section(devices.outputs)
    return "\n".join(lines)


def resolve_settings(args: argparse.Namespace, audio: pyaudio.PyAudio) -> Settings:
    """Resolve the chosen microphone and the loopback of the chosen output."""
    devices = list_devices(audio)
    if args.mic is None:
        mic = _default_device(audio, is_output=False)
    else:
        mic = _select_device(devices.microphones, args.mic, "microphone")
    if args.output is None:
        output = _default_device(audio, is_output=True)
    else:
        output = _select_device(devices.outputs, args.output, "output device")
    return Settings(mic=mic, loopback=_find_loopback(audio, output))


def _to_device(info: DeviceInfo, *, is_output: bool) -> AudioDevice:
    channels_key = "maxOutputChannels" if is_output else "maxInputChannels"
    return AudioDevice(
        index=info["index"],
        name=info["name"],
        channels=info[channels_key],
        sample_rate_hz=int(info["defaultSampleRate"]),
    )


def _format_section(devices: list[AudioDevice]) -> list[str]:
    if not devices:
        return ["  (none found)"]
    return [f"  [{device.index}] {device.name}" for device in devices]


def _select_device(devices: list[AudioDevice], query: str, kind: str) -> AudioDevice:
    """Pick a device by index, exact name or unique part of its name."""
    if query.isdigit():
        matches = [device for device in devices if device.index == int(query)]
    else:
        needle = query.casefold()
        # An exact name wins, so a name that is part of another stays selectable.
        matches = [
            device for device in devices if device.name.casefold() == needle
        ] or [device for device in devices if needle in device.name.casefold()]
    if not matches:
        raise DeviceError(f"No {kind} matches '{query}'. {LIST_DEVICES_HINT}")
    if len(matches) > 1:
        names = ", ".join(f"[{device.index}] {device.name}" for device in matches)
        raise DeviceError(
            f"'{query}' matches several {kind}s: {names}. Use the index instead."
        )
    return matches[0]


def _default_device(audio: pyaudio.PyAudio, *, is_output: bool) -> AudioDevice:
    kind, flag = ("output device", "--output") if is_output else ("microphone", "--mic")
    try:
        info = audio.get_default_wasapi_device(d_out=is_output, d_in=not is_output)
    except OSError as error:
        raise DeviceError(
            f"No default {kind} is set in Windows ({error}). "
            f"Choose one with {flag}. {LIST_DEVICES_HINT}"
        ) from error
    return _to_device(info, is_output=is_output)


def _find_loopback(audio: pyaudio.PyAudio, output: AudioDevice) -> AudioDevice:
    try:
        info = audio.get_wasapi_loopback_analogue_by_index(output.index)
    except (LookupError, ValueError) as error:
        raise DeviceError(
            f"Sound from '{output.name}' cannot be captured (no WASAPI loopback). "
            f"Choose another output device. {LIST_DEVICES_HINT}"
        ) from error
    # Loopback devices are inputs, so their channels are input channels.
    return _to_device(info, is_output=False)
