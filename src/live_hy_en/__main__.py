"""Entry point: `python -m live_hy_en`."""

import logging
import sys
from collections.abc import Sequence

import pyaudiowpatch as pyaudio

from live_hy_en import metrics
from live_hy_en.capture import CaptureError
from live_hy_en.config import (
    DeviceError,
    format_device_list,
    list_devices,
    parse_args,
    resolve_settings,
)
from live_hy_en.output import configure_terminal, print_error, print_message
from live_hy_en.pipeline import run
from live_hy_en.segmenter import VadModelError
from live_hy_en.transcriber import ModelLoadError

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def main(argv: Sequence[str] | None = None) -> int:
    """List the devices or run a session; return the process exit code."""
    configure_terminal()
    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
    args = parse_args(argv)
    if args.debug:
        # Only the latency view, not every library's debug output.
        logging.getLogger(metrics.__name__).setLevel(logging.DEBUG)
    audio = pyaudio.PyAudio()
    try:
        if args.list_devices:
            print_message(format_device_list(list_devices(audio)))
        else:
            run(resolve_settings(args, audio), audio)
    except (DeviceError, CaptureError, ModelLoadError, VadModelError) as error:
        print_error(str(error))
        return 1
    finally:
        audio.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
