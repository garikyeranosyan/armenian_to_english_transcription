# DECISIONS — Important project decisions

This file records important decisions made during the project. Entries are added only when the user asks, newest last. Only important decisions are recorded (library choices, strategies, models and similar), not implementation details.

Each entry uses this format:

```markdown
## <number>. <Short title> (<YYYY-MM-DD>)

- **Decision:** what was decided.
- **Reason:** why it was decided.
- **Alternatives considered:** what else was considered and why it was not chosen.
```

---

## capture.py

### 1. Shared data types live in the module that produces them (2026-10-23)

- **Decision:** `AudioFrame` and `Source` are defined in `capture.py`; later types follow the same rule (`SpeechSegment` in `segmenter.py`, `TranscriptEntry` in `output.py`).
- **Reason:** keeps exactly the 9 modules from `PLAN.md` and avoids circular imports.
- **Alternatives considered:** a shared `types.py`, which is clearer to find but adds a 10th module.

### 2. Streaming resampling with soxr (2026-10-23)

- **Decision:** convert to 16 kHz with `soxr.ResampleStream`.
- **Reason:** it keeps filter state between callbacks, so chunk edges don't click; fast C code; small dependency with Windows wheels.
- **Alternatives considered:** `scipy.signal.resample_poly` (large dependency, per-chunk edge artefacts); numpy linear interpolation (aliasing that may hurt ASR accuracy).

### 3. Wall-clock timestamps on one shared session clock (2026-10-23)

- **Decision:** both sources read one `SessionClock` (`time.monotonic()`), sampled in each callback.
- **Reason:** WASAPI loopback sends no data while nothing plays, so counting samples would drift after every silent gap.
- **Alternatives considered:** a sample counter per source (jitter-free, but wrong after silence unless gaps are padded).

### 4. Unbounded queue from capture to the segmenter (2026-10-23)

- **Decision:** each source puts frames on its own `queue.SimpleQueue`, with no size limit.
- **Reason:** capture never waits and no audio is dropped; the segmenter is cheap and expected to keep up. The bounded queue required by `PLAN.md` risk 8 goes later, between the segmenter and the models.
- **Alternatives considered:** a bounded queue dropping the oldest or newest frames (caps memory, but loses audio).

### 5. Common format is 16 kHz mono float32 (2026-10-23)

- **Decision:** every frame leaves capture as 16 kHz, mono, float32 samples.
- **Reason:** it is the input format of the VAD, ASR and speaker-embedding candidates (Silero, Whisper, ECAPA), so no later module converts again.
- **Alternatives considered:** keeping the device's native format and converting per model (repeated work in several places).

### 6. Stereo is downmixed by averaging the channels (2026-10-23)

- **Decision:** mono = mean of all channels.
- **Reason:** standard, simple and keeps voices that are centred in the mix at full level.
- **Alternatives considered:** keeping the loudest channel (better for a voice on one channel only, but more code and can switch channels mid-phrase).

### 7. Devices are opened as float32 at their own rate and channel count (2026-10-23)

- **Decision:** open each stream with `paFloat32`, the device's native sample rate and its channel count.
- **Reason:** WASAPI shared mode only accepts the device's own rate; float32 means no integer conversion in our code.
- **Alternatives considered:** asking the device for 16 kHz directly (rejected by WASAPI shared mode); int16 samples (an extra conversion step).

### 8. Callback mode instead of a blocking read loop (2026-10-23)

- **Decision:** PortAudio calls `_on_audio` on its own thread.
- **Reason:** no reader thread of our own is needed, and capture runs independently of model work.
- **Alternatives considered:** a thread per source calling blocking `read()` (more threads and shutdown code).
