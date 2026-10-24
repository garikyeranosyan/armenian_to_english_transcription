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

## segmenter.py

### 1. Silero VAD run directly with onnxruntime (2026-10-23)

- **Decision:** detect speech with Silero VAD's ONNX model (v6.2.3, MIT licence), loaded with `onnxruntime`; the model file is downloaded once into `models/`.
- **Reason:** most accurate of the candidates on noisy and compressed call audio, and onnxruntime needs no PyTorch for this stage.
- **Alternatives considered:** the official `silero-vad` package (simpler API, but pulls in PyTorch); WebRTC VAD (tiny and fast, but more false triggers on noise and call audio).

### 2. Segments are capped at 10 s (2026-10-23)

- **Decision:** a phrase longer than 10 s is cut there, even without a pause.
- **Reason:** keeps transcription time per segment inside the 5 s latency limit while still fitting most sentences.
- **Alternatives considered:** 6 s (safer for latency, but splits more sentences and can hurt translation); 15 s (fewer cuts, but slower transcription and a higher risk to the limit).

### 3. VAD runs on the CPU (2026-10-23)

- **Decision:** the VAD uses onnxruntime's CPU provider with one thread.
- **Reason:** scoring a 32 ms window is cheap, and the GPU is kept free for transcription and translation, which need it most.
- **Alternatives considered:** running the VAD on the GPU (competes with the heavier models for no real gain).

## transcriber.py

### 1. Transcription model is large-v3-turbo (2026-10-24)

- **Decision:** transcribe with Whisper `large-v3-turbo`.
- **Reason:** in the M1 benchmark (200 FLEURS Armenian clips, this laptop's GPU) it was the most accurate model that fits the 2 s transcription budget: 46.1 % WER, 0.63 s mean and 0.93 s worst case per clip.
- **Alternatives considered:** `large-v3` (42.6 % WER, but 3.11 s mean and 175 of 200 clips over budget); `medium` (56.1 % WER, 123 clips over budget); `small` (81.3 % WER).

### 2. faster-whisper (CTranslate2) as the ASR runtime (2026-10-23)

- **Decision:** run Whisper models with `faster-whisper`.
- **Reason:** several times faster than PyTorch Whisper, supports float16 and int8, uses less memory and needs no PyTorch.
- **Alternatives considered:** Hugging Face `transformers` with PyTorch (runs any fine-tune directly, but slower, heavier and adds about 2.5 GB; faster-whisper needs a one-time conversion for fine-tunes instead).

### 3. The model sits behind a generic SpeechRecognizer interface (2026-10-24)

- **Decision:** the transcriber depends only on a `SpeechRecognizer` interface; the concrete model is set in one constant, `ASR_MODEL`.
- **Reason:** a later benchmark winner can be swapped in without changing the rest of the program, and tests can use a fake model.
- **Alternatives considered:** calling faster-whisper directly throughout the code.

### 4. The model is chosen by a benchmark on FLEURS Armenian (2026-10-23)

- **Decision:** compare candidates with `benchmarks/asr_benchmark.py` on the FLEURS `hy_am` test split: WER, CER, wrong-script output and time per clip against the 2 s budget, run through the program's own recognizer on this laptop's GPU.
- **Reason:** `PLAN.md` requires model choices to be decided by benchmarks on this machine; published scores don't reflect this hardware or this code.
- **Alternatives considered:** choosing from published WER figures.

### 5. GPU first, CPU as the fallback (2026-10-24)

- **Decision:** use CUDA with float16 when a GPU is found; otherwise use the CPU with int8 and log a warning.
- **Reason:** `PLAN.md` requires the GPU first with a CPU fallback; float16 is fastest on the GPU and int8 is fastest on the CPU.
- **Alternatives considered:** requiring a GPU (fails on machines without one); one compute type for both devices (float16 is slow or unsupported on CPU).
