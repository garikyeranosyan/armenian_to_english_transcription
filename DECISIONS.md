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

## translator.py

### 1. CTranslate2 with sentencepiece as the MT runtime (2026-10-24)

- **Decision:** run translation models with CTranslate2 and tokenize with sentencepiece; float16 on the GPU, int8 on the CPU.
- **Reason:** CTranslate2 is already a dependency through faster-whisper, it is fast and memory-efficient, and it runs every candidate family (NLLB, M2M100, MADLAD-400, OPUS-MT) without loading PyTorch at runtime.
- **Alternatives considered:** Hugging Face `transformers` with PyTorch (runs every model directly, but slower, heavier and adds PyTorch to the running program).

### 2. Candidates are converted once from the publishers' own models (2026-10-24)

- **Decision:** `benchmarks/convert_mt.py` converts each candidate from its original Hugging Face upload into `models/mt/`, in the project `.venv`; `transformers` and `torch` are pinned in `requirements.txt` but only the converter imports them.
- **Reason:** the weights come from Meta, Google and Helsinki-NLP rather than third parties, and every candidate can be converted the same way.
- **Alternatives considered:** community pre-converted CTranslate2 uploads (no conversion step, but third-party files and not every candidate has one); a separate conversion virtual environment (keeps PyTorch out of `.venv`, but extra setup).

### 3. The model sits behind a generic TextTranslator interface (2026-10-24)

- **Decision:** the translator depends only on a `TextTranslator` interface; the concrete model is set in one constant, `MT_MODEL`, and each model family's prompt format (language codes or tags) is data in `MT_MODELS`.
- **Reason:** the benchmark winner can be swapped in without changing the rest of the program, and tests can use a fake model.
- **Alternatives considered:** calling CTranslate2 directly throughout the code; one adapter class per model family (more code for what is only a difference in prompt tokens).

### 4. The model is chosen by a benchmark on clean text and ASR output (2026-10-24)

- **Decision:** `benchmarks/mt_benchmark.py` translates the FLEURS `hy_am` sentences twice, from the reference Armenian and from the large-v3-turbo transcripts, and scores both against FLEURS `en_us` (same FLORES-200 sentences, joined by ID) with sacrebleu chrF++ and BLEU, plus time per sentence against the 0.5 s budget.
- **Reason:** `PLAN.md` requires model choices by benchmark on this laptop and testing translation on real ASR output (risk 3); chrF++ and BLEU are the standard scores, comparable with published FLORES-200 results.
- **Alternatives considered:** clean text only (hides how a model copes with ASR errors); an in-house chrF (no dependency, but scores may not match published numbers).

### 5. The final model is chosen once the skeleton runs (2026-10-24)

- **Decision:** `MT_MODEL` is provisionally `nllb-200-distilled-600M`; the benchmark is run and the model chosen once all modules are connected (`TODO.md`).
- **Reason:** gets the end-to-end program working first (`PLAN.md` risk 13: build a thin slice first), then picks the model with the full latency picture.
- **Alternatives considered:** running the benchmark before building the remaining modules.

### 6. Translation model is NLLB-200 distilled 1.3B, over its 0.5 s share (2026-10-26)

- **Decision:** `MT_MODEL` is `nllb-200-distilled-1.3B`, replacing the provisional choice in entry 5, and accepting that translation alone takes longer than its 0.5 s share of the latency budget: mean 0.57 s, p95 0.87 s and max 1.41 s per FLEURS sentence on this laptop's GPU (263 of 400 translations over 0.5 s).
- **Reason:** it is the most accurate candidate on 200 FLEURS sentences: chrF++ 55.0 from our ASR transcripts (`whisper-turbo-armenian`) and 63.2 from correct Armenian. The hard rule is the 5 s limit per phrase, and the end-to-end replay with this model stays far inside it (max 2.18 s, p95 1.70 s and 2.01 s), because live phrases are shorter than whole FLEURS sentences and translation overlaps the next transcription.
- **Alternatives considered:** `nllb-200-distilled-600M` (fits the share at p95 0.43 s with 4 of 400 over, but chrF++ 52.6); `m2m100_1.2B` (chrF++ 42.4); `opus-mt-hy-en` (chrF++ 33.2, 11 of 200 over 0.5 s); `madlad400-3b-mt` (would not load on this laptop's GPU); NLLB-1.3B in `int8_float16` (probably faster, not measured).

## speakers.py

### 1. WeSpeaker ResNet34 voiceprints run with onnxruntime (2026-10-24)

- **Decision:** compute speaker embeddings with WeSpeaker's VoxCeleb ResNet34 (ONNX export from sherpa-onnx, downloaded once into `models/`), run with onnxruntime; its 80-band log-mel features are computed with `kaldi-native-fbank` using WeSpeaker's training settings.
- **Reason:** a strong, ungated speaker model that runs on the onnxruntime already used by the VAD, so no PyTorch at runtime; `kaldi-native-fbank` is small, has Windows wheels and matches Kaldi's features.
- **Alternatives considered:** SpeechBrain ECAPA-TDNN (simple API, but loads PyTorch, torchaudio and speechbrain at runtime); pyannote embedding models (also PyTorch-based).

### 2. Online clustering with running-mean voiceprints and a cosine threshold (2026-10-24)

- **Decision:** each call-side speaker keeps the mean of their voiceprints; a segment joins the most similar speaker if cosine similarity reaches `SIMILARITY_THRESHOLD`, otherwise it becomes the next `Speaker N`. Labels are never renamed or merged once given.
- **Reason:** decides as each segment arrives, costs the same however long the call runs, and keeps the labels already printed stable, as `PLAN.md` requires.
- **Alternatives considered:** comparing with every past segment (copes better with a varying voice, but cost grows over the call and one noisy segment can mislabel later ones); offline re-clustering (better grouping, but would rename labels already printed).

### 3. Short segments are treated like any other (2026-10-24)

- **Decision:** there is no minimum length for a call-side segment to create a new speaker or update a voiceprint.
- **Reason:** keeps the rule simple; whether short replies cause extra speakers is checked when tuning on real call audio.
- **Alternatives considered:** below about 1 s, joining the nearest speaker without creating one or updating its voiceprint (`PLAN.md` risk 4 mitigation, at the cost of an extra rule).

### 4. The speaker model runs on the CPU (2026-10-24)

- **Decision:** onnxruntime's CPU provider runs the speaker model.
- **Reason:** keeps the GPU for transcription and translation, which need it most; the model is small.
- **Alternatives considered:** the GPU (would need `onnxruntime-gpu` and compete with the heavier models).

### 5. The similarity threshold is tuned on real call audio later (2026-10-24)

- **Decision:** `SIMILARITY_THRESHOLD` is provisionally 0.4 and is tuned with a speaker benchmark on recorded call audio once the skeleton runs (`TODO.md`).
- **Reason:** the right value depends on call-app compression and the actual voices (`PLAN.md` risk 4), which only a real call recording shows.
- **Alternatives considered:** tuning now on public datasets (clean studio audio, not call quality).

## output.py

### 1. Entries show the wall-clock time the phrase started (2026-10-24)

- **Decision:** each entry is printed as `[HH:MM:SS] Speaker`, then `HY:` and `EN:` lines, where the time is the local time of day at the start of the phrase (session start time plus the phrase's session seconds).
- **Reason:** easy to match with the call app and with what was said when; both sources still share one clock.
- **Alternatives considered:** time since the session started, as in the `PLAN.md` example (matches the brief's sample, but harder to relate to the call).

### 2. All terminal output goes through output.py (2026-10-24)

- **Decision:** transcript entries, status lines, errors and the `--list-devices` list are all printed by `output.py`, which also switches the terminal to UTF-8 for Armenian.
- **Reason:** keeps the `INSTRUCTIONS.md` rule that printing happens only in `output.py`, and one place handles encoding (`PLAN.md` risk 11).
- **Alternatives considered:** one `print` for the device list in the entry point (more direct, but an exception to the rule).

## pipeline.py

### 1. Two model stages: transcription, then translation (2026-10-24)

- **Decision:** a segmenter thread per source feeds one transcription thread (ASR, then speaker label), which feeds one translation thread (MT, then printing).
- **Reason:** translating one phrase overlaps transcribing the next (`PLAN.md` risk 2), while each stage stays sequential so models are never run from two threads at once.
- **Alternatives considered:** one worker running every model per phrase (simplest, but each phrase also waits for the previous translation).

### 2. The bounded segment queue never drops phrases (2026-10-24)

- **Decision:** the queue between segmentation and transcription holds 8 phrases; when it is full, segmentation waits and a warning is logged, while capture keeps recording into its own unbounded queue.
- **Reason:** no speech is ever lost from the transcript.
- **Alternatives considered:** dropping the oldest phrase (stays close to live, but loses speech); dropping the newest (keeps order, but loses speech and stays behind).

### 3. Phrases are transcribed before their speaker is assigned (2026-10-24)

- **Decision:** the speaker label is computed only for phrases whose transcript is kept.
- **Reason:** noise and hallucinations dropped by the transcriber never create a `Speaker N` or shift a voiceprint.
- **Alternatives considered:** labelling first, as drawn in the `PLAN.md` diagram (wastes a speaker-model run on dropped phrases and lets noise create speakers).

### 4. The program is an installable package started with python -m live_hy_en (2026-10-24)

- **Decision:** `pyproject.toml` declares the `live-hy-en` package from `src/`, and `requirements.txt` installs it in editable mode (`-e .`), so `python -m live_hy_en` works from any folder.
- **Reason:** `PLAN.md` asks for one command to start; the editable install also needs no reinstall after code changes.
- **Alternatives considered:** setting `PYTHONPATH=src` before each run (an extra step); adding `src` to `sys.path` inside the program (a path hack in the code).
