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

### 4. Audio around each phrase is kept: about 300 ms before, up to 200 ms after (2026-10-25)

- **Decision:** a segment's audio starts up to 300 ms before speech was detected and ends up to 200 ms after it; `start_s` and `end_s` still mark the speech itself, and `speech_samples` gives the speech without the padding.
- **Reason:** live, the first word was cut off in most phrases: Silero scores a soft first sound as speech only after it has begun, and Whisper drops words cut right at a segment's edge. The tail comes from silence the segmenter already waits for, so it adds no latency, and latency is still measured from the real end of speech.
- **Alternatives considered:** one 32 ms window before the speech (the previous rule, which clipped first words); a 500 ms lead-in (catches even slower starts, but adds more noise and silence to each phrase).

### 5. A phrase ends after 700 ms of silence (2026-10-25)

- **Decision:** `END_SILENCE_MS` is 700 ms instead of 500 ms.
- **Reason:** 500 ms split sentences at natural pauses in conversation, giving the transcriber and translator half sentences. It costs about 0.2 s per phrase: pause detection has a p95 of 0.7–1.2 s, over the ~0.5 s step budget in `PLAN.md`, which is accepted because whole phrases still finish within 2.5 s.
- **Alternatives considered:** 500 ms (the original, faster but more mid-sentence splits); 800 ms or more (fewer splits, slower).

### 6. Long phrases end at a 100 ms pause once they pass 5 s (2026-10-25)

- **Decision:** after 5 s of speech, a gap of 100 ms between words ends the phrase; the 10 s cap stays as the hard limit.
- **Reason:** the 10 s cap cut words in half. Measured with Silero on 200 FLEURS clips, cuts inside speech fell from 82 of 92 (89 %) to 16 of 125 (13 %), and forced cuts at 10 s from 87 to 18. The cost is more cuts (92 to 125), so translation gets more partial sentences.
- **Alternatives considered:** 250 ms after 6 s (still 66 % of cuts inside speech, since readers rarely pause that long); 100 ms after 6 s (105 cuts, 24 inside speech); cutting at the quietest point before the cap (not tried); a 15 s cap (fewer cuts, but slower transcription and still mid-word).

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

### 6. Transcription model is the Armenian fine-tune whisper-large-v3-turbo-armenian (2026-10-24)

- **Decision:** `ASR_MODEL` is `Chillarmo/whisper-large-v3-turbo-armenian`, converted once to CTranslate2 by `benchmarks/convert_asr.py`; this replaces entry 1.
- **Reason:** on the same 200 FLEURS clips it has 19.5 % WER and 10.3 % CER against 45.9 % and 13.6 % for large-v3-turbo, for about 0.02 s more per clip (p95 0.87 s, max 1.01 s, inside the 2 s budget). Against Metric-AI's corrected FLEURS references its WER is 17.8 %, and through the full pipeline in the end-to-end replay 12–17 %.
- **Alternatives considered:** large-v3-turbo (2.4 times the word errors); `BuzzASR/armenian` (its non-standard tokenizer converts without language tokens, so faster-whisper runs it as English-only; `convert_asr.py` now rejects such models); large-v3 (too slow); Meta MMS, SeamlessM4T v2 and NVIDIA's Armenian FastConformer (not benchmarked; FastConformer is the best open model on ArmBench-ASR).

### 7. Transcription model is BuzzASR/armenian, with its token layout repaired (2026-10-27)

- **Decision:** `ASR_MODEL` is `BuzzASR/armenian` (Whisper large-v3 fine-tuned with its own Armenian tokenizer, MIT licence); this replaces entry 6. `benchmarks/convert_asr.py` gives each special token the ID the model was trained with (from its `added_tokens.json`), names a freed slot with the empty token CTranslate2 uses to recognise a multilingual model, and refuses any conversion whose layout faster-whisper would misread.
- **Reason:** on the 200 FLEURS clips it has 10.3 % WER and 3.1 % CER against 19.5 % and 10.3 % for entry 6's model, at 0.80 s mean per clip (at most 1.08 s on clips up to 10 s). Converted as is, its tokenizer gave Whisper's special tokens the wrong IDs and the model had no empty token, so faster-whisper ran it as English-only.
- **Alternatives considered:** keeping `whisper-turbo-armenian` (faster and lighter on the GPU, but about twice the word errors); rejecting BuzzASR as before; running it through `transformers` and PyTorch, which read its generation config directly (slower, and PyTorch at runtime).

### 8. The speech model runs in 8-bit with greedy decoding on the GPU (2026-10-27)

- **Decision:** on the GPU, `GPU_COMPUTE_TYPE` is `int8_float16` and `BEAM_SIZE` is 1.
- **Reason:** in float16 with beam 5, BuzzASR (2.9 GB) beside NLLB-1.3B (2.6 GB) overflowed the GPU memory and slowed every model: in the soak, 4 phrases went over 5 s (max 9.97 s) and translation p95 rose from 0.56 s to 1.82 s. With 8-bit and greedy decoding, the replays stay under 2.94 s and the soak p95 is 2.09 s, for 11.4 % WER instead of 10.3 %. Greedy decoding is also how BuzzASR reports its own scores.
- **Alternatives considered:** float16 with beam 5 (most accurate, misses the 5 s limit); 8-bit with beam 5 (not measured); a shorter 6 s phrase cap (more split sentences, see `segmenter.py` 2); going back to `whisper-turbo-armenian`.

### 9. Unclear phrases are retried at most twice (2026-10-27)

- **Decision:** faster-whisper decodes at temperature 0 and, if the text repeats itself or the model is unsure, retries at 0.2 and then 0.4 (`TEMPERATURES`), instead of its default of up to five retries.
- **Reason:** each retry is a full decode; the default took one soak phrase to 8.55 s and one 22 s benchmark clip to 7.84 s, while other phrases take about 1 s. Two retries keep most of the rescue for loops and unsure output at a bounded cost.
- **Alternatives considered:** no retries (fastest and most predictable, but nothing rescues a greedy loop); keeping the default (breaks the 5 s limit on rare phrases); BuzzASR's repetition controls (`no_repeat_ngram_size`, `repetition_penalty`) instead of retries (not measured).

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

### 7. The translation model runs in 8-bit on the GPU (2026-10-27)

- **Decision:** `GPU_COMPUTE_TYPE` is `int8_float16` for NLLB-200 distilled 1.3B.
- **Reason:** it halves the model's GPU memory, so it fits beside BuzzASR (`transcriber.py` 8); on the 200 FLEURS sentences its scores stay the same (chrF++ 63.0 against 63.2 from correct text, 55.2 against 55.0 from ASR text), at 0.61 s mean against 0.57 s.
- **Alternatives considered:** float16 (no more accurate, and with BuzzASR it overflows the GPU); NLLB-600M (fits easily, chrF++ 52.6 from ASR text).

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

### 6. Voiceprints are made from the speech alone; the threshold stays 0.40 (2026-10-25)

- **Decision:** the voiceprint is computed from a segment's speech without its padding (entry 4 of `segmenter.py`), and `SIMILARITY_THRESHOLD` stays 0.40, now chosen by `benchmarks/speaker_benchmark.py`. This settles entry 5, and entry 3 stays as it is.
- **Reason:** the benchmark simulates 60 calls of 2 to 4 remote speakers from 40 Common Voice 27.0 Armenian speakers, sent through the Opus codec at about 28 kbit/s, with 20 % short replies. At 0.40, speech-only voiceprints cut phrases filed under the wrong person from 3.8 % to 2.5 % with splits about the same (5.9 % to 5.6 %). 0.40 was kept over 0.45 so that fewer extra speakers appear.
- **Alternatives considered:** 0.45 (0.8 % merged, but 8.8 % split, about one extra label per call); short phrases taking the nearest label without creating or changing a speaker (mislabelled 26 to 38 of 187 short replies, turning splits into merges); updating a voiceprint only on clear matches (the same trade-off as raising the threshold, with an extra rule); subtracting the call's average voiceprint (splits rose to 18 to 41 %).

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

### 5. A phrase is closed when its source sends nothing for 0.7 s (2026-10-26)

- **Decision:** each segmenter thread waits for frames with a 0.7 s timeout (the end-of-phrase pause) and, when none arrive, closes the open phrase.
- **Reason:** WASAPI loopback sends nothing while the call is silent, so the segmenter used to notice the end of the last phrase only when the next frame came, when someone on the call spoke again: one phrase took 16 s in the replay.
- **Alternatives considered:** waiting for the next frame (the bug); a 0.1 s timeout like the segmenter's frame-gap rule (faster, but a delayed audio callback could cut a phrase short).

## tests

### 1. End-to-end tests share one set of models, loaded with the network blocked (2026-10-26)

- **Decision:** a session fixture loads every model once with all network connections blocked; every end-to-end test, the soak and the offline test use that set.
- **Reason:** two sets of models don't fit in the laptop's GPU memory and slow every model 3 to 10 times, which made the replays fail on latency; loading with the network blocked also proves the `PLAN.md` requirement of local models only.
- **Alternatives considered:** each test file loading its own models; checking local-only behaviour by hand with the network switched off.

### 2. Step budgets are checked, with listed overruns accepted (2026-10-26)

- **Decision:** every replay checks the 5 s limit, the 3.5 s p95 target and each step's p95 against the `PLAN.md` §2 budget; pause detection and translation are allowed over budget, and the queue too when both sides talk at once.
- **Reason:** the 5 s limit is the hard rule and holds with a wide margin (max 2.48 s over 24 minutes); the overruns come from deliberate choices (`segmenter.py` 5, `translator.py` 6, `pipeline.py` 1) and are printed, not hidden.
- **Alternatives considered:** failing on any step over budget (would reject those choices); checking only the total (hides which step is slow).
