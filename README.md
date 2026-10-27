# Live Armenian → English call transcription

A Python terminal program for Windows that listens to a live call and, for every
spoken Armenian phrase, prints when it was said, who said it, what was said in
Armenian and its English translation:

```
[14:02:17] Speaker 1
HY: Բարև, ինչպե՞ս ես։
EN: Hi, how are you?

[14:02:21] Me
HY: Լավ եմ, շնորհակալություն։
EN: I'm fine, thank you.
```

- **Two sources at once:** your microphone is always `Me`; everyone heard through
  the computer's sound output is `Speaker 1`, `Speaker 2`, … and keeps their label
  for the whole session.
- **Live:** each phrase is printed about 1.5–2.5 s after it ends (hard limit 5 s).
- **Local only:** every model runs on the laptop (GPU first, CPU fallback). Nothing
  is sent to a cloud service, and the program runs with the network switched off.

---

## Contents

1. [Setup](#1-setup)
2. [Running the program](#2-running-the-program)
3. [Audio device setup](#3-audio-device-setup)
4. [Architecture and modules](#4-architecture-and-modules)
5. [Models: what was chosen and why](#5-models-what-was-chosen-and-why)
6. [Testing and measured results](#6-testing-and-measured-results)
7. [Known limitations](#7-known-limitations)
8. [What could be improved with more time](#8-what-could-be-improved-with-more-time)

---

## 1. Setup

### Requirements

| | |
|---|---|
| OS | Windows 10/11, x64 (system audio is captured with WASAPI loopback) |
| Python | 3.10 or newer (developed and tested with 3.14.3) |
| GPU | NVIDIA GPU with a recent driver. The CUDA 12 libraries (cuBLAS, cuDNN) come from pip wheels, so no separate CUDA install is needed. Without a GPU the models run on the CPU, but too slowly for the 5 s limit |
| Disk | About 4.2 GB for the models, plus several GB of Python packages (PyTorch, needed only for the one-time model conversion, is the largest) |
| Internet | Only during setup, to download packages and models |

### 1.1 Get the code and install the packages

```powershell
git clone https://github.com/garikyeranosyan/armenian_to_english_transcription.git
cd armenian_to_english_transcription
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins every version and also installs the program itself in
editable mode (`-e .`), so `python -m live_hy_en` works from any folder.
`transformers` and `torch` are only used by the one-time model conversion in 1.3;
the running program never imports them.

### 1.2 (Optional) Hugging Face token

All models and datasets used here are public, so a token isn't required, but
downloads are faster and less likely to be rate-limited with one:

```powershell
copy .env.example .env
# then put your token after HF_TOKEN= in .env (a "Read" token is enough)
```

`.env` is git-ignored and never committed.

### 1.3 Download and convert the models (once)

Models go into `models/`, which is git-ignored. Run these from the project folder:

```powershell
mkdir models -Force

# Voice activity detection: Silero VAD v6.2.3 (2.3 MB)
Invoke-WebRequest https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx -OutFile models\silero_vad.onnx

# Speaker voiceprints: WeSpeaker ResNet34, VoxCeleb (26 MB)
Invoke-WebRequest https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx -OutFile models\wespeaker_resnet34_LM.onnx
(Get-FileHash models\wespeaker_resnet34_LM.onnx).Hash
# must print E9848563DA86F263117134DFD7AD63C92355B37DE492B55E325400C9D9C39012

# Transcription: BuzzASR's Armenian Whisper, converted to CTranslate2 (3.1 GB)
.\.venv\Scripts\python.exe benchmarks\convert_asr.py buzzasr-armenian

# Translation: NLLB-200 distilled 1.3B, converted to CTranslate2 (2.6 GB)
.\.venv\Scripts\python.exe benchmarks\convert_mt.py nllb-200-distilled-1.3B
```

The converters download each model from its publisher's own Hugging Face page and
write it into its own folder (`models\asr\buzzasr-armenian`,
`models\mt\nllb-200-distilled-1.3B`). A failed conversion never leaves a
half-written folder behind. Without arguments, `convert_asr.py` converts both
transcription candidates and `convert_mt.py` all five translation candidates,
which is only needed for the benchmarks.

If a model is missing, the program stops at startup with an error naming it.
For the VAD, speaker and translation models, the error also says how to get the
model.

---

## 2. Running the program

```powershell
.\run.bat
```

`run.bat` switches the console to UTF-8, so Armenian letters show correctly, then
starts `python -m live_hy_en` from the project's `.venv`. Any options are passed
through:

| Option | Meaning |
|---|---|
| `--list-devices` | List microphones and output devices with their numbers, then exit |
| `--mic <number or name>` | Microphone to record (default: the Windows default microphone) |
| `--output <number or name>` | Output device whose sound is captured (default: the Windows default output) |
| `--debug` | Also print each phrase's latency, broken down by pipeline step |

A device can be given by its number from `--list-devices` or by part of its name,
ignoring case, for example `.\run.bat --mic "headset" --output "speakers"`. An
unknown or ambiguous name stops the program with a clear message.

What you'll see:

1. `Loading models…`: all models load and run once to warm up, so the first
   phrase isn't slow.
2. `Listening to '<mic>' and '<output>'. Press Ctrl+C to stop.`
3. One entry per phrase as the call goes on.
4. On Ctrl+C, a latency summary for the session, for example
   `Latency over 187 phrases: p50 1.36 s, p95 2.08 s, max 2.48 s; 0 over the 5 s limit.`

If a phrase takes longer than 4 s, a warning shows which step was slow, before the
5 s limit is reached.

---

## 3. Audio device setup

**No virtual audio cable is needed.** The program records the chosen output
device through WASAPI loopback, Windows' built-in copy of what the device plays,
while you keep hearing the call normally.

1. **Use headphones.** With speakers, your microphone also picks up the other side
   of the call, and their words are printed twice, once as `Me`. Echo
   cancellation is out of scope.
2. **Prefer wired headphones, or Bluetooth for listening only.** When a Bluetooth
   headset's own microphone is in use, Windows switches it to "hands-free" mode,
   which drops both directions to telephone quality and hurts recognition.
3. **Pick the devices:** run `.\run.bat --list-devices`. Under *Output devices*,
   choose the device the call app plays through: the headphones you'll listen
   on, not another output. Under *Microphones*, choose the microphone you talk
   into.
4. **Make the call app use the same output device** as `--output`: in the call
   app's audio settings, or leave both on the Windows default.
5. **Windows sound settings (recommended):**
   - *Sound → More sound settings → Communications*: choose "Do nothing", so
     Windows doesn't lower other sounds during calls.
   - *Microphone → Properties*: turn off "Audio enhancements" if the transcript of
     your own speech is poor.
6. **Check it:** start the program and say a sentence, and play a few seconds of
   any Armenian audio through the output device. A `Me` entry and a `Speaker 1`
   entry should both appear.

Device sample rates and channel counts don't matter: every device is opened in
its own format and converted to 16 kHz mono.

---

## 4. Architecture and modules

### 4.1 How the pieces fit together

```
 Microphone ──► Capture (mic) ───────► Segmenter (mic) ──────┐
                                                             │  speech segments, tagged
 Output dev ──► Capture (loopback) ──► Segmenter (system) ───┤  with source and time
                                                             ▼
                                        bounded queue (8 phrases)
                                                             │
                                  ┌──────────────────────────▼─────────────────┐
                                  │ Transcription thread                       │
                                  │   ASR: Armenian speech → Armenian text     │
                                  │   Speaker ID: mic → "Me",                  │
                                  │               system → "Speaker N"         │
                                  └──────────────────────────┬─────────────────┘
                                        bounded queue (8 transcripts)
                                  ┌──────────────────────────▼─────────────────┐
                                  │ Translation thread                         │
                                  │   MT: Armenian → English                   │
                                  │   Print the entry                          │
                                  └────────────────────────────────────────────┘
                           Metrics timestamps every phrase at every step
```

Design principles:

- **Capture never waits.** PortAudio calls the capture callback on its own thread.
  It converts the audio, puts it on an unbounded queue and returns at once,
  however busy the models are.
- **Phrases are cut at pauses.** Silero VAD finds speech in 32 ms windows. A phrase
  ends after 700 ms of silence, or after only 100 ms once it is longer than 5 s, so
  long speech is split between words. A phrase is never longer than 10 s. About
  300 ms of audio before the speech and up to 200 ms after it are kept, so the
  first and last words aren't clipped.
- **Silent sources don't hold phrases back.** Windows loopback sends nothing while
  the call is quiet, so if a source sends nothing for 0.7 s, its open phrase is
  closed.
- **The source decides `Me`.** Microphone phrases never reach the speaker model.
  Only system audio is compared by voiceprint.
- **Two model stages run side by side.** Translating one phrase overlaps
  transcribing the next. Each stage runs its models from one thread, so no model
  is ever used by two threads at once.
- **Nothing is dropped.** When transcription falls behind, the bounded queue makes
  segmentation wait and logs a warning, while capture keeps recording.
- **Models load and warm up at startup**, before "Listening…" is shown.
- **Every model sits behind a small interface** (`SpeechRecognizer`,
  `TextTranslator`, `VoiceEncoder`, `SpeechDetector`). Candidates can be swapped
  by changing one constant and benchmarked, and the unit tests use fakes.
- **Only Armenian is shown.** The transcriber's language is fixed to Armenian, and
  empty output or output that is mostly not in Armenian script (a known Whisper
  failure on low-resource languages) is dropped before it reaches the speaker
  model or the screen.

### 4.2 Modules (`src/live_hy_en/`)

| Module | Responsibility |
|---|---|
| `__main__.py` | Entry point: parse options, list devices or run a session, show setup errors clearly |
| `config.py` | Command-line options; list devices; resolve the microphone and the output device's loopback by number or name |
| `capture.py` | Record the microphone and WASAPI loopback; convert to 16 kHz mono float32 (streaming soxr resampler); time-stamp frames on one shared session clock |
| `segmenter.py` | Silero VAD (onnxruntime, CPU): cut each source's audio into phrases with their start and end times |
| `speakers.py` | `Me` for the microphone; WeSpeaker voiceprints and online clustering into stable `Speaker N` labels for system audio |
| `transcriber.py` | Armenian speech → Armenian text with faster-whisper (CTranslate2); drop empty or wrong-script output |
| `translator.py` | Armenian → English with CTranslate2 and sentencepiece; the candidate models and their prompt formats |
| `pipeline.py` | Threads and queues connecting the modules; model loading and warm-up; start-up and Ctrl+C shutdown |
| `output.py` | All terminal output, with the console switched to UTF-8 |
| `metrics.py` | Timestamps at every step; per-phrase latency, session p50/p95/max, early warning for slow phrases |

### 4.3 Repository layout

```
├── README.md, PLAN.md         this file; the original plan with requirements and test map
├── DECISIONS.md               design and model decisions with alternatives and reasons
├── TODO.md                    deferred work
├── INSTRUCTIONS.md            coding and working rules for the project
├── requirements.txt           pinned dependencies; pyproject.toml declares the package
├── run.bat                    start the program from the project's .venv
├── src/live_hy_en/            the program (modules above)
├── benchmarks/                offline model comparisons on this laptop
│   ├── asr_benchmark.py         Whisper models and fine-tunes on FLEURS Armenian
│   ├── convert_asr.py           convert Armenian Whisper fine-tunes to CTranslate2
│   ├── mt_benchmark.py          translation candidates on FLEURS, from text and from ASR output
│   ├── convert_mt.py            convert translation candidates to CTranslate2
│   ├── speaker_benchmark.py     speaker labels on simulated calls (Common Voice + Opus codec)
│   └── results/                 transcripts and translations from the benchmark runs
├── tests/
│   ├── unit/                    fast tests, models replaced by fakes
│   ├── e2e/                     real models, recorded audio replayed in real time
│   └── outputs/                 logs of the test and benchmark runs
└── models/                    downloaded and converted models (git-ignored)
```

---

## 5. Models: what was chosen and why

Every model was chosen by measurement on the target laptop: the most accurate
candidate whose speed fits its share of the latency budget (PLAN.md §2:
transcription ≤ 2.0 s, translation ≤ 0.5 s, speaker ID ≤ 0.2 s per phrase, 5 s
hard limit end to end). The full reasoning is in `DECISIONS.md`.

### 5.1 Transcription: `BuzzASR/armenian`

BuzzASR's Armenian model: Whisper large-v3 fine-tuned with its own Armenian
tokenizer (MIT licence, Findings of EMNLP 2026), run with faster-whisper
(CTranslate2, 8-bit weights on the GPU, greedy decoding, at most two retries).

Benchmark: `benchmarks/asr_benchmark.py` on 200 clips (38.5 min) of the FLEURS
`hy_am` test set, timed through the program's own recognizer on the laptop's GPU.
WER is word error rate, CER is character error rate; lower is better.

| Model | WER | CER | Mean time per clip | Max time | Result |
|---|---|---|---|---|---|
| **BuzzASR/armenian, 8-bit, greedy** | **11.4 %** | **3.4 %** | 0.81 s | 7.84 s | **chosen** |
| BuzzASR/armenian, float16, beam 5 | 10.3 % | 3.1 % | 0.80 s | 6.58 s | most accurate, but too slow beside NLLB (below) |
| whisper-turbo-armenian (fine-tune) | 19.5 % | 10.3 % | 0.65 s | 0.96 s | 1.9× the word errors |
| Whisper large-v3-turbo | 45.9 % | 13.6 % | 0.64 s | 0.95 s | 2.4× the word errors |
| Whisper large-v3 | 42.6 % | – | 3.11 s | – | 175 of 200 clips over the 2 s budget |
| Whisper medium | 56.1 % | – | – | – | 123 clips over budget |
| Whisper small | 81.3 % | – | – | – | too inaccurate |

(The large-v3, medium and small figures come from the first benchmark run on
24 Oct; dashes mark values that run didn't keep.)

- **Why 8-bit and greedy:** in float16 with beam search, BuzzASR (2.9 GB) and
  NLLB-1.3B (2.6 GB) together overflowed the laptop's GPU memory, which slowed
  every model: in the 24-minute soak, 4 phrases went over 5 s (max 9.97 s) and
  translation p95 rose from 0.56 s to 1.82 s. 8-bit weights halve the memory and
  greedy decoding, which is how BuzzASR reports its own scores, shortens each
  decode. Together they cost 1.1 points of WER.
- **Speed on live-length phrases:** on the 90 clips up to 10 s (the segmenter's
  cap), BuzzASR takes at most 0.88 s, inside the 2 s budget.
- **Retries:** when a result repeats itself or the model is unsure,
  faster-whisper decodes again at a higher temperature. Its default of up to five
  retries took one phrase to 8.55 s in the soak, and one 22 s benchmark clip to
  7.84 s. The program allows at most two (temperatures 0, 0.2 and 0.4), which
  keeps most of the rescue at a bounded cost.
- **Training data:** BuzzASR was trained on FLEURS and Common Voice. Its test
  split was not used for training, but the recordings and style are the same,
  so its lead on conversational audio may be smaller than on FLEURS.
- **Conversion:** its Armenian tokenizer lists Whisper's special tokens twice,
  and the copies that `transformers` uses have different IDs from the ones the
  model was trained with. It also lacks the empty token by which CTranslate2
  recognises a multilingual model. Converted as is, faster-whisper ran it as
  English-only. `convert_asr.py` now gives each special token its trained ID
  (from the model's `added_tokens.json`), adds the empty token in a freed slot,
  and refuses any conversion whose token layout faster-whisper would misread.
- The previous choice, Chillarmo's turbo fine-tune, scores 17.8 % WER and 9.5 %
  CER against Metric-AI's manually corrected FLEURS transcripts (used by the
  ArmBench-ASR leaderboard), and 12–17 % WER in the end-to-end replay.
- **Not benchmarked here:** Meta MMS, SeamlessM4T v2 and NVIDIA's Armenian
  FastConformer (see section 8).

### 5.2 Translation: `facebook/nllb-200-distilled-1.3B`

Run with CTranslate2 (8-bit weights on the GPU, beam size 4) and sentencepiece.

Benchmark: `benchmarks/mt_benchmark.py` translates the same 200 FLEURS sentences
twice: from the correct Armenian text, and from an ASR model's transcripts,
which is what the program really translates (still the previous model's,
Chillarmo's turbo fine-tune, so the ASR-text column is pessimistic). Both are scored against
FLEURS's English references with chrF++ and BLEU (higher is better).

| Model | chrF++ correct text | chrF++ ASR text | BLEU ASR text | Mean time | p95 time | Result |
|---|---|---|---|---|---|---|
| **NLLB-200 distilled 1.3B, 8-bit** | **63.0** | **55.2** | **30.5** | 0.61 s | 0.90 s | **chosen** |
| NLLB-200 distilled 1.3B, float16 | 63.2 | 55.0 | 29.8 | 0.57 s | 0.87 s | same accuracy, twice the GPU memory |
| NLLB-200 distilled 600M | 60.4 | 52.6 | 27.7 | 0.28 s | 0.43 s | fits 0.5 s, less accurate |
| M2M100 1.2B | 49.1 | 42.4 | 18.3 | – | – | much less accurate |
| OPUS-MT hy→en | 37.5 | 33.2 | 10.7 | 0.15 s | 0.24 s | least accurate; 11 of 200 translations over 0.5 s |
| MADLAD-400 3B | – | – | – | – | – | wouldn't load in the laptop's GPU memory |

NLLB-1.3B takes longer than its 0.5 s share on whole FLEURS sentences: 279 of 400
translations went over in 8-bit (263 in float16). This was accepted because
accuracy matters more here and live phrases are shorter than whole FLEURS
sentences: in the 24-minute soak, translation p95 was 0.65 s. 8-bit was chosen
for memory, not speed: alone on the GPU it is no faster, but it leaves room for
the speech model.

### 5.3 Speaker identification: WeSpeaker ResNet34 + online clustering

- **Voiceprints:** WeSpeaker's VoxCeleb ResNet34 (the ONNX export from
  sherpa-onnx), run with onnxruntime on the CPU, with 80-band log-mel features
  from kaldi-native-fbank. The voiceprint is made from the speech only, without
  the padding around it.
- **Clustering:** each call-side speaker keeps the mean of their voiceprints. A
  phrase joins the most similar speaker if cosine similarity is at least
  `SIMILARITY_THRESHOLD = 0.40`; otherwise it becomes the next `Speaker N`. Labels
  are never renamed or merged once printed.

Benchmark: `benchmarks/speaker_benchmark.py` simulates 60 calls with 2–4 remote
speakers each, 16 turns per call (20 % of them short 0.7 s replies). The calls are
built from 40 Common Voice 27.0 Armenian speakers, and every recording is passed
through the Opus codec at a voice-call bitrate (about 28 kbit/s). *Merged* is the
share of phrases filed under another person's label; *split* is the share away
from their speaker's main label.

| Rules | Threshold | Merged | Split | Labels per call (3 real speakers) |
|---|---|---|---|---|
| **Speech-only voiceprints (chosen)** | **0.40** | **2.5 %** | **5.6 %** | 3.67 |
| Speech-only voiceprints | 0.45 | 0.8 % | 8.8 % | 4.12 |
| Voiceprints including padding (previous) | 0.40 | 3.8 % | 5.9 % | 3.68 |

0.40 was kept, so fewer extra speakers are created. The threshold is the main
trade-off between merging two people and splitting one person. Three other rules
were tried and dropped because they didn't help:

- giving short phrases the nearest existing label (it turned splits into merges)
- updating a voiceprint only on confident matches (the same effect as raising the
  threshold)
- subtracting the call's average voice (splits rose to 18–41 %)

Alternatives considered: SpeechBrain ECAPA-TDNN and pyannote embedding models,
which both need PyTorch at runtime.

### 5.4 Voice activity detection: Silero VAD v6.2.3

Silero VAD's ONNX model, run with onnxruntime on one CPU thread, so the GPU stays
free for the larger models. Chosen over WebRTC VAD, which triggers more often on
noise and compressed call audio, and over the `silero-vad` package, which pulls
in PyTorch.

The segmentation rules from section 4.1 were tuned on the 200 FLEURS clips. The
5 s / 100 ms short-pause rule cut the share of cuts made in the middle of speech
from 89 % to 13 %.

---

## 6. Testing and measured results

Tests are derived from the requirements in PLAN.md §5, not from the code.

```powershell
# Unit tests: fast, models replaced by fakes (201 tests)
.\.venv\Scripts\python.exe -m pytest

# End-to-end: real models, FLEURS audio replayed through the pipeline in real time (~5 min)
.\.venv\Scripts\python.exe -m pytest -m e2e -s

# Soak: 24 minutes of replayed conversation (~25 min, progress every minute,
# log saved to tests\outputs\soak_output.txt)
.\.venv\Scripts\python.exe -m pytest -m soak -s

# Lint
.\.venv\Scripts\python.exe -m ruff check .
```

The end-to-end and soak tests need the models from 1.3 and the FLEURS data,
which `benchmarks\asr_benchmark.py` downloads into `benchmarks\data\`. They are
skipped automatically where those are missing.

Results below were measured on 27 Oct with BuzzASR in 8-bit with greedy
decoding. The replays ran with faster-whisper's default five retries, the soak
with the two-retry limit (5.1). The soak still fails on one phrase in 187
(7.00 s; 8.55 s with five retries), most likely a decode stuck in a loop up to
faster-whisper's 448-token limit; see section 7.

| Requirement | Test | Latest result (27 Oct 2026) |
|---|---|---|
| Latency ≤ 5 s, p95 ≈ 3.5 s | every replay, measured by Metrics | replays: max 2.94 s, p95 1.64–2.19 s; soak: p95 2.09 s, but one phrase 7.00 s (see above) |
| Step budgets (PLAN.md §2) | every replay prints each step's p95 against its budget | all within budget except the accepted overruns below |
| Printed live, nothing lost | alternating call, both sides at once, long monologue | all phrases shown during playback |
| `Me` vs `Speaker N` | replay + unit tests | correct for every phrase |
| Accuracy through the pipeline | every replay | 12.5–12.8 % WER (13.7 % with the previous model) |
| Both sides talking at once | replay | max 2.19 s |
| Long monologue (47 s nonstop) | replay | split into 6 phrases, max 1.68 s |
| Reliable for 20+ min | soak | 187 phrases, none lost; delay growth and memory not checked, as the run stopped at the latency check (26 Oct, previous model: delay not growing, memory +97 MB) |
| Local models only | models load and run with all network connections blocked | no connection attempts |
| No weights, audio or secrets committed | `test_repository.py` | passes |

**Accepted step overruns.** The 5 s limit still holds with a wide margin:

- *Pause detection* is about 0.7–1.2 s instead of about 0.5 s, because a phrase
  ends after 700 ms of silence so sentences aren't split at natural pauses.
- *Translation* is about 0.5–1.0 s instead of 0.5 s (NLLB-1.3B, see 5.2).
- When both sides talk at once, one phrase waits for the other's transcription.

---

## 7. Known limitations

- **One phrase in about 200 can still miss the 5 s limit.** When greedy decoding
  gets stuck in a loop, faster-whisper keeps generating up to 448 tokens and then
  retries, although a 10 s Armenian phrase needs only about 40 of BuzzASR's
  tokens. In the soak this took one phrase in 187 to 7.00 s. Capping the tokens
  per decode (about 128) is the planned fix; it is not yet measured.
- **Accuracy on real conversation is unmeasured and likely lower.** All ASR and MT
  numbers come from FLEURS, which is read Wikipedia-style sentences, and BuzzASR
  was trained on FLEURS-style audio. On the ArmBench-ASR leaderboard, the previous
  ASR model (Chillarmo's turbo fine-tune) scores about 25 % WER on FLEURS but
  about 82 % on film dialogue; BuzzASR isn't on that leaderboard. Fast, casual or accented speech, Western
  Armenian and noisy calls will produce more errors.
- **No rehearsal with a real call yet.** Speaker labels and latency were measured
  on simulated and replayed audio, not on a call with real remote participants.
- **Translation errors.** Wrong word senses (for example "maria" → "water"), a
  dropped second sentence when one phrase contains two, «» quotes appearing as
  "⁇", and half-sentence translations when a long sentence is split into two
  phrases.
- **Speaker labels aren't perfect.** On simulated calls, 2.5 % of phrases were
  filed under the wrong person and 5.6 % were split off. Similar voices can merge,
  and short replies and two people in one phrase (fast turn-taking) are the
  hardest cases. Labels last for one session only and aren't names.
- **Echo.** With loudspeakers, the microphone hears the call, and those phrases
  appear again as `Me`. Use headphones.
- **One model run at a time.** When both sides speak at once, the second phrase
  waits for the first one's transcription, about 1 s.
- **Windows only.** Capture relies on WASAPI loopback through PyAudioWPatch.
- **Armenian only.** Phrases in other languages are dropped or transcribed as
  Armenian. Foreign names inside Armenian speech are kept.
- **The GPU is effectively required** for the 5 s limit. The CPU fallback works,
  but it's too slow for live use with these models.
- **Setup is heavy.** The one-time model conversion needs PyTorch and
  transformers, which take several GB.

## 8. What could be improved with more time

**Transcription**

- Build a test set of real conversational and call audio, recorded and
  transcribed locally, and benchmark on it instead of read speech.
- Benchmark NVIDIA's `stt_hy_fastconformer_hybrid_large_pc`: the best open model
  on ArmBench-ASR, at 19.9 % average WER against 33.0 % for the previous model
  (BuzzASR isn't listed), and much better on film dialogue. Also benchmark SeamlessM4T v2, which also
  translates speech directly to English.
- Fine-tune on conversational and call-quality Armenian, for example with LoRA.

**Translation**

- Try MADLAD-400 3B in 8-bit, which may then fit in GPU memory.
- Split multi-sentence phrases at "։" before translating (fixes dropped
  sentences), and normalise «» quotes.
- Translate a sentence that was split across two phrases as one, or give the
  translator the previous phrase as context.

**Speaker identification**

- Tune the threshold on a recorded call with several real remote speakers.
- Compare stronger voiceprint models (CAM++, TitaNet) on the same benchmark.
- Detect speaker changes inside a phrase, for fast turn-taking.

**Pipeline and usability**

- Process the two sources' phrases in parallel when both sides talk at once.
- Cut long phrases at the quietest point instead of at the 10 s cap.
- One setup script that downloads, converts and checks every model.
- A check at startup that both sources actually deliver sound.
