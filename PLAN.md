# PLAN — Live Armenian Transcription → English Translation

> Source: *Mathrix AI Coding Challenge* brief (Oct 21, 2026).
> Budget: ~20 hours over one week. Target machine: **my Windows x64 laptop with GPU** (also the interview device).

---

## 1. Goal

A Python terminal program that listens to a live call and, for every spoken Armenian phrase, prints:

- the **time** of the phrase,
- the **speaker**: `Me` for the microphone; `Speaker 1`, `Speaker 2`, … for people heard through system audio,
- the **Armenian transcript** (`HY:`),
- the **English translation** (`EN:`).

```
[00:01:12] Speaker 1
HY: Բարև, ինչպե՞ս ես։
EN: Hi, how are you?

[00:01:15] Me
HY: Լավ եմ, շնորհակալություն։
EN: I'm fine, thank you.
```

### Definition of done

| Requirement (from the brief) | Measurable target |
|---|---|
| Two audio sources at once | Microphone and system output are captured at the same time; the user picks both devices |
| Armenian transcription | Clear Armenian speech becomes readable Armenian text |
| English translation | Each segment gets an English line that keeps its meaning |
| Speaker identification | Mic is always `Me`. Call-side speakers are told apart (≥ 2 tested) and each keeps the same label for the whole session |
| Live terminal output | Each segment is printed as soon as it is processed, with time, speaker, HY and EN |
| Latency | **≤ 5 s (hard limit)** from the end of a phrase to its EN line |
| Local models only | No cloud speech, translation or diarization services at runtime |
| Runs on the interview laptop | Starts from one command and runs a full call without crashing |

### Constraints

- **Local inference.** Model weights are downloaded once before the interview and are never committed.
- **Windows.** System audio is captured by **WASAPI loopback** of the chosen output device. The user keeps hearing the call normally; no virtual cable is needed.
- **GPU first, CPU fallback.** Models must meet the latency limit on this laptop's GPU. Final model sizes are decided by benchmarks.
- **Armenian is a low-resource language.** Models are chosen by measurement, and the choice is explained in the README.

### Out of scope (not built)

GUI, installer, echo handling, recognising speakers by name or across sessions, saving transcripts, non-Armenian input. Anything else beyond the required features (interim results, confidence scores, colours, etc.) is also left out. Spare time goes into making the core reliable.

---

## 2. Architecture overview

```
 Microphone ──► Capture (mic) ──► Segmenter ──┐
                                              │  speech segments
 System out ──► Capture (loopback) ► Segmenter┤  (tagged with source + time)
                                              ▼
                                  ┌───────────────────────┐
                                  │       Pipeline        │
                                  │  Speaker ID           │  mic → "Me"
                                  │     ↓                 │  loopback → "Speaker N"
                                  │  Transcription (HY)   │
                                  │     ↓                 │
                                  │  Translation (EN)     │
                                  └──────────┬────────────┘
                                             ▼
                                     Terminal output
                         (Metrics records timings at every stage)
```

Principles:

1. **Capture never waits.** Recording both sources keeps going regardless of how busy the models are.
2. **Phrases are cut at pauses.** Voice activity detection ends a segment at a short pause. A maximum length caps long monologues so they can't break the latency limit.
3. **The source decides `Me`.** Mic audio needs no speaker model. Only system audio is clustered into `Speaker N`.
4. **Both sides are processed.** If both sides talk at once, both segments are transcribed and printed in the order they finish, each with its real start time.
5. **Models load and warm up at startup**, before "Listening…" is shown, so the first phrase is not slow.
6. **Every model sits behind a simple interface.** Candidates can be swapped and benchmarked, and models can be mocked in tests.

### Latency budget (hard limit 5 s)

| Stage | Target |
|---|---|
| Pause detection (end of phrase) | ~0.5 s |
| Waiting in the queue | ≤ 0.3 s |
| Speaker identification | ≤ 0.2 s |
| Transcription | ≤ 2.0 s |
| Translation | ≤ 0.5 s |
| Printing | ~0 s |
| **Total** | **≈ 3.5 s** (leaves ~1.5 s headroom) |

---

## 3. Modules

| # | Module | Responsibility | Inputs | Outputs |
|---|---|---|---|---|
| 1 | **Config & devices** | Read command-line options, list audio devices, resolve the user's mic and output choices, fail early with a clear message | Command-line flags; available Windows audio devices | Printed device list (on request); validated settings including the chosen mic and output devices |
| 2 | **Audio capture** | Record the microphone and a WASAPI loopback of the output device at the same time, converted to one common format with a shared session clock | Chosen devices | Two continuous streams of timestamped audio frames, tagged `mic` / `system` |
| 3 | **Segmentation (VAD)** | Turn each stream into phrase-sized speech segments; drop non-speech and tiny bursts; force a cut on very long speech | Audio frames from one source | Speech segments: audio, source, start time, end time |
| 4 | **Speaker identification** | Give each segment a speaker label. `mic` → `Me`. `system` → compare the voice to known call-side speakers, reuse a label or create the next `Speaker N`. Labels are never renamed or merged once printed | Speech segment | Speaker label |
| 5 | **Transcription (ASR)** | Spoken Armenian → Armenian text, with language fixed to Armenian; skip empty or hallucinated output | Segment audio | Armenian text |
| 6 | **Translation (MT)** | Armenian text → English text | Armenian text | English text |
| 7 | **Pipeline** | Connect modules 2–6 and 8–9; keep capture separate from model work; process segments from both sources; keep delays bounded when work piles up; start up, warm up and shut down cleanly on Ctrl+C | Settings; loaded modules | Completed transcript entries (time, speaker, HY, EN) |
| 8 | **Terminal output** | Print each entry as soon as it is ready, with correct Armenian characters | Transcript entry | Lines on the terminal |
| 9 | **Metrics** | Timestamp each segment at every stage; report per-segment latency and session p50 / p95 / max; warn when the 5 s limit is at risk | Stage timestamps from the pipeline | Latency figures (debug view + summary at exit); data for benchmarks and tests |

### Model candidates (final pick by benchmark, Section 5.2)

| Stage | Candidates |
|---|---|
| Segmentation | Silero VAD; WebRTC VAD as a lighter fallback |
| Transcription | Whisper large-v3 and large-v3-turbo; community Whisper fine-tunes for Armenian; Meta MMS (Armenian adapter); SeamlessM4T v2 |
| Translation | NLLB-200 (distilled 600M / 1.3B); MADLAD-400; OPUS-MT hy→en; M2M100 |
| Speaker embeddings | SpeechBrain ECAPA-TDNN; WeSpeaker / pyannote embedding models (no gated access), compared with online clustering by similarity threshold |

How to choose: pick the most accurate model whose measured time fits its stage budget on this laptop. The README records the scores, the alternatives and the trade-off.

---

## 4. Project structure

```
live-hy-en/
├── README.md                # setup, Windows audio setup, model choices, architecture, limitations
├── PLAN.md
├── requirements.txt         # pinned versions
├── .gitignore               # model weights, audio recordings, keys
├── src/live_hy_en/
│   ├── __main__.py          # entry point: start the program
│   ├── config.py            # 1 Config & devices
│   ├── capture.py           # 2 Audio capture (mic + WASAPI loopback)
│   ├── segmenter.py         # 3 Segmentation (VAD)
│   ├── speakers.py          # 4 Speaker identification
│   ├── transcriber.py       # 5 Transcription
│   ├── translator.py        # 6 Translation
│   ├── pipeline.py          # 7 Pipeline
│   ├── output.py            # 8 Terminal output
│   └── metrics.py           # 9 Metrics
├── benchmarks/              # offline model comparisons (ASR, MT, speaker ID, latency)
├── tests/
│   ├── unit/
│   └── e2e/                 # replay of recorded audio through the full pipeline
└── chats/                   # every Claude Cowork session, one .md file each
```

---

## 5. Testing

Each test is derived from a requirement in the brief, not from the code as written.

### 5.1 Requirement → test map

| Requirement | How it is tested |
|---|---|
| User picks the input devices | Unit: a device can be chosen by index or name; an unknown device gives a clear error. Manual: the device list shows the mic and output devices |
| Two sources captured at once | E2E/manual: both sources produce segments in the same session; one source being silent doesn't block the other |
| Armenian transcription | Benchmark: word/character error rate on Armenian test clips (Common Voice hy-AM, FLEURS hy_am, my own conversational recordings) |
| English translation | Benchmark: chrF/BLEU on Armenian–English sentence pairs (FLORES-200) **plus** translation of real ASR output; manual review of conversational phrases |
| Mic is always `Me` | Unit: every mic segment is labelled `Me` and never reaches the call-side speaker logic |
| Call-side speakers kept apart, labels stable | Benchmark: mixes of 2–4 different voices from call-quality audio measure label purity. Unit (mocked voices): a returning speaker keeps their first label; labels are never renamed |
| Output has time, speaker, HY, EN | Unit: each printed entry contains all four; Armenian script prints correctly in the Windows terminal |
| Each segment printed while the call runs | E2E replay: entries appear during playback, not at the end |
| **Latency ≤ 5 s** | E2E replay at real-time speed: every segment is ≤ 5 s (p95 target ≈ 3.5 s), measured by Metrics. Unit (mocked slow models): delays stay bounded under load |
| Local models only | Manual: full run with the network disconnected |
| Runs reliably on the interview laptop | Soak: 20+ minutes of replayed conversation with no crash, no memory growth and no growing delay. Stress: both sides talking at once; a long monologue |

### 5.2 Test layers

1. **Unit tests**: fast, with models mocked behind their interfaces.
2. **Model benchmarks**: offline accuracy and speed on this laptop; the results drive model choice and go into the README.
3. **End-to-end replay**: recorded audio is fed through the full pipeline in place of live capture (normal run, soak, stress).
4. **Live rehearsal**: a real call with at least 2 remote Armenian speakers plus me, using the same call app and laptop as the interview.

Test recordings stay local and are never committed.

---

## 6. Risks and mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | Poor Armenian ASR quality (wrong words, hallucinations, wrong script) | High | Benchmark several models early, including Armenian fine-tunes; fix language to Armenian; filter empty or hallucinated output; be honest about accuracy in the README |
| 2 | Latency exceeds 5 s with the most accurate models | High | Measure speed first; use the GPU; choose smaller/faster variants if needed; cap segment length; run translation alongside the next transcription |
| 3 | Translation errors pile on top of ASR errors | Medium | Test translation on real ASR output, not only clean text; HY line stays visible for judgment |
| 4 | Speaker labels unstable: one person split in two, or two people merged | High | Tune the similarity threshold on call-quality (compressed) audio; a minimum duration before creating a new speaker; rehearse with real call audio |
| 5 | Two call-side speakers inside one segment (fast turn-taking) | Medium | Short pause threshold and maximum segment length; documented as a known limitation |
| 6 | Loopback capture fails or picks the wrong device | High | Clear device list and selection; startup check that both sources produce signal; rehearse on the interview laptop |
| 7 | Device sample rates / channel counts differ | Medium | Always convert to one common format; never assume a device's native format |
| 8 | Work piles up (GPU shared by ASR, MT and speaker models) and delay grows | Medium | Bounded queue with a defined overflow policy; Metrics warn before 5 s is exceeded |
| 9 | Mic picks up the call from the speakers (echo) | Low (out of scope) | Recommend headphones; listed as a limitation |
| 10 | Dependency problems (GPU drivers, library versions, gated models) | Medium | Pinned versions; ungated models; clean-install test on the laptop |
| 11 | Armenian characters break in the Windows terminal | Medium | Force UTF-8 output; test in the terminal used at the interview |
| 12 | Slow startup or need for internet at the interview | High | Pre-download models; warm up at start; offline test |
| 13 | Running over the 20 h budget | Medium | Build a thin end-to-end slice first, then improve quality |
| 14 | Deliverable rules broken (missing chats, committed weights/audio) | High | `.gitignore` check; export every session to `chats/`; final checklist |

---

## 7. Milestones (~20 h)

| # | Milestone | Est. | Done when |
|---|---|---|---|
| M0 | Setup: repo skeleton, environment, both devices captured on Windows | 2 h | Mic and loopback recorded at the same time |
| M1 | ASR + MT benchmarks on this laptop | 4 h | Models chosen, results recorded |
| M2 | Thin slice: mic → segmentation → ASR → MT → terminal | 3 h | Armenian speech prints HY + EN |
| M3 | Both sources + `Me` labelling + shared clock | 2 h | Both sides appear, correctly tagged |
| M4 | Call-side speaker identification | 3 h | ≥ 2 call speakers keep stable labels in replay |
| M5 | Latency: metrics, warm-up, bounded queues | 2 h | Every replayed segment ≤ 5 s |
| M6 | Tests and reliability fixes | 2 h | Suite passes; soak run clean |
| M7 | README, chat exports, live rehearsal | 2 h | Deliverables checklist complete |

---

## 8. Deliverables checklist

- [ ] Git repository with source code
- [ ] README: setup and run (including Windows audio device setup), model choices with alternatives and reasons, short architecture, known limitations and future improvements
- [ ] `chats/`: every Claude Cowork session exported in full, unedited, one `.md` file per session; originals kept available
- [ ] No model weights, audio recordings or API keys in the repository
- [ ] Program installed, models pre-downloaded and audio routing tested on the interview laptop
