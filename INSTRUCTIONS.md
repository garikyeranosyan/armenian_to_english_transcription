# INSTRUCTIONS — Rules for working on this project

These instructions apply to every Claude Cowork session working on this project. They are derived from `PLAN.md`. If anything here conflicts with `PLAN.md`, `PLAN.md` wins for *what* to build, and this file wins for *how* to build it.

---

## 1. Project

- **Repository:** https://github.com/garikyeranosyan/armenian_to_english_transcription
- **Default branch:** `main` (receives changes only through pull requests)
- **Working branch:** `dev`
- **Local folder:** `Desktop\armenian_to_english_transcription`
- **What it is:** a Python terminal program that captures the microphone and system audio (WASAPI loopback) on Windows. It transcribes spoken Armenian, translates it to English and prints each segment with time, speaker (`Me` / `Speaker N`), `HY:` and `EN:`, at most **5 s** after the phrase ends. All models run locally.
- **Target machine:** this Windows x64 laptop with GPU (also the interview device). Use the GPU first and fall back to the CPU.

Read `PLAN.md` before starting any task. It defines the goal, the 9 modules, their inputs and outputs, the latency budget, the tests and the risks.

---

## 2. Scope rules

1. **Build only what the brief and `PLAN.md` require.** The required features are: two audio sources at once, Armenian transcription, English translation, speaker identification (`Me` + stable `Speaker N`) and live terminal output.
2. **No additional features.** No GUI, installer, echo handling, speaker names, cross-session speaker memory, transcript saving or non-Armenian input. Also no interim results, confidence scores, colours, caching, extra CLI options or "nice to have" extras. If something seems useful but isn't required, mention it to the user instead of building it.
3. **Spare time goes into reliability** of the core (accuracy, latency, stability), not new features.
4. **Local models only.** Never call cloud speech, translation or diarization APIs.
5. **The 5 s latency limit is a hard rule.** Any change that may affect latency must be measured with the metrics module.
6. **Model choices are decided by benchmarks** on this laptop, and recorded with alternatives and reasons for the README.

---

## 3. Architecture rules

1. Keep exactly the **9 modules** from `PLAN.md`, one responsibility each:

   | Module file | Responsibility |
   |---|---|
   | `config.py` | Command-line options, device listing and selection |
   | `capture.py` | Mic + WASAPI loopback capture, common format, shared clock |
   | `segmenter.py` | Voice activity detection → speech segments |
   | `speakers.py` | `Me` for mic; `Speaker N` for system audio |
   | `transcriber.py` | Armenian speech → Armenian text |
   | `translator.py` | Armenian text → English text |
   | `pipeline.py` | Connects modules, concurrency, bounded queues, start-up and shutdown |
   | `output.py` | Terminal printing |
   | `metrics.py` | Stage timestamps, latency figures |

2. Follow the project structure in `PLAN.md` (`src/live_hy_en/`, `benchmarks/`, `tests/unit/`, `tests/e2e/`, `chats/`). Don't add new top-level folders or modules without a reason in `PLAN.md`.
3. Each module talks to the others only through its inputs and outputs from `PLAN.md`. No hidden shared state.
4. Every model (VAD, ASR, MT, speaker embeddings) sits behind a small interface, so it can be swapped in benchmarks and mocked in tests.
5. Audio capture must never wait on model work.
6. Models load and warm up once at startup.

---

## 4. Python coding style

Follow **PEP 8** throughout. Check with a linter and formatter (e.g. `ruff` and `ruff format` / `black`, line length 88).

### Naming

| Kind | Style | Example |
|---|---|---|
| Modules and packages | `lowercase`, short, underscores only if needed | `transcriber.py`, `live_hy_en` |
| Classes | `CapWords` | `SpeakerTracker`, `SpeechSegment` |
| Functions and methods | `snake_case`, verb first | `assign_speaker()`, `list_devices()` |
| Variables and arguments | `snake_case`, descriptive | `segment_audio`, `end_time` |
| Constants | `UPPER_SNAKE_CASE`, module level | `SAMPLE_RATE_HZ`, `MAX_LATENCY_S` |
| Private names | leading underscore | `_update_centroid()` |
| Booleans | read as a question | `is_speech`, `has_signal` |

- Put units in names where they matter: `_s`, `_ms`, `_hz` (e.g. `silence_ms`).
- No single-letter names except trivial loop indices; no unclear abbreviations.

### Code

- Python 3.10+; **type hints** on all public functions and classes.
- Use **docstrings** (short, one line where possible) for public modules, classes and functions. Comments explain *why*, not *what*.
- Imports at the top, grouped: standard library → third-party → local; no wildcard imports.
- Use `dataclasses` for data passed between modules (segments, transcript entries).
- Use `logging`, not `print`, for diagnostics; `print`/terminal output only in `output.py`.
- Handle errors at module boundaries with clear messages; never silently swallow exceptions.
- No magic numbers: tunable values (thresholds, durations) are named constants or settings.
- Keep functions short and focused; prefer plain code over clever code.

---

## 5. Code quality rules

1. **No redundant code.** No dead code, unused imports, variables or parameters, commented-out blocks, duplicated logic or speculative abstractions "for later".
2. **No over-engineering.** Implement the simplest thing that meets `PLAN.md`; add configuration only where the brief or plan needs it.
3. **One way to do each thing.** Reuse existing helpers instead of writing parallel versions.
4. **Pin dependencies** in `requirements.txt`; don't add a library unless it is needed by a planned module.
5. **Keep the virtual environment in step.** The project uses a Windows virtual environment in `.venv/` (never committed). Whenever a change adds or changes a dependency, update `requirements.txt` in the same change and update `.venv` with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`; if Claude cannot run it, Claude gives the user this command.
6. Before finishing a change: run the linter and the tests, and remove anything left over from debugging.

---

## 6. Testing rules

- Tests come from the **requirements in `PLAN.md` §5**, not from the code as written.
- Use `pytest`. Unit tests mock models behind their interfaces and must run fast.
- Every required feature has at least one test from the requirement → test map in `PLAN.md`.
- Latency is checked with end-to-end replay of recorded audio, measured by `metrics.py`.
- Test audio stays local and is never committed.

---

## 7. Repository and git rules

- **Never commit** model weights, audio recordings, API keys or other secrets; `.gitignore` must cover them.
- Commit small, focused changes with clear messages in the imperative mood (e.g. `Add WASAPI loopback capture`).
- **Work only on the `dev` branch.** Never commit or push directly to `main`; changes reach `main` only through a pull request from `dev` to `main`.
- Commit, push or open a pull request only when the user asks.
- Keep the README up to date with the brief: setup and run (including Windows audio setup), model choices with alternatives and reasons, architecture, known limitations and improvements.
- `chats/` must contain **every** Claude Cowork session for this project, exported in full and unedited, one `.md` file per session. No other AI assistant may be used.

---

## 8. Working with Claude

- Explain decisions and trade-offs honestly; flag limitations instead of hiding them.
- When a requirement is unclear, ask the user rather than guessing or adding features.
- Verify claims (model quality, latency, library behaviour) by measuring or checking sources, not by assuming.
