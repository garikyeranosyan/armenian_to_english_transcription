# TODO — Deferred work

Tasks agreed to be done later, with when they become due. Remove an entry once it is done.

---

## 1. Run the MT benchmark and choose the translation model

- **When:** once the working skeleton runs end to end (all 9 modules connected in `pipeline.py`).
- **Why:** `MT_MODEL` in `translator.py` is provisionally `nllb-200-distilled-600M`. `PLAN.md` requires model choices to be decided by benchmarks on this laptop and recorded in the README.
- **Steps:**
  1. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
  2. `.\.venv\Scripts\python.exe benchmarks\convert_mt.py` (converts the 5 candidates into `models/mt/`)
  3. `.\.venv\Scripts\python.exe benchmarks\mt_benchmark.py`
  4. Pick the most accurate model (chrF++ on ASR input) that fits the 0.5 s translation budget; set `MT_MODEL`.
  5. Add the results table and the choice to the README, and the decision to `DECISIONS.md`.
- **Check while doing it:**
  - `Helsinki-NLP/opus-mt-hy-en` may not exist on the Hub; drop it if conversion says so.
  - If conversion fails with transformers 5, try `transformers==4.57.1`.
  - If MADLAD's scores look broken, rerun it with `--compute-type int8_float16` (T5 models can misbehave in float16).
  - If every model scores very low, check that FLEURS sentence IDs line up between `hy_am` and `en_us`.

## 2. Tune speaker identification on real call audio

- **When:** once the working skeleton runs end to end.
- **Why:** `SIMILARITY_THRESHOLD` in `speakers.py` is a provisional 0.4. `PLAN.md` M4 requires ≥ 2 call-side speakers to keep stable labels, and risk 4 says to tune the threshold on call-quality (compressed) audio.
- **Steps:**
  1. Record a call with at least 2 remote Armenian speakers through the interview call app (kept local, never committed).
  2. Write the speaker benchmark (`benchmarks/`): label purity and number of speakers created, for a range of thresholds.
  3. Set `SIMILARITY_THRESHOLD` to the value that neither splits one person nor merges two; record it in the README.
- **Check while doing it:**
  - Short replies (under ~1 s) are treated like any segment by choice; watch whether they create extra speakers.

## 3. Download and test the speaker embedding model

- **When:** before running the program or tuning speakers (item 2).
- **Why:** `models/wespeaker_resnet34_LM.onnx` is not on the laptop yet (the download through Claude's shell was too slow), and the model has only run on synthetic audio in Claude's workspace.
- **Steps:**
  1. `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` (adds `kaldi-native-fbank`)
  2. `Invoke-WebRequest https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx -OutFile models\wespeaker_resnet34_LM.onnx`
  3. `(Get-FileHash models\wespeaker_resnet34_LM.onnx).Hash` must be `E9848563DA86F263117134DFD7AD63C92355B37DE492B55E325400C9D9C39012`.
  4. `.\.venv\Scripts\python.exe -m pytest` (all tests pass in the Windows `.venv`).
  5. Embed a few FLEURS clips: two halves of the same clip should score clearly higher cosine similarity than clips of different speakers.
- **Check while doing it:**
  - Time `WeSpeakerEncoder.embed` on this laptop for 1, 3 and 10 s of audio: 10 s took 0.28 s on a 2-core machine, over the 0.2 s speaker budget.
