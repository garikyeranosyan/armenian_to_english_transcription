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
