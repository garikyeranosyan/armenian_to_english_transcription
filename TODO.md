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

## 4. Fix wrong Armenian transcripts in live use

- **When:** now; it blocks everything downstream (translation is only as good as its Armenian input).
- **Why:** in live testing almost nothing is transcribed correctly, and some phrases are dropped with "Dropped transcript not in Armenian script". The ASR benchmark on clean read speech was already weak (large-v3-turbo: 46 % WER, 13.7 % CER), and live use adds microphone/call audio, conversational speech and our segmentation.
- **Checklist — first find which stage is at fault:**
  - [ ] Note where it fails: mic (`Me`), call side (`Speaker N`), or both; headphones or speakers; which call app.
  - [ ] Check the captured audio itself: save a few live segments to WAV (a diagnostic script in `benchmarks/`, output git-ignored) and listen. Wrong pitch or speed means a sample-rate bug; distortion means clipping or a channel problem; very quiet means mic gain.
  - [ ] Run those saved segments through `benchmarks/asr_benchmark.py`-style transcription offline: bad there too means the model; fine there means the live pipeline.
  - [ ] Run the end-to-end replay (`pytest -m e2e -s`): FLEURS through the full pipeline vs. the whole-clip benchmark shows how much segmentation costs.
  - [ ] Look at what the dropped "not in Armenian" texts were (Latin transliteration, Russian, English?) to see what Whisper falls back to.
- **Checklist — fixes to try, measured on the same recordings:**
  - [ ] Segmentation: cut long phrases at the quietest point instead of the hard 10 s cap (30 % of FLEURS segments are cut mid-speech); consider a longer cap and a longer minimum phrase (11 of 291 FLEURS segments were under 1 s).
  - [ ] Decoding options: an Armenian `initial_prompt`, temperature fallback, `no_speech_threshold` / `log_prob_threshold`, beam size.
  - [ ] Bigger model: `large-v3` (42.6 % WER, but 3.1 s mean per clip — over the 2 s budget on this GPU; check int8_float16).
  - [ ] Armenian-specific models: community Whisper fine-tunes for Armenian (one-time CTranslate2 conversion), Meta MMS, SeamlessM4T (handover open item 5).
  - [ ] Benchmark normalisation: map `եվ` → `և` so WER isn't inflated (handover open item 3).
  - [ ] Record the chosen model and settings in `DECISIONS.md` and the README.

## 5. Fix incorrect English translations

- **When:** after item 4 is measured, since most translation errors come from wrong Armenian input.
- **Why:** live translations are not correct. NLLB-600M scores chrF 62 on correct Armenian but only 44 on Whisper transcripts, so ASR errors dominate; segments cut mid-sentence also give the translator half sentences.
- **Checklist — first separate MT errors from ASR errors:**
  - [ ] Translate a few correct, typed Armenian sentences (conversational, not FLEURS) with the current model: if these are wrong too, the translator is at fault, not ASR.
  - [ ] Compare live English lines against the Armenian line shown above them: wrong Armenian → item 4; right Armenian, wrong English → this item.
- **Checklist — fixes to try:**
  - [ ] Bigger or different model: NLLB-1.3B was ~3 chrF better than 600M; also MADLAD-400 and OPUS-MT (all converted in `models/mt/`). Run the MT benchmark (item 1) and check the 0.5 s budget.
  - [ ] Repetition guard in `translate_batch` (e.g. `no_repeat_ngram_size`): NLLB looped 3 times in 400 outputs ("the grapes, the grapes, …"), M2M100 20 times.
  - [ ] Normalise «» quotes before translation: NLLB turns them into "⁇".
  - [ ] Translate whole sentences: avoid cutting phrases mid-sentence (item 4), or join a cut phrase with its continuation before translating.
  - [ ] Record the chosen model in `DECISIONS.md` and the README.
