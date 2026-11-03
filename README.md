# Fake Voice Check

Tests whether detectors can tell AI-generated voices from real ones, on voices and voice generators they have never seen, including over a simulated phone line. Includes a demo that runs in your browser: play or upload a recording and see how likely it is to be AI-generated.

**Live demo:** https://eshwarpotturi.github.io/fakevoice/
**Results page:** https://eshwarpotturi.github.io/fakevoice/stats.html

Nothing you play or record is uploaded. The detector runs in your browser. It uses only the audio: no file names, tags or metadata.

## What we found

The first version was trained on one collection (54 public figures) and scored 94% on 12 held-out people from that collection. Tested on voice generators it had never seen, it did little better than guessing. We then retrained it on many more sources, holding out whole generators for the test.

**Held-out generators** (223 fake clips from 19 generators, 300 real clips, clean audio):

| | First version | Retrained (in the demo) |
|---|---|---|
| Fakes caught | 43% | 78% (83% over a simulated phone line) |
| Real clips wrongly flagged as AI | 34% | 7% (9% over a simulated phone line) |
| Score for telling fake from real (0.5 = guessing) | 0.59 | 0.94 |

**By dataset, clean audio** (fakes caught / real clips wrongly flagged):

| Held-out test | First version | Retrained |
|---|---|---|
| Original dataset, 12 people | 90% / 1% | 97% / 12% |
| MLAAD, 14 new generators | 46% / 71% | 80% / 12% |
| WaveFake, 2 new vocoders | 18% / 0% | 80% / 2% |
| ASVspoof 2021, 3 new attacks | 58% / 6% | 70% / 6% |

**Real speech wrongly flagged as AI:**

| Real speech | First version | Retrained |
|---|---|---|
| Indian-accented English, 46 clips, speakers never seen | 43% | 0% |
| Meeting headsets (AMI), 250 clips, a source never seen | 4% | 1% |
| Crowd recordings (FLEURS), 50 clips | 48% | 0% |
| Noisy audiobook recordings (LibriSpeech), 56 clips | 34% | 18% |

When the detector sounded sure a clip was human (under 5% AI), about 4 in 100 were actually fake, so the demo never shows less than 5% AI. When it sounded sure a clip was fake (over 95%), 99 in 100 were.

### First test (original dataset only)

Tested on 604 clips from 12 people held out of training (half real, half AI-generated). This is the first version.

| Detector | Right on clean audio | Right over a simulated phone line |
|---|---|---|
| WavLM, trained on clean and phone-quality audio (the first demo) | 94% | 94% |
| WavLM, trained on clean audio only | 94% | 85% |
| wav2vec 2.0, trained on clean audio only | 95% | 93% |
| Gemini Embedding 2, trained on clean audio only | 88% | 86% |
| Ready-made detector from the Hugging Face model hub, as downloaded | 45% | 61% |

Training on phone-quality copies of the clips lifted the phone-line result from 85% to 94%. A general-purpose Gemini embedding carries a real signal too, but a speech model scored higher. The ready-made detector did no better than guessing on clean audio.

## Try it on a clone of your own voice

The demo accepts any recording, so you can test a clone you made yourself. `clone_test/generate.py` makes new sentences from a 30-second recording of **your own voice** (or someone who agreed), using the open Chatterbox model on a normal computer. `score_files.py` then scores any audio files the same way the demo page does, on clean audio and over the simulated phone line.

```bash
python clone_test/generate.py --reference my_voice.wav --out clones/
python score_files.py my_voice.wav clones/clone_1.wav clones/clone_2.wav clones/clone_3.wav
```

A small test we ran (one Indian-accented speaker, three clones, an open-model clone, not the fakes in the datasets):

| Audio | Clean audio | Over a simulated phone line |
|---|---|---|
| Real recording | called human (5% AI) | called human (5% AI) |
| 3 clones | 2 caught (60%, 74%, 20% AI) | 1 caught (34%, 61%, 8% AI) |

The first version caught all 3 clones on clean audio (97%, 93% and 93% AI) and called the real recording human (10% AI). Three clips is far too few to give a rate. Our reading is that no fake Indian-accent voices were in training, so the retrained detector may treat the accent as a sign of a real voice. The next step is to train on accent-matched fakes. The demo page includes the owner's own recording and the three clones as samples, shared with their permission. They are the only voice recordings of a private person in this repository.

## What this does not show

- **Fakes in an Indian accent.** See the clone test above: 2 of 3 caught on clean audio, 1 of 3 over a phone line.
- **Small tests.** 223 fake clips from 19 held-out generators, 10 to 20 clips each (2 generators for WaveFake, 3 for ASVspoof). Indian-accent real speech: 46 clips.
- **Still missed.** Two Japanese vocoder sets and some modern generators (Llasa-8B, Cartesia Sonic-3, Kyutai, minimax, ASVspoof attacks A12 and A16) are caught on under 70% of clips.
- **More false alarms on the first dataset.** On the original 12 people, real clips wrongly flagged rose from 1% to 12%.
- **Recording differences.** Real and fake clips mostly come from different sources, so part of any score may reflect how a clip was recorded, not whether the voice is synthetic. On the first dataset, a trivial check using only loudness and tone balance scores 0.62 (0.5 is guessing).
- **Real calls are mostly genuine.** The test sets are about half fake. If only 1 call in 1,000 is fake, a 7% false-alarm rate wrongly flags about 70 genuine callers for each fake call.
- **Simulated phone line.** Real calls add dropouts, noise and other compression.
- **Languages.** English only. Other languages and real calls are untested.
- **A signal, not proof.** Use the percentage alongside other checks, such as a call-back.

## Data

5,760 six-second clips from these sources. Licences are as stated by each dataset. MLAAD is for non-commercial use (CC BY-NC 4.0), so check its terms before any commercial use of the retrained model. We made no fake audio.

| Source | Used for |
|---|---|
| [In-the-Wild](https://deepfake-total.com/in_the_wild) (Müller et al., 2022, Apache 2.0) | Real and fake clips of 54 public figures, generators unknown |
| [MLAAD-tiny](https://huggingface.co/datasets/mueller91/MLAAD-tiny) (CC BY-NC 4.0) | Fakes from 69 named speech generators; real audiobook clips |
| [WaveFake](https://zenodo.org/records/5642694) (CC BY-SA 4.0) | Fakes from 11 vocoder sets |
| [ASVspoof 2021 DF](https://zenodo.org/records/4835108) | Fakes from 13 named attacks (A07 to A19); real clips |
| LJSpeech | Real clips matching the WaveFake fakes |
| LibriSpeech, FLEURS, AMI | Real speech on ordinary microphones |
| Common Voice, Indian-accent subset | Real Indian-accented English speech |

The sample clips in `docs/data/samples` are 6-second excerpts from In-the-Wild.

> Müller, Czempin, Diekmann, Froghyar, Böttinger. *Does Audio Deepfake Detection Generalize?* Interspeech 2022.

## How it was built

First version:

1. `pull_clips.py`: reads a sample of up to 30 real and 30 fake 6-second clips per person straight from the In-the-Wild archive, split by person into train (34), check (8) and test (12).
2. `degrade.py`: makes phone-quality copies (300 to 3,400 Hz, 8 kHz, mu-law compression).
3. `extract.py`: listens to every clip with WavLM and wav2vec 2.0, and scores the ready-made detector. `gemini_embed.py`: embeds every clip with Gemini Embedding 2 through an LLM gateway (about 360,000 tokens; the key is read from a file and is not in this repository).
4. `train2.py`: trains a small classifier on top. Layer and settings are chosen on the check people only; the test people are used once, for the final numbers.
5. `calibrate.py`: turns the score into a percentage using held-out folds of the training people.
6. `export_onnx.py`, `shrink_onnx.py`: export the chosen detector as one browser file (weights stored at half precision).
7. `fingerprint.py`: tests whether embeddings alone can name a speaker, with no training. `finalize.py`: writes the data for the two web pages. `shortcut.py` runs the recording-differences check. `make_samples.py` picks the demo clips at random.

Retrained version (in the demo), in `retrain/`. Run the scripts from the repository root. `list_archives.py` first lists the remote archives and fetches the ASVspoof labels; `pull_clips.py` writes `work/manifest.json`, which is copied to `work/itw_manifest.json`.

1. `build_pool.py`, `build_mic.py`, `build_india.py`: download the extra fakes and the real speech, cut each clip to 6 seconds, and record its dataset, generator and speaker or book.
2. `feats.py`: listens to every clip, clean and phone-quality, with WavLM and keeps the average of every layer.
3. `train_eval3.py`: splits by whole generator (about 20% test, 10% check, the rest train) and by speaker or book for real clips, with the AMI meeting set held out entirely. It compares 8 layers, 3 regularisation strengths and 7 training lengths on the check split only, then scores the test split once. The chosen model uses layer 5.
4. `old_scores.py`, `old_mic.py`, `final_eval3.py`: score the first version on the same test clips and write the comparison tables in `retrain/results/`.
5. `export_onnx.py`, `shrink_onnx.py`: export the retrained detector. The first version is kept in `docs/model/previous/` for rollback.

Run `./run_all.sh` to rebuild the first version's results from the extracted features.

## Run locally

```bash
cd docs
python3 -m http.server 8000
# open http://localhost:8000
```

## Licences

Code: MIT. Datasets: see the table above. Models: WavLM (MIT), wav2vec 2.0 (Apache 2.0). Browser runtime: ONNX Runtime Web (MIT).
