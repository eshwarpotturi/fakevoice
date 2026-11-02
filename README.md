# Fake Voice Check

Tests whether detectors can tell AI-generated voices from real ones, on people they have never heard, including over a simulated phone line. Includes a demo that runs in your browser: play or upload a recording and see how likely it is to be AI-generated.

**Live demo:** https://eshwarpotturi.github.io/fake-voice-check/
**Results page:** https://eshwarpotturi.github.io/fake-voice-check/stats.html

Nothing you play or record is uploaded. The detector runs in your browser.

## What we found

Tested on 604 clips from 12 people held out of training (half real, half AI-generated):

| Detector | Right on clean audio | Right over a simulated phone line |
|---|---|---|
| WavLM, trained on clean and phone-quality audio (used in the demo) | 94% | 94% |
| WavLM, trained on clean audio only | 94% | 85% |
| wav2vec 2.0, trained on clean audio only | 95% | 93% |
| Gemini Embedding 2, trained on clean audio only | 88% | 86% |
| Ready-made detector from the Hugging Face model hub, as downloaded | 45% | 61% |

Training on phone-quality copies of the clips lifted the phone-line result from 85% to 94%. A general-purpose Gemini embedding carries a real signal too, but a speech model scored higher. The ready-made detector did no better than guessing on clean audio.

The percentage in the demo is calibrated on people not used for the test. When the detector sounded sure a clip was human (under 5% AI), about 6 in 100 were actually fake, so the demo never shows less than 5% AI.

## What this does not show

- **Recording differences.** In this collection, real and fake clips come from different sources, so part of any score may reflect how a clip was recorded, not whether the voice is synthetic. A trivial check using only loudness and tone balance scores 0.62 (0.5 is guessing).
- **One collection.** Training and testing use the same kind of recordings. New voice generators and recording settings are untested.
- **Real calls are mostly genuine.** The test is half fake. If only 1 call in 1,000 is fake, a 5% false-alarm rate wrongly flags about 50 genuine callers for each fake caught.
- **Few people.** 12 test people with about 50 clips each.
- **Simulated phone line.** Real calls add dropouts, noise and other compression.
- **Voices and languages.** The people are mostly English-speaking public figures from the US and UK. Indian voices and other languages are not yet tested.
- **A signal, not proof.** Use the percentage alongside other checks, such as a call-back.

## Data

[In-the-Wild](https://deepfake-total.com/in_the_wild) (Müller et al., 2022, Apache 2.0): real and AI-faked recordings of 54 public figures, collected from the internet by the dataset's researchers. We made no fake audio. The sample clips in `docs/data/samples` are 6-second excerpts from it.

> Müller, Czempin, Diekmann, Froghyar, Böttinger. *Does Audio Deepfake Detection Generalize?* Interspeech 2022.

## How it was built

1. `pull_clips.py`: reads a sample of up to 30 real and 30 fake 6-second clips per person straight from the dataset archive, split by person into train (34), check (8) and test (12).
2. `degrade.py`: makes phone-quality copies (300 to 3,400 Hz, 8 kHz, mu-law compression).
3. `extract.py`: listens to every clip with WavLM and wav2vec 2.0, and scores the ready-made detector. `gemini_embed.py`: embeds every clip with Gemini Embedding 2 through an LLM gateway (about 360,000 tokens; the key is read from a file and is not in this repository).
4. `train2.py`: trains a small classifier on top. Layer and settings are chosen on the check people only; the test people are used once, for the final numbers.
5. `calibrate.py`: turns the score into a percentage using held-out folds of the training people.
6. `export_onnx.py`, `shrink_onnx.py`: export the chosen detector as one browser file (weights stored at half precision).
7. `finalize.py`: writes the data for the two web pages. `shortcut.py` runs the recording-differences check. `make_samples.py` picks the demo clips at random.

Run `./run_all.sh` to rebuild the results from the extracted features.

## Run locally

```bash
cd docs
python3 -m http.server 8000
# open http://localhost:8000
```

## Licences

Code: MIT. Dataset: Apache 2.0. Models: WavLM (MIT), wav2vec 2.0 (Apache 2.0). Browser runtime: ONNX Runtime Web (MIT).
