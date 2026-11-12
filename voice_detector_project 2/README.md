# Deepfake Voice Detection

A local speech analysis application for the Ahlia University graduation project. Upload a WAV or MP3 file in the Flask interface, or use the CLI with WAV, MP3 or FLAC. The interface includes waveform and spectrogram views, band-filtered playback, optional sound cues, and an exportable summary.

## Quick start

From this directory, with Python 3.11–3.13:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. FFmpeg is used for MP3 conversion in the tests and compression experiments. The dependency list includes an FFmpeg binary fallback through imageio-ffmpeg.

```sh
python inference.py /path/to/recording.wav
python inference.py /path/to/recording.flac --detector traditional
python app.py --detector traditional
```

Each model gives a synthetic-speech score on a 0–1 scale. Scores below 0.4 are labeled likely real and scores above 0.6 are labeled likely synthetic. The interval between them, including both endpoints, is labeled uncertain. These are screening scores with provisional operating thresholds. Performance depends on the source, language, codec and generator. Verify disputed recordings with independent evidence.

## Models and preprocessing

| Detector | Input | Model |
| --- | --- | --- |
| `aasist` (default) | Mono 16 kHz raw-amplitude waveform, 64,600-sample windows with a 2-second hop | Official pretrained AASIST checkpoint from NAVER |
| `traditional` | Mono 16 kHz peak-normalized waveform, 4-second windows with a 2-second hop | Standardized MFCC features and class-weighted logistic regression |
| `cnn` (optional) | Mono 16 kHz peak-normalized log-mel spectrograms | Existing project CNN, which requires local training |

Both shipped detectors repeat short clips to fill a window and include a final window aligned to the end. Audio under one second, near-silent audio and invalid samples are rejected. AASIST averages the bona fide-minus-spoof logit margin across windows before applying the existing score transform. The published benchmark uses a different utterance scoring protocol.

The traditional detector computes 20 MFCCs from 25 ms Hann-windowed frames with a 10 ms hop and 40 mel filters. Mean and standard deviation summarize the coefficients and their first differences, producing 80 features per window. Training averages these vectors per utterance. Inference averages the linear margins and applies a sigmoid. Scaling and classifier weights are saved as JSON and loaded without deserializing a Python estimator.

The thesis design describes a locally trained ResNet CNN. This implementation uses AASIST as the default and adds a logistic-regression comparison. The existing Flask interface and optional CNN training path remain available.

## Reproduce training and evaluation

Download [ASVspoof 2019 Logical Access](https://doi.org/10.7488/ds/2555) from the University of Edinburgh, then extract the LA directory with the official train, dev and eval FLAC folders and countermeasure protocols.

```sh
python train_traditional.py --root /path/to/LA
python evaluate.py --root /path/to/LA --detector aasist --split eval
python train.py --root /path/to/LA --model resnet
python app.py --detector cnn
```

The traditional trainer fits the scaler and classifier on official training utterances only and checks that speakers and utterance IDs are disjoint across partitions. Development and evaluation metrics use the fitted model without tuning on either partition. The evaluation attacks differ from the training attacks. Training produces [model coefficients](artifacts/traditional.json) and [metrics and protocol hashes](artifacts/traditional.metrics.json). The reported EER is the mean of the two error rates at the sampled ROC point nearest their intersection. Other thresholded metrics use a binary cutoff of 0.5 and do not measure the application's uncertain outcome.

## Measured baseline results

| Official partition | Scored clips | Rejected clips | ROC AUC | Approx. EER | Balanced accuracy at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 24,719 | 125 | 0.9114 | 17.35% | 81.24% |
| Evaluation | 70,178 | 1,059 | 0.8665 | 21.84% | 78.57% |

The baseline was fitted on 25,287 accepted training utterances. Results describe accepted clips from this corpus. AASIST was separately smoke-tested on 52 accepted evaluation clips, including two per spoof attack and 26 bona fide clips. AASIST's full benchmark and the application's performance on recent generators remain unevaluated here.

## Tests

```sh
python tests/test_pipeline.py
python tests/test_traditional.py
python check_aasist.py
```

The tests cover resampling, segmentation, score direction, invalid uploads, playback, waveform data, spectrogram export, and score agreement between the CLI pipeline and Flask. The checkpoint smoke test uses a generated signal to verify execution.

## Attribution

- Jung et al. (2022), *AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks*. [Paper](https://arxiv.org/abs/2110.01200) · [NAVER upstream](https://github.com/clovaai/aasist/tree/a04c9863f63d44471dde8a6abcb3b082b07cd1d1) · [MIT license and notices](third_party/aasist/NOTICE). The vendored model code and checkpoint match upstream Git blobs. The checkpoint SHA-256 is `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0`.
- Yamagishi et al. (2019), *ASVspoof 2019 challenge database*. [University of Edinburgh DataShare](https://doi.org/10.7488/ds/2555) · [Dataset paper](https://doi.org/10.1016/j.csl.2020.101114). Training data consists of labeled human and spoofed speech from the LA partition. The repository contains coefficients and metrics. Download the audio separately under the dataset terms.
