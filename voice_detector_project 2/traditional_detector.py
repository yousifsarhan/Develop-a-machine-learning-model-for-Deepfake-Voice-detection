"""MFCC summary features and a portable logistic-regression model."""
import json
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
from scipy.fft import dct, rfft
from scipy.special import expit

import audio_utils as audio
import config as C
from scoring import label_from_score

FEATURE_VERSION = "mfcc20-delta-v1"
FEATURE_COUNT = 80
MODEL_PATH = Path(__file__).with_name("artifacts") / "traditional.json"


@lru_cache(maxsize=1)
def mel_bank():
    return librosa.filters.mel(sr=C.SAMPLE_RATE, n_fft=512, n_mels=40)


def features(segment):
    """Summarize 20 MFCCs and their first differences with means and standard deviations."""
    frames = np.lib.stride_tricks.sliding_window_view(segment, 400)[::160]
    spectrum = rfft(frames * np.hanning(400).astype(np.float32), n=512, axis=1)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    mel = power @ mel_bank().T
    mfcc = dct(np.log(np.maximum(mel, 1e-10)), type=2, norm="ortho", axis=1)[:, :20]
    delta = np.diff(mfcc, axis=0)
    return np.concatenate([mfcc.mean(0), mfcc.std(0), delta.mean(0), delta.std(0)])


def utterance_features(y):
    """Pool segment features into one recording-level feature vector."""
    return np.mean([features(s) for s in audio.segment(audio.peak_normalize(y))], axis=0)


class TraditionalDetector:
    normalize_input = False

    def __init__(self, checkpoint=MODEL_PATH):
        with open(checkpoint, encoding="utf-8") as handle:
            model = json.load(handle)
        if model["feature_version"] != FEATURE_VERSION:
            raise ValueError("Traditional model feature version is incompatible.")
        self.mean, self.scale, self.coef = (
            np.asarray(model[key], dtype=np.float64) for key in ("mean", "scale", "coef")
        )
        self.intercept = float(model["intercept"])
        if (any(a.shape != (FEATURE_COUNT,) or not np.isfinite(a).all()
                for a in (self.mean, self.scale, self.coef))
                or np.any(self.scale <= 0) or not np.isfinite(self.intercept)):
            raise ValueError("Invalid traditional model parameters.")

    def score_audio(self, y):
        windows = np.stack([features(s) for s in audio.segment(audio.peak_normalize(y))])
        margins = ((windows - self.mean) / self.scale) @ self.coef + self.intercept
        return float(expit(margins.mean())), expit(margins).tolist()

    def analyze(self, y):
        score, scores = self.score_audio(y)
        return {"score": score, "label": label_from_score(score),
                "segment_scores": scores, "n_segments": len(scores)}
