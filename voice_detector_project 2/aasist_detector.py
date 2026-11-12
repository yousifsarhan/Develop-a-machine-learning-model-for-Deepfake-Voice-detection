"""Pretrained AASIST detector (Jung et al., 2022), used unchanged from third_party/aasist.
Same interface as detector.Detector: score_audio(y) and analyze(y)."""
import importlib.util
import json
import os

import numpy as np

import config as C
from scoring import label_from_score, margin_to_score

HERE = os.path.dirname(os.path.abspath(__file__))
AASIST_DIR = os.path.join(HERE, "third_party", "aasist")
CALIBRATION_FILE = os.path.join(HERE, "calibration.json")
NB_SAMP = 64600                    # the model input: 64,600 samples = about 4.04 s at 16 kHz
HOP = 32000                        # 2 s hop between segments, as in the design chapter


def segments_for_aasist(y):
    """Cut audio into windows of 64,600 samples. Short audio is repeated, the end of a long file is covered."""
    if len(y) < NB_SAMP:
        y = np.tile(y, int(np.ceil(NB_SAMP / len(y))))[:NB_SAMP]
    starts = list(range(0, len(y) - NB_SAMP + 1, HOP))
    if starts[-1] + NB_SAMP < len(y):
        starts.append(len(y) - NB_SAMP)
    return np.stack([y[s:s + NB_SAMP] for s in starts]).astype(np.float32)


def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    return {"offset": 0.0, "temperature": 2.0}


class AasistDetector:
    normalize_input = False        # AASIST was trained on raw amplitudes, so no peak normalization

    def __init__(self, device=None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else ("mps" if torch.backends.mps.is_available() else "cpu"))
        spec = importlib.util.spec_from_file_location("AASIST", os.path.join(AASIST_DIR, "AASIST.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with open(os.path.join(AASIST_DIR, "aasist_config.json")) as f:
            cfg = json.load(f)
        self.model = module.Model(cfg).to(self.device)
        path = os.path.join(AASIST_DIR, "AASIST.pth")
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()                                   # fixed behaviour: same file, same result (NFR04)
        self.calibration = load_calibration()

    def margins(self, y):
        """Raw-amplitude 16 kHz mono audio -> bonafide-minus-spoof margin of each segment."""
        torch = self.torch
        windows = torch.from_numpy(segments_for_aasist(y))
        margins = []
        with torch.inference_mode():
            for batch in windows.split(1 if self.device == "cpu" else 4):
                _, logits = self.model(batch.to(self.device))
                margins.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
        return np.concatenate(margins)

    def score_audio(self, y):
        m = self.margins(y)
        score = margin_to_score(float(m.mean()), self.calibration["offset"], self.calibration["temperature"])
        return score, [margin_to_score(float(v), self.calibration["offset"], self.calibration["temperature"]) for v in m]

    def analyze(self, y):
        score, seg_scores = self.score_audio(y)
        return {"score": score, "label": label_from_score(score),
                "segment_scores": seg_scores, "n_segments": len(seg_scores)}
