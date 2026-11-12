import numpy as np

from aasist_detector import NB_SAMP
from app import create_app
from inference import load_detector


class HostedDetector:
    normalize_input = False

    def __init__(self):
        self.detector = None

    def analyze(self, audio):
        if self.detector is None:
            import torch

            torch.set_num_threads(1)
            self.detector = load_detector("aasist")
        # Bound CPU work on the hosted service while sampling the entire timeline.
        window_count = 4
        if len(audio) <= NB_SAMP:
            return self.detector.analyze(audio)

        from scoring import label_from_score, margin_to_score

        starts = np.linspace(0, len(audio) - NB_SAMP, window_count, dtype=int)
        starts = np.unique(starts)
        margins = [float(self.detector.margins(audio[start:start + NB_SAMP])[0]) for start in starts]
        calibration = self.detector.calibration
        score = margin_to_score(float(np.mean(margins)), calibration["offset"], calibration["temperature"])
        return {
            "score": score,
            "label": label_from_score(score),
            "segment_scores": [
                margin_to_score(value, calibration["offset"], calibration["temperature"])
                for value in margins
            ],
            "n_segments": len(margins),
            "sampled": True,
        }


app = create_app(HostedDetector())
