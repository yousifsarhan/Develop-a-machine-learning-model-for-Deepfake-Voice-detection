"""Loads the trained model and analyzes one file (model layer)."""
import numpy as np
import torch

import audio_utils as A
import config as C
from model import build_model
from scoring import label_from_score


class Detector:
    normalize_input = True        # the CNN was trained on peak-normalized audio

    def __init__(self, checkpoint=C.CHECKPOINT, device=None):
        # CHANGED: uses the Apple GPU ("mps") when there is no NVIDIA GPU
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.model = build_model(ckpt["model_name"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()                      # fixed behaviour: same file, same result (NFR04)

    @torch.no_grad()
    def score_audio(self, y):
        """Preprocessed audio -> (final score, per-segment scores)."""
        feats = np.stack([A.log_mel(s) for s in A.segment(y)])
        x = torch.from_numpy(feats).unsqueeze(1).to(self.device)
        probs = torch.sigmoid(self.model(x)).cpu().numpy()
        return float(probs.mean()), probs.tolist()

    def analyze(self, y):
        score, seg_scores = self.score_audio(y)
        return {"score": score, "label": label_from_score(score),
                "segment_scores": seg_scores, "n_segments": len(seg_scores)}
