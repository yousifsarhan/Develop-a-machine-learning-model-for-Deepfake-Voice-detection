"""First check: does the pretrained AASIST model load and run?   python check_aasist.py"""
import time

import numpy as np

from aasist_detector import AasistDetector, segments_for_aasist

det = AasistDetector()
n_params = sum(p.numel() for p in det.model.parameters())
print(f"loaded AASIST: {n_params:,} parameters on {det.device}")
rng = np.random.default_rng(0)
t = np.arange(16000 * 10) / 16000
y = (0.1 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.normal(size=len(t))).astype(np.float32)   # 10 s test tone
print("segments:", segments_for_aasist(y).shape)
t0 = time.time()
print("margins:", det.margins(y))
print("analysis result:", det.analyze(y))
print(f"time for 10 s of audio: {time.time() - t0:.2f} s")
print("OK: the model loads and runs (a test tone is not speech, so the score itself means nothing).")
