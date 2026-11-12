"""Turning model outputs into a score between 0 (real) and 1 (synthetic), and into a label."""
import math

import config as C


def label_from_score(score):
    if score < C.REAL_BELOW:
        return "real"
    if score > C.SYNTHETIC_ABOVE:
        return "synthetic"
    return "uncertain"


def margin_to_score(margin, offset=0.0, temperature=2.0):
    """margin = bonafide logit - spoof logit. A larger margin means more real, so the score goes down.
    offset is the margin at the equal error rate (found by calibrate/evaluate_folder), temperature sets the softness."""
    z = (margin - offset) / temperature
    z = max(min(z, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(z))
