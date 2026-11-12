"""Run local deepfake voice inference on a WAV, MP3 or FLAC file."""
import argparse
import json
from pathlib import Path

import audio_utils as audio

DETECTORS = ("aasist", "traditional", "cnn")


def load_detector(name):
    if name == "aasist":
        from aasist_detector import AasistDetector
        return AasistDetector()
    if name == "traditional":
        from traditional_detector import TraditionalDetector
        return TraditionalDetector()
    if name == "cnn":
        from detector import Detector
        return Detector()
    raise ValueError(f"Unknown detector: {name}")


def analyze_file(path, detector):
    y, error = audio.preprocess(path, normalize=detector.normalize_input)
    if error:
        raise ValueError(error)
    return detector.analyze(y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("--detector", choices=DETECTORS, default="aasist")
    args = parser.parse_args()
    try:
        result = analyze_file(args.audio_file, load_detector(args.detector))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Analysis failed: {error}\n")
    print(json.dumps({"detector": args.detector, **result}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
