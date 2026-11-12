"""Test a detector on your own folders of real and fake recordings, and (optionally) calibrate it.
  python evaluate_folder.py --real data/real --fake data/fake
  python evaluate_folder.py --real data/real --fake data/fake --mp3 64
  python evaluate_folder.py --real data/real --fake data/fake --save-calibration
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import audio_utils as A
import config as C
from evaluate import compute_eer
from scoring import label_from_score

EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def list_audio(folder):
    return sorted(p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if p.lower().endswith(EXT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True, help="folder of real recordings")
    p.add_argument("--fake", required=True, help="folder of AI-generated recordings")
    p.add_argument("--detector", default="aasist", choices=["aasist", "cnn"])
    p.add_argument("--checkpoint", default=C.CHECKPOINT)
    p.add_argument("--mp3", type=int, default=0, help="compress every file to MP3 at this kbps first")
    p.add_argument("--save-calibration", action="store_true", help="save the equal-error threshold (AASIST only)")
    args = p.parse_args()

    if args.detector == "aasist":
        from aasist_detector import AasistDetector, CALIBRATION_FILE
        det = AasistDetector()
    else:
        from detector import Detector
        det = Detector(args.checkpoint)

    files = [(f, 0) for f in list_audio(args.real)] + [(f, 1) for f in list_audio(args.fake)]
    if not files:
        raise SystemExit("No audio files found.")
    labels, scores, margins, names = [], [], [], []
    for path, label in files:
        y, error = A.preprocess(path, normalize=det.normalize_input)
        if error:
            print("skipped", path, "-", error)
            continue
        if args.mp3:
            y = A.mp3_roundtrip(y, C.SAMPLE_RATE, args.mp3)
        score, _ = det.score_audio(y)
        labels.append(label); scores.append(score); names.append(os.path.basename(path))
        if args.detector == "aasist":
            margins.append(float(det.margins(y).mean()))
    labels, scores = np.array(labels), np.array(scores)

    eer, thr = compute_eer(labels, scores)
    pred = np.array([1 if label_from_score(s) == "synthetic" else 0 for s in scores])
    uncertain = int(sum(label_from_score(s) == "uncertain" for s in scores))
    prec, rec, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    out = {"detector": args.detector, "mp3_kbps": args.mp3, "n_real": int((labels == 0).sum()), "n_fake": int((labels == 1).sum()),
           "eer": eer, "accuracy": float((pred == labels).mean()), "precision": float(prec), "recall": float(rec), "f1": float(f1),
           "uncertain_files": uncertain, "confusion_matrix": confusion_matrix(labels, pred).tolist(),
           "mistakes": [n for n, l, pr in zip(names, labels, pred) if l != pr]}
    print(json.dumps(out, indent=2))
    with open(f"results_folder_{args.detector}{'_mp3_' + str(args.mp3) if args.mp3 else ''}.json", "w") as f:
        json.dump(out, f, indent=2)

    if args.save_calibration and args.detector == "aasist":
        m = np.array(margins)                         # threshold on the margin that gives equal error
        fpr_thr = compute_eer(labels, -m)[1]
        with open(CALIBRATION_FILE, "w") as f:
            json.dump({"offset": float(-fpr_thr), "temperature": 2.0}, f, indent=2)
        print("Saved calibration to", CALIBRATION_FILE, "- run the evaluation again to see the effect.")


if __name__ == "__main__":
    main()
