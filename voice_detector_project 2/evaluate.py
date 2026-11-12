"""Evaluate a checkpoint: python evaluate.py --root /path/to/LA --checkpoint checkpoints/resnet_best.pt --split eval"""
import argparse
import json

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve


def compute_eer(labels, scores):
    """labels: 1 = fake, 0 = real. Returns (EER, threshold at EER)."""
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def main():
    import soundfile as sf
    import torch
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x: x
    import audio_utils as A
    import config as C
    from data import read_protocol

    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--detector", default="aasist", choices=["aasist", "cnn"])
    p.add_argument("--checkpoint", default=C.CHECKPOINT, help="only for --detector cnn")
    p.add_argument("--split", default="eval", choices=["dev", "eval"])
    p.add_argument("--mp3", type=int, default=0, help="compress every test file to MP3 at this kbps first")
    p.add_argument("--limit", type=int, default=0, help="only use the first N files (quick test)")
    args = p.parse_args()

    if args.detector == "aasist":
        from aasist_detector import AasistDetector
        det = AasistDetector()
    else:
        from detector import Detector
        det = Detector(args.checkpoint)
    items = read_protocol(args.root, args.split)
    if args.limit:
        items = items[:args.limit]

    labels, scores, attacks = [], [], []
    for path, label, attack in tqdm(items):
        y, sr = sf.read(path, dtype="float32")
        y = A.resample(y, sr)
        if det.normalize_input:
            y = A.peak_normalize(y)
        if args.mp3:
            y = A.mp3_roundtrip(y, C.SAMPLE_RATE, args.mp3)
        score, _ = det.score_audio(y)             # same code path as the application
        labels.append(label); scores.append(score); attacks.append(attack)
    labels, scores, attacks = np.array(labels), np.array(scores), np.array(attacks)

    eer, thr = compute_eer(labels, scores)
    pred = (scores >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    out = {"detector": args.detector, "split": args.split, "mp3_kbps": args.mp3, "n_files": int(len(labels)), "eer": eer, "eer_threshold": thr,
           "accuracy": float((pred == labels).mean()), "precision": float(prec), "recall": float(rec), "f1": float(f1),
           "confusion_matrix": confusion_matrix(labels, pred).tolist(), "per_attack_eer": {}}
    for a in sorted(set(attacks) - {"-"}):         # EER of each attack against all real files
        mask = (attacks == a) | (labels == 0)
        out["per_attack_eer"][a] = compute_eer(labels[mask], scores[mask])[0]
    print(json.dumps(out, indent=2))
    with open(f"results_{args.detector}_{args.split}{'_mp3_' + str(args.mp3) if args.mp3 else ''}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
