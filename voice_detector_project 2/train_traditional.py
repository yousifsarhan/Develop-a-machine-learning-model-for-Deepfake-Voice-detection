"""Fit on the official ASVspoof 2019 LA train partition and report dev/eval metrics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import sklearn
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import audio_utils as audio
from traditional_detector import FEATURE_VERSION, MODEL_PATH, utterance_features

SPLITS = {"train": "trn", "dev": "trl", "eval": "trl"}


def read_protocol(root, split):
    path = root / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{split}.{SPLITS[split]}.txt"
    rows = []
    for line in path.read_text().splitlines():
        speaker, utterance, _, attack, label = line.split()
        if label not in {"bonafide", "spoof"}:
            raise ValueError(f"Unexpected protocol label: {label}")
        rows.append((speaker, utterance, attack, int(label == "spoof")))
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def extract(root, split, rows):
    folder = root / f"ASVspoof2019_LA_{split}" / "flac"
    vectors, labels = [], []
    skipped = 0
    for _, utterance, _, label in rows:
        y, rate = sf.read(folder / f"{utterance}.flac", dtype="float32", always_2d=True)
        y = audio.resample(y.mean(axis=1), rate)
        if audio.check_audio(y):
            skipped += 1
            continue
        vectors.append(utterance_features(y))
        labels.append(label)
    counts = np.bincount(labels, minlength=2).tolist()
    print(f"{split}: {counts[0]} bona fide, {counts[1]} spoof, {skipped} rejected", flush=True)
    return np.stack(vectors), np.asarray(labels), {
        "bonafide": counts[0], "spoof": counts[1], "rejected": skipped,
    }


def metrics(labels, scores):
    predicted = scores >= 0.5
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "eer_approx": float((fpr[i] + fnr[i]) / 2),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(labels, predicted)),
        "spoof_f1_at_0_5": float(f1_score(labels, predicted)),
        "confusion_matrix_at_0_5": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            f.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def train(root, output):
    protocols = {split: read_protocol(root, split) for split in SPLITS}
    seen_speakers, seen_ids = set(), set()
    for split, (rows, _) in protocols.items():
        speakers = {row[0] for row in rows}
        utterances = {row[1] for row in rows}
        if len(utterances) != len(rows) or speakers & seen_speakers or utterances & seen_ids:
            raise ValueError(f"Duplicate utterance or partition overlap at {split}")
        seen_speakers.update(speakers)
        seen_ids.update(utterances)

    with threadpool_limits(limits=1):
        x, y, counts = extract(root, "train", protocols["train"][0])
        estimator = make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", solver="lbfgs", max_iter=1000,
        ))
        estimator.fit(x, y)
        scaler, classifier = estimator.steps[0][1], estimator.steps[1][1]
        model = {
            "feature_version": FEATURE_VERSION,
            "dataset": "ASVspoof 2019 Logical Access",
            "scikit_learn_version": sklearn.__version__,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
        }
        report = {"train": counts}
        for split in ("dev", "eval"):
            x, y, counts = extract(root, split, protocols[split][0])
            report[split] = counts | metrics(y, estimator.predict_proba(x)[:, 1])

    for split, (rows, digest) in protocols.items():
        report[split]["protocol_sha256"] = digest
        report[split]["speakers"] = len({row[0] for row in rows})
        report[split]["spoof_attack_ids"] = sorted({row[2] for row in rows if row[3]})
    write_json(output, model)
    write_json(output.with_suffix(".metrics.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Extracted ASVspoof 2019 LA directory")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    train(args.root, args.output)
