"""Requirement tests for Chapter 6: checks FR01-FR11 and NFR01, NFR03, NFR04, NFR05 on the running application.
Uses the real traditional detector (no PyTorch needed). Run:  python tests/test_requirements.py
Writes results_requirements.json and prints a table."""
import io
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audio_utils as A  # noqa: E402
import config as C  # noqa: E402
from app import create_app  # noqa: E402
from traditional_detector import TraditionalDetector  # noqa: E402

RESULTS = []
RNG = np.random.default_rng(7)
TMP = tempfile.mkdtemp(prefix="reqtest_")
FFMPEG = A.ffmpeg_exe()


def voice_like(seconds, sr=16000):
    """A synthetic voiced signal: harmonics with a moving pitch and syllable-like loudness."""
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = sum(np.sin(k * phase) / k for k in range(1, 10))
    y *= 0.5 * (1 + np.sin(2 * np.pi * 3 * t)) * 0.3
    return (y + 0.005 * RNG.normal(size=len(t))).astype(np.float32)


def write(name, seconds=6, fmt="wav", sr=16000):
    y = voice_like(seconds, sr)
    base = os.path.join(TMP, name)
    if fmt in ("wav", "flac", "ogg"):
        path = f"{base}.{fmt}"
        sf.write(path, y, sr)
        return path
    wav = base + "_src.wav"
    sf.write(wav, y, sr)
    path = f"{base}.{fmt}"
    if fmt == "mp3":
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", wav, "-codec:a", "libmp3lame", "-b:a", "128k", path]
    elif fmt == "m4a":
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", wav, "-codec:a", "aac", "-b:a", "128k", path]
    else:                                                   # mp4 and mov: a small video with this audio track
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=160x120:r=10",
               "-i", wav, "-shortest", "-c:v", "mpeg4", "-c:a", "aac", path]
    subprocess.run(cmd, check=True)
    return path


def client():
    return create_app(TraditionalDetector()).test_client()


def upload(c, path, name=None):
    with open(path, "rb") as f:
        return c.post("/api/upload", data={"file": (io.BytesIO(f.read()), name or os.path.basename(path))},
                      content_type="multipart/form-data")


def record(req, test, expected, passed, detail=""):
    RESULTS.append({"requirement": req, "test": test, "expected": expected, "passed": bool(passed), "detail": detail})
    print(("PASS " if passed else "FAIL ") + f"{req:6} {test}  {detail}")


def main():
    c = client()

    # FR01 + NFR05: formats
    fids = {}
    for fmt in ["wav", "mp3", "flac", "ogg", "m4a", "mp4", "mov"]:
        path = write(f"clip_{fmt}", 6, fmt)
        r = upload(c, path)
        j = r.get_json()
        ok = r.status_code == 200 and abs(j.get("duration", 0) - 6) < 0.5
        record("FR01", f"upload {fmt.upper()} file", "accepted, duration about 6 s", ok, f"status {r.status_code}, duration {j.get('duration', 0):.2f} s" if ok else str(j))
        if ok:
            fids[fmt] = j["id"]
    wavpath = write("clip_upper", 4, "wav")
    r = upload(c, wavpath, "CLIP.WAV")
    record("FR01", "upload file with capital letters in the extension", "accepted", r.status_code == 200)

    # FR08: validation
    bad = os.path.join(TMP, "notes.txt"); open(bad, "w").write("hello")
    r = upload(c, bad); record("FR08", "text file", "rejected with a message", r.status_code == 400 and "error" in r.get_json(), r.get_json().get("error", ""))
    fake = os.path.join(TMP, "fake.wav"); open(fake, "wb").write(os.urandom(40000))
    r = upload(c, fake); record("FR08", "random bytes named .wav", "rejected with a message", r.status_code == 400, r.get_json().get("error", ""))
    short = write("short", 0.5, "wav"); r = upload(c, short)
    record("FR08", "recording shorter than 1 s", "rejected", r.status_code == 400, r.get_json().get("error", ""))
    silent = os.path.join(TMP, "silent.wav"); sf.write(silent, np.zeros(16000 * 3, dtype=np.float32), 16000)
    r = upload(c, silent); record("FR08", "silent recording", "rejected", r.status_code == 400, r.get_json().get("error", ""))
    long_ = write("long", 310, "wav"); r = upload(c, long_)
    record("FR08", "recording longer than 5 minutes (310 s)", "rejected", r.status_code == 400, r.get_json().get("error", ""))
    huge = os.path.join(TMP, "huge.wav"); open(huge, "wb").write(os.urandom(51 * 1024 * 1024))
    r = upload(c, huge); record("FR08", "file larger than 50 MB", "rejected (413)", r.status_code == 413, r.get_json().get("error", ""))
    cut = write("cut", 6, "mp3"); data = open(cut, "rb").read()[:300]; open(cut, "wb").write(data)
    r = upload(c, cut); record("NFR04", "truncated MP3 file", "clear message, no crash", r.status_code == 400, r.get_json().get("error", ""))
    empty = os.path.join(TMP, "empty.wav"); open(empty, "wb").write(b"")
    r = upload(c, empty); record("NFR04", "empty file", "clear message, no crash", r.status_code in (400,), r.get_json().get("error", ""))

    # FR02, FR03, FR04: analysis
    fid = fids["wav"]
    r = c.post(f"/api/analyze/{fid}"); j = r.get_json()
    ok = r.status_code == 200 and j["label"] in ("real", "synthetic", "uncertain") and 0 <= j["score"] <= 1 and j["n_segments"] >= 1
    record("FR03", "classify a recording", "label and score between 0 and 1", ok, f"label {j.get('label')}, score {j.get('score', 0):.3f}, segments {j.get('n_segments')}")
    record("FR02", "preprocessing (mono, 16 kHz, segments)", "6 s file gives 1 segment of 4 s at a 2 s hop plus the end", j["n_segments"] == 2, f"{j['n_segments']} segments")
    record("FR04", "result with a confidence score", "score is returned", isinstance(j["score"], float))
    j2 = c.post(f"/api/analyze/{fid}").get_json()
    record("NFR04", "same file analyzed twice", "identical score", j["score"] == j2["score"], f"{j['score']:.6f} and {j2['score']:.6f}")
    results_by_fmt = {}
    for fmt, f in fids.items():
        results_by_fmt[fmt] = c.post(f"/api/analyze/{f}").get_json()["score"]
    spread = max(results_by_fmt.values()) - min(results_by_fmt.values())
    record("NFR05", "same content in 7 formats", "all analyzed", len(results_by_fmt) == 7, f"score spread {spread:.3f}")

    # FR05, FR06, FR07
    peaks = j["peaks"]
    record("FR05", "waveform data", "about 900 min/max pairs within -1 and 1", 800 <= len(peaks) <= 900 and max(abs(v) for p in peaks for v in p) <= 1.0001, f"{len(peaks)} pairs")
    a = c.get(f"/api/audio/{fid}?band=off"); y, sr = sf.read(io.BytesIO(a.data))
    record("FR06", "playback audio", "WAV of the same length", a.status_code == 200 and abs(len(y) / sr - 6) < 0.2, f"{len(y) / sr:.2f} s at {sr} Hz")
    s = c.get(f"/api/spectrogram/{fid}.png")
    record("FR07", "spectrogram image", "PNG image", s.status_code == 200 and s.data[:4] == b"\x89PNG" and len(s.data) > 5000, f"{len(s.data) // 1024} KB")

    # FR09: band filters on a two-tone signal
    two = os.path.join(TMP, "two.wav"); t = np.arange(16000 * 4) / 16000
    sf.write(two, (0.4 * np.sin(2 * np.pi * 500 * t) + 0.4 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32), 16000)
    tid = upload(c, two).get_json()["id"]

    def band_ratio(band):
        out, sr_ = sf.read(io.BytesIO(c.get(f"/api/audio/{tid}?band={band}").data))
        sp = np.abs(np.fft.rfft(out)); fr = np.fft.rfftfreq(len(out), 1 / sr_)
        return sp[np.argmin(abs(fr - 6000))] / sp[np.argmin(abs(fr - 500))]
    low, mid, high = band_ratio("low"), band_ratio("mid"), band_ratio("high")
    record("FR09", "low band keeps 500 Hz, removes 6 kHz", "6 kHz/500 Hz ratio below 0.05", low < 0.05, f"ratio {low:.4f}")
    record("FR09", "high band keeps 6 kHz, removes 500 Hz", "6 kHz/500 Hz ratio above 20", high > 20, f"ratio {high:.1f}")
    cu = c.get(f"/api/audio/{tid}?band=custom&low=300&high=3400"); bad_cu = c.get(f"/api/audio/{tid}?band=custom&low=3400&high=300")
    record("FR09", "custom range", "valid range accepted, invalid range refused", cu.status_code == 200 and bad_cu.status_code == 400)

    # FR10 cues
    cues = {k: sf.read(io.BytesIO(c.get(f"/api/cue/{k}.wav").data)) for k in ["real", "synthetic", "uncertain"]}
    dur = {k: len(v[0]) / v[1] for k, v in cues.items()}
    distinct = not np.array_equal(cues["real"][0], cues["synthetic"][0])
    record("FR10", "three sound cues", "short, different sounds", all(0.1 < d < 1.0 for d in dur.values()) and distinct, ", ".join(f"{k} {d:.2f} s" for k, d in dur.items()))

    # FR11 export
    before = upload(c, write("fresh", 5, "wav")).get_json()["id"]
    r0 = c.get(f"/api/export/{before}")
    e = c.get(f"/api/export/{fid}")
    record("FR11", "export summary", "PNG after analysis, refused before", e.status_code == 200 and e.data[:4] == b"\x89PNG" and r0.status_code == 400)

    # NFR03: temporary files and names
    ids = [upload(c, write("same", 3, "wav"), "my private name.wav").get_json()["id"] for _ in range(2)]
    record("NFR03", "file ids", "random, different, name not used", ids[0] != ids[1] and all(len(i) == 16 and "private" not in i for i in ids), f"{ids[0]}, {ids[1]}")
    d = c.delete(f"/api/file/{ids[0]}"); after = c.post(f"/api/analyze/{ids[0]}")
    record("NFR03", "remove a file", "file is gone afterwards", d.status_code == 200 and after.status_code == 404)
    record("NFR03", "server address", "only the local address in the local version", "127.0.0.1" in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")).read())

    # NFR01: analysis time for a 30 s recording
    big = write("t30", 30, "wav"); times = []; totals = []
    for _ in range(5):
        t0 = time.time(); fid30 = upload(c, big).get_json()["id"]; t1 = time.time()
        c.post(f"/api/analyze/{fid30}"); t2 = time.time()
        times.append(t2 - t1); totals.append(t2 - t0)
    med = statistics.median(times)
    record("NFR01", "analyze a 30 s recording (median of 5)", "10 s or less", med <= 10, f"analysis {med:.2f} s, upload and analysis {statistics.median(totals):.2f} s")
    t0 = time.time(); c.get(f"/api/spectrogram/{fid30}.png"); sp_t = time.time() - t0
    record("NFR01", "spectrogram image of a 30 s recording", "3 s or less", sp_t <= 3, f"{sp_t:.2f} s")

    passed = sum(r["passed"] for r in RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    with open("results_requirements.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
