"""Tests for the audio pipeline and the Flask app (a stub replaces the CNN, so PyTorch is not needed).
Run: python -m pytest tests   or   python tests/test_pipeline.py"""
import io
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audio_utils as A  # noqa: E402
import config as C  # noqa: E402
from app import create_app  # noqa: E402

RNG = np.random.default_rng(0)


def tone(freqs, seconds, sr):
    t = np.arange(int(seconds * sr)) / sr
    return sum(np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float32) / len(freqs)


def make_wav(path, seconds=5, sr=44100, stereo=True):
    y = tone([220, 440, 880], seconds, sr) + 0.05 * RNG.normal(size=int(seconds * sr)).astype(np.float32)
    sf.write(path, np.stack([y, 0.5 * y], axis=1) if stereo else y, sr)


def test_preprocess_mono_16k_peak():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.wav"); make_wav(p)
        y, err = A.preprocess(p)
    assert err is None and y.ndim == 1
    assert abs(len(y) - 5 * C.SAMPLE_RATE) < 5 and abs(np.max(np.abs(y)) - 1.0) < 1e-5


def test_check_audio():
    assert A.check_audio(np.zeros(100, dtype=np.float32)) is not None
    assert A.check_audio(np.zeros(C.SAMPLE_RATE * 3, dtype=np.float32)) == "The recording is silent."
    assert A.check_audio(tone([300], 2, C.SAMPLE_RATE)) is None


def test_segment_counts_and_shapes():
    seg = C.SEGMENT_SECONDS * C.SAMPLE_RATE
    assert A.segment(np.ones(3 * C.SAMPLE_RATE, dtype=np.float32)).shape == (1, seg)     # short file repeated
    assert A.segment(np.ones(10 * C.SAMPLE_RATE, dtype=np.float32)).shape == (4, seg)    # 0, 2, 4, 6 s
    assert A.segment(np.ones(9 * C.SAMPLE_RATE, dtype=np.float32)).shape == (4, seg)     # end gets its own segment


def test_log_mel_shape_and_scaling():
    x = A.log_mel(tone([440], 4, C.SAMPLE_RATE))
    assert x.shape == (128, 401) and abs(x.mean()) < 1e-3 and abs(x.std() - 1) < 1e-2


def test_gain_is_removed_by_feature_normalization():
    y = tone([300, 900], 4, C.SAMPLE_RATE) + 0.01 * RNG.normal(size=4 * C.SAMPLE_RATE).astype(np.float32)
    a, b = A.log_mel(y), A.log_mel(0.3 * y)       # a gain change is almost cancelled: why gain augmentation was dropped
    assert np.abs(a - b).mean() < 0.02 and np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.999


def test_band_filter_high_keeps_6k_removes_500():
    y = tone([500, 6000], 2, C.SAMPLE_RATE)
    hi = A.band_filter(y, C.SAMPLE_RATE, 4000, None)
    spec = np.abs(np.fft.rfft(hi)); freqs = np.fft.rfftfreq(len(hi), 1 / C.SAMPLE_RATE)
    assert spec[np.argmin(abs(freqs - 6000))] > 20 * spec[np.argmin(abs(freqs - 500))]


def test_noise_snr():
    y = tone([440], 2, C.SAMPLE_RATE)
    n = A.add_noise(y, 20) - y
    snr = 10 * np.log10(np.mean(y ** 2) / np.mean(n ** 2))
    assert abs(snr - 20) < 1.0


def test_mp3_roundtrip_length():
    y = tone([300, 1200], 3, C.SAMPLE_RATE)
    out = A.mp3_roundtrip(y, C.SAMPLE_RATE, 64)
    assert len(out) == len(y) and not np.allclose(out, y)


def test_cues():
    for label in A.CUE_NOTES:
        y, sr = A.make_cue(label)
        assert len(y) > 0 and np.max(np.abs(y)) <= 0.26


# ---------------- Flask app with a stub detector ----------------
class StubDetector:
    def analyze(self, y):
        return {"score": 0.87, "label": "synthetic", "segment_scores": [0.87], "n_segments": 1}


def client():
    return create_app(StubDetector()).test_client()


def upload(c, name, data):
    return c.post("/api/upload", data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def wav_bytes(seconds=5, sr=44100):
    buf = io.BytesIO(); y = tone([220, 440], seconds, sr)
    sf.write(buf, y, sr, format="WAV"); return buf.getvalue()


def test_upload_rejects_bad_files():
    c = client()
    assert upload(c, "notes.txt", b"hello").status_code == 400
    assert upload(c, "fake.wav", b"this is not audio").status_code == 400
    assert upload(c, "short.wav", wav_bytes(0.4)).status_code == 400
    silent = io.BytesIO(); sf.write(silent, np.zeros(44100 * 3, dtype=np.float32), 44100, format="WAV")
    r = upload(c, "silent.wav", silent.getvalue())
    assert r.status_code == 400 and "silent" in r.get_json()["error"]


def test_full_flow_wav_and_mp3():
    c = client()
    for name, data in [("clip.wav", wav_bytes()), ("clip.mp3", mp3_bytes())]:
        r = upload(c, name, data); assert r.status_code == 200, r.get_json()
        fid = r.get_json()["id"]
        r = c.post(f"/api/analyze/{fid}"); j = r.get_json()
        assert r.status_code == 200 and j["label"] == "synthetic" and len(j["peaks"]) > 100
        for band in ["off", "low", "mid", "high"]:
            a = c.get(f"/api/audio/{fid}?band={band}")
            assert a.status_code == 200 and a.data[:4] == b"RIFF"
        assert c.get(f"/api/audio/{fid}?band=custom&low=300&high=3400").status_code == 200
        assert c.get(f"/api/audio/{fid}?band=custom&low=3400&high=300").status_code == 400
        assert c.get(f"/api/spectrogram/{fid}.png").data[:4] == b"\x89PNG"
        assert c.get("/api/cue/synthetic.wav").data[:4] == b"RIFF"
        e = c.get(f"/api/export/{fid}"); assert e.status_code == 200 and e.data[:4] == b"\x89PNG"
        assert c.delete(f"/api/file/{fid}").status_code == 200
        assert c.post(f"/api/analyze/{fid}").status_code == 404      # temporary file is gone


def test_export_needs_analysis_and_index_loads():
    c = client()
    fid = upload(c, "a.wav", wav_bytes()).get_json()["id"]
    assert c.get(f"/api/export/{fid}").status_code == 400
    assert b"Voice Detection Application" in c.get("/").data


def mp3_bytes():
    with tempfile.TemporaryDirectory() as d:
        w, m = os.path.join(d, "a.wav"), os.path.join(d, "a.mp3")
        open(w, "wb").write(wav_bytes())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w, "-codec:a", "libmp3lame", "-b:a", "128k", m], check=True)
        return open(m, "rb").read()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f(); print("PASS", n)
        except Exception as e:
            failed += 1; print("FAIL", n, "->", repr(e))
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
