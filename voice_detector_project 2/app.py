"""Flask backend: python app.py  ->  http://localhost:5000"""
import atexit
import io
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display  # noqa: F401  (used for the spectrogram axes)
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

import audio_utils as A
import config as C


# Bound decoded audio independently of the compressed upload size.
MAX_MEDIA_SECONDS = 300
MEDIA_DEMUXERS = (
    "aac,aiff,amr,ape,asf,avi,caf,flac,flv,matroska,webm,mov,mp3,"
    "mpeg,mpegts,ogg,rm,tta,w64,wav,wv"
)


def extract_audio(source, destination):
    """Decode the first audio stream to a bounded, mono PCM WAV."""
    subprocess.run(
        [
            A.ffmpeg_exe(), "-nostdin", "-v", "error",
            "-protocol_whitelist", "file", "-format_whitelist", MEDIA_DEMUXERS,
            "-i", source, "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(C.SAMPLE_RATE), "-c:a", "pcm_f32le",
            "-t", str(MAX_MEDIA_SECONDS + 1), destination,
        ],
        check=True, timeout=30, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def create_app(detector):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = C.MAX_UPLOAD_MB * 1024 * 1024
    tmp_dir = tempfile.mkdtemp(prefix="voicedet_")          # temporary store D1
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    files = {}                                              # id -> info about one uploaded file

    def cleanup_old(max_age=3600):
        for fid in [k for k, v in files.items() if time.time() - v["time"] > max_age]:
            drop(fid)

    def drop(fid):
        info = files.pop(fid, None)
        if info and os.path.exists(info["path"]):
            os.remove(info["path"])

    def get_file(fid):
        info = files.get(fid)
        if info is None:
            return None
        return info

    @app.errorhandler(413)
    def too_big(_):
        return jsonify(error=f"The file is larger than {C.MAX_UPLOAD_MB} MB."), 413

    @app.route("/")
    def index():
        return render_template("index.html")

    # ---- UC01: upload and validate (FR01, FR08) ----
    @app.route("/api/upload", methods=["POST"])
    def upload():
        cleanup_old()
        f = request.files.get("file")
        if f is None or f.filename == "":
            return jsonify(error="No file was selected."), 400
        fid = secrets.token_hex(8)
        path = os.path.join(tmp_dir, fid + ".wav")
        try:
            with tempfile.TemporaryDirectory(dir=tmp_dir) as upload_dir:
                source = os.path.join(upload_dir, "upload")
                f.save(source)
                size = os.path.getsize(source)
                extract_audio(source, path)
            y, error = A.preprocess(path)
            if not error and len(y) > MAX_MEDIA_SECONDS * C.SAMPLE_RATE:
                error = "The audio track is longer than 5 minutes."
        except subprocess.TimeoutExpired:
            error = "Audio extraction timed out. Try a shorter recording."
        except (subprocess.CalledProcessError, OSError, ValueError):
            error = "No readable audio track was found. Try a recording with sound."
        if error:
            if os.path.exists(path):
                os.remove(path)
            return jsonify(error=error), 400
        files[fid] = {"path": path, "name": os.path.basename(f.filename), "time": time.time(),
                      "size": size, "duration": len(y) / C.SAMPLE_RATE, "result": None}
        return jsonify(id=fid, name=files[fid]["name"], duration=files[fid]["duration"], size=files[fid]["size"])

    # ---- UC02: classify (FR02, FR03) and data for UC04 ----
    @app.route("/api/analyze/<fid>", methods=["POST"])
    def analyze(fid):
        info = get_file(fid)
        if info is None:
            return jsonify(error="File not found. Please upload it again."), 404
        y, error = A.preprocess(info["path"], normalize=getattr(detector, "normalize_input", True))
        if error:
            return jsonify(error=error), 400
        try:
            result = detector.analyze(y)
        except Exception as e:
            return jsonify(error=f"The detection model failed: {e}"), 500
        result["peaks"] = A.waveform_peaks(A.peak_normalize(y))
        result["duration"] = info["duration"]
        info["result"] = result
        return jsonify(result)

    # ---- UC05: playback, original or filtered (FR06, FR09) ----
    @app.route("/api/audio/<fid>")
    def audio(fid):
        info = get_file(fid)
        if info is None:
            return jsonify(error="File not found."), 404
        y, _ = A.preprocess(info["path"])
        band = request.args.get("band", "off")
        if band in C.BANDS:
            low, high = C.BANDS[band]
            y = A.band_filter(y, C.SAMPLE_RATE, low, high)
        elif band == "custom":
            try:
                low, high = float(request.args["low"]), float(request.args["high"])
            except (KeyError, ValueError):
                return jsonify(error="Custom band needs low and high values."), 400
            if not 0 < low < high < C.SAMPLE_RATE / 2:
                return jsonify(error="Use 0 < low < high < 8000 Hz."), 400
            y = A.band_filter(y, C.SAMPLE_RATE, low, high)
        return send_file(io.BytesIO(A.wav_bytes(y, C.SAMPLE_RATE)), mimetype="audio/wav")

    # ---- UC04: spectrogram image (FR07) ----
    def spectrogram_png(fid, with_axes=True, figsize=(8, 3)):
        info = get_file(fid)
        y, _ = A.preprocess(info["path"])
        S = A.spectrogram_db(y, C.SAMPLE_RATE)
        fig, ax = plt.subplots(figsize=figsize, dpi=110)
        img = librosa.display.specshow(S, sr=C.SAMPLE_RATE, hop_length=256, x_axis="time", y_axis="hz", ax=ax, cmap="gray_r")
        ax.set_ylim(0, 8000)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(img, ax=ax, format="%+d dB")
        buf = io.BytesIO()
        fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
        return buf.getvalue()

    @app.route("/api/spectrogram/<fid>.png")
    def spectrogram(fid):
        if get_file(fid) is None:
            return jsonify(error="File not found."), 404
        return send_file(io.BytesIO(spectrogram_png(fid)), mimetype="image/png")

    # ---- UC03: sound cue (FR10) ----
    @app.route("/api/cue/<label>.wav")
    def cue(label):
        if label not in A.CUE_NOTES:
            return jsonify(error="Unknown cue."), 404
        y, sr = A.make_cue(label)
        return send_file(io.BytesIO(A.wav_bytes(y, sr)), mimetype="audio/wav")

    # ---- UC06: export (FR11) ----
    @app.route("/api/export/<fid>")
    def export(fid):
        info = get_file(fid)
        if info is None or info["result"] is None:
            return jsonify(error="Analyze the file first."), 400
        r = info["result"]
        fig = plt.figure(figsize=(8, 5.2), dpi=110)
        fig.text(0.05, 0.94, "Voice analysis summary", fontsize=15, weight="bold")
        lines = [f"File: {info['name']}", f"Duration: {info['duration']:.1f} s",
                 f"Result: likely {r['label']}" if r["label"] != "uncertain" else "Result: uncertain",
                 f"Score: {r['score']:.2f}  (0 = real, 1 = synthetic)",
                 f"Date: {datetime.now():%Y-%m-%d %H:%M}",
                 "This result is an indication, not a proof."]
        for i, t in enumerate(lines):
            fig.text(0.05, 0.86 - i * 0.055, t, fontsize=10)
        ax = fig.add_axes([0.1, 0.1, 0.82, 0.38])
        S = A.spectrogram_db(A.preprocess(info["path"])[0], C.SAMPLE_RATE)
        librosa.display.specshow(S, sr=C.SAMPLE_RATE, hop_length=256, x_axis="time", y_axis="hz", ax=ax, cmap="gray_r")
        ax.set_ylim(0, 8000)
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="analysis_summary.png")

    @app.route("/api/file/<fid>", methods=["DELETE"])
    def remove(fid):
        drop(fid)
        return jsonify(ok=True)

    return app


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    from inference import DETECTORS, load_detector

    ap.add_argument("--detector", default="aasist", choices=DETECTORS)
    args = ap.parse_args()
    create_app(load_detector(args.detector)).run(host="127.0.0.1", port=5000)
