"""Audio loading, preprocessing, features, filtering and cues."""
import io
import os
import shutil
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

import config as C


# ---------- loading and preprocessing ----------
def load_audio(path):
    y, sr = librosa.load(path, sr=None, mono=False)
    return y, sr


def to_mono(y):
    return y if y.ndim == 1 else y.mean(axis=0)


def resample(y, sr, target=C.SAMPLE_RATE):
    if sr == target:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=target)


def check_audio(y, sr=C.SAMPLE_RATE):
    """Return an error message, or None if the audio can be analyzed."""
    if len(y) / sr < C.MIN_DURATION:
        return "The recording is shorter than 1 second."
    if np.sqrt(np.mean(y ** 2)) < C.SILENCE_RMS:
        return "The recording is silent."
    return None


def peak_normalize(y):
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def preprocess(path, normalize=True):
    """File -> (mono 16 kHz audio, error message). Peak-normalized unless normalize=False."""
    y, sr = load_audio(path)
    y = resample(to_mono(y), sr)
    error = check_audio(y)
    if error:
        return None, error
    return (peak_normalize(y) if normalize else y).astype(np.float32), None


def segment(y):
    """Cut audio into 4 s segments with a 2 s hop. Short files are repeated."""
    seg = C.SEGMENT_SECONDS * C.SAMPLE_RATE
    hop = C.HOP_SECONDS * C.SAMPLE_RATE
    if len(y) < seg:
        y = np.tile(y, int(np.ceil(seg / len(y))))[:seg]
    starts = list(range(0, len(y) - seg + 1, hop))
    if starts[-1] + seg < len(y):          # make sure the end is covered
        starts.append(len(y) - seg)
    return np.stack([y[s:s + seg] for s in starts])


def log_mel(seg):
    """One 4 s segment -> normalized log-mel spectrogram (128 x 401)."""
    mel = librosa.feature.melspectrogram(
        y=seg, sr=C.SAMPLE_RATE, n_fft=C.N_FFT, win_length=C.WIN_LENGTH,
        hop_length=C.HOP_LENGTH, n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX, power=2.0)
    x = np.log(mel + 1e-6)
    return ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)


# ---------- augmentation (training only) ----------
def add_noise(y, snr_db, rng=np.random):
    signal_power = np.mean(y ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    return (y + rng.normal(0, np.sqrt(noise_power), len(y))).astype(np.float32)


def ffmpeg_exe():
    """FFmpeg from the PATH, or the copy that comes with the imageio-ffmpeg package."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def mp3_roundtrip(y, sr, bitrate_kbps):
    """Compress to MP3 with FFmpeg and decode again."""
    with tempfile.TemporaryDirectory() as d:
        wav_path, mp3_path = os.path.join(d, "a.wav"), os.path.join(d, "a.mp3")
        sf.write(wav_path, y, sr)
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", wav_path,
                        "-codec:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k", mp3_path], check=True)
        out, _ = librosa.load(mp3_path, sr=sr, mono=True)
    out = out[:len(y)]
    return np.pad(out, (0, len(y) - len(out))).astype(np.float32)


# ---------- sound design (user interaction) ----------
def band_filter(y, sr, low=None, high=None):
    """Butterworth band-pass (or low/high-pass) filter, zero phase."""
    nyq = sr / 2
    if low and high:
        sos = butter(C.FILTER_ORDER, [low / nyq, min(high, 0.99 * nyq) / nyq], btype="band", output="sos")
    elif high:
        sos = butter(C.FILTER_ORDER, min(high, 0.99 * nyq) / nyq, btype="low", output="sos")
    elif low:
        sos = butter(C.FILTER_ORDER, low / nyq, btype="high", output="sos")
    else:
        return y
    out = sosfiltfilt(sos, y)
    return peak_normalize(out).astype(np.float32) * 0.9   # make-up gain for quiet bands


def spectrogram_db(y, sr, n_fft=1024, hop=256):
    """Linear-frequency spectrogram in dB for the display (not the model input)."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    return librosa.amplitude_to_db(S, ref=np.max, top_db=80)


def waveform_peaks(y, n=900):
    """Min and max of each bin, for drawing the waveform in the browser."""
    bins = np.array_split(y, n)
    return [[float(b.min()), float(b.max())] for b in bins if len(b)]


CUE_NOTES = {"real": [523.25, 659.25], "synthetic": [659.25, 440.0], "uncertain": [392.0]}


def make_cue(label, sr=22050, note_len=0.2, fade=0.015):
    """Short sine-tone cue: rising = real, falling = synthetic, one note = uncertain."""
    parts = []
    for f in CUE_NOTES[label]:
        t = np.arange(int(note_len * sr)) / sr
        tone = 0.25 * np.sin(2 * np.pi * f * t)
        n = int(fade * sr)
        tone[:n] *= np.linspace(0, 1, n)
        tone[-n:] *= np.linspace(1, 0, n)
        parts.append(tone)
    return np.concatenate(parts).astype(np.float32), sr


def wav_bytes(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
