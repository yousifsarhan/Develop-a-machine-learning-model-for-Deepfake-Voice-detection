"""All settings in one place (NFR06)."""
SAMPLE_RATE = 16000          # model works at 16 kHz
SEGMENT_SECONDS = 4          # length of one model input
HOP_SECONDS = 2              # hop between segments
N_FFT = 512
WIN_LENGTH = 400             # 25 ms at 16 kHz
HOP_LENGTH = 160             # 10 ms at 16 kHz
N_MELS = 128
FMIN, FMAX = 0, 8000

MIN_DURATION = 1.0           # seconds
SILENCE_RMS = 1e-4
REAL_BELOW = 0.4             # score < 0.4  -> likely real
SYNTHETIC_ABOVE = 0.6        # score > 0.6  -> likely synthetic

MAX_UPLOAD_MB = 50
ALLOWED_EXT = {".wav", ".mp3"}

FILTER_ORDER = 4
BANDS = {"low": (None, 1000), "mid": (1000, 4000), "high": (4000, None)}

CHECKPOINT = "checkpoints/resnet_best.pt"
