# Develop-a-machine-learning-model-for-Deepfake-Voice-detection
Develop a machine learning model capable of analyzing audio recordings and distinguishing between real human voices and AI-generated voices. The model will use audio processing and feature extraction techniques to identify patterns associated with synthetic speech, providing users with a detection result and a confidence score.

Detects likely AI-generated speech in an uploaded WAV or MP3 file and lets the user explore the recording
(waveform, spectrogram, playback, band-filtered playback, result sound cues, export).

## Install
    pip install -r requirements.txt        # also needs FFmpeg on the PATH (MP3 files and MP3 augmentation)

## Train (ASVspoof 2019 LA)
Download ASVspoof 2019 and point --root at the LA folder (it contains ASVspoof2019_LA_train, _dev, _eval and ASVspoof2019_LA_cm_protocols).
    python train.py --root /path/to/LA --model resnet
    python train.py --root /path/to/LA --model baseline

## Evaluate
    python evaluate.py --root /path/to/LA --checkpoint checkpoints/resnet_best.pt --split eval
    python evaluate.py --root /path/to/LA --checkpoint checkpoints/resnet_best.pt --split eval --mp3 64     # compressed copies

## Run the application
    python app.py          # then open http://localhost:5000     (needs checkpoints/resnet_best.pt)

## Tests (no PyTorch needed; a stub replaces the model)
    python tests/test_pipeline.py

## Files
config.py (all settings) | audio_utils.py (preprocessing, features, filters, cues) | model.py (CNNs) | data.py (dataset)
train.py | evaluate.py | detector.py (model layer) | app.py (Flask backend) | templates/ and static/ (frontend)
