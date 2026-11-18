# Testing instructions for Mac Terminal

Run these commands in Terminal on your Mac. The requirement tests create their own audio samples and use the saved traditional detector. The ASVspoof dataset is optional for the separate accuracy evaluation.

## 1. Open your existing project and get the latest files

If you used the folder from the setup instructions, paste this block.

```bash
cd "$HOME/Downloads/deepfake-voice-github"
git pull --ff-only
cd "voice_detector_project 2"
pwd
ls tests/test_requirements.py requirements.txt
```

The last command should list both files. If your clone is elsewhere, change the first path to its actual location. Resolve any Git pull error before continuing.

### First download, if you have no clone

Use this block once instead of the existing-project block.

```bash
cd "$HOME/Downloads"
git clone https://github.com/yousifsarhan/Develop-a-machine-learning-model-for-Deepfake-Voice-detection.git deepfake-voice-github
cd "deepfake-voice-github/voice_detector_project 2"
```

## 2. Prepare Python

From the project folder, run this once. Python 3.11–3.13 is recommended on Mac.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -r requirements.txt
```

Wait for installation to finish successfully. A disk-space or missing-package error must be resolved before running tests.

For later Terminal sessions, return to the project folder and activate the environment.

```bash
cd "$HOME/Downloads/deepfake-voice-github/voice_detector_project 2"
source .venv/bin/activate
```

## 3. Run the 34 requirement checks

The test script starts its own Flask test client. You can run it directly from this folder.

```bash
python tests/test_requirements.py
```

A passing run ends with this line.

```text
34/34 checks passed
```

The script writes `results_requirements.json` into the current folder. Its exit code is 0 when all checks pass and 1 when any recorded check fails. An exception can stop the script before it writes the results.

The checks cover uploads, invalid input, analysis, repeatability, playback, waveform data, spectrograms, filters, sound cues, export, file removal, and timing.

Decoder warnings during the deliberately corrupt, empty, and truncated file checks are expected. Read the PASS/FAIL lines and final count to assess the outcome. Timing varies with your computer and its workload.

### Save the complete Terminal output

Use this block instead if you also want a log. The output appears in Terminal and is saved to a file.

```bash
set -o pipefail
python -u tests/test_requirements.py 2>&1 | tee requirements-terminal.log
```

Open the saved files in Finder.

```bash
open -R requirements-terminal.log
open -R results_requirements.json
```

Each run replaces these output files. Copy them elsewhere first if you want to preserve an earlier run.

## 4. Check that AASIST loads and runs

```bash
python -u check_aasist.py 2>&1 | tee aasist-terminal.log
```

A successful run ends with `OK: the model loads and runs` and an explanation about the generated test tone.

The printed time measures the script's calls to `margins()` and `analyze()` for a 10-second signal. It excludes model loading. The score for this generated signal does not measure speech detection accuracy.

## 5. Read the earlier recorded results

The checked-in evidence from 18 November 2026 is in this folder.

```bash
open "test-results/2026-11-18"
```

The summary explains the environment and which files were used. New test runs produce their own timings.

## Optional accuracy evaluation with the LA dataset

Download and extract ASVspoof 2019 LA before using these commands. Replace the dataset path if needed.

```bash
python evaluate.py --root "$HOME/Datasets/LA" --detector aasist --split eval --limit 3000
python evaluate.py --root "$HOME/Datasets/LA" --detector traditional --split eval --limit 3000
```

These commands evaluate labeled dataset recordings. The requirement checks and AASIST smoke check use generated signals to test software behavior. They do not establish accuracy on recent AI voices, hosted Render performance, or human usability.

## Common Terminal errors

| Message | Action |
| --- | --- |
| `command not found: python` | Activate `.venv` using Step 2. Use `python3` to create it. |
| `requirements.txt` or the script cannot be found | Return to the `voice_detector_project 2` folder using Step 1. |
| `ModuleNotFoundError` | Activate the environment and finish installing requirements. |
| `No space left on device` | Free disk space, then repeat the installation command. |
| Dataset path cannot be found | Extract LA and supply its actual path for accuracy evaluation. |
| A check prints `FAIL` | Keep the full log and JSON results for diagnosis. |
