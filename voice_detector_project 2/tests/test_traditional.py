"""Regression tests for the fitted baseline, CLI pipeline and Flask integration."""
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app
from inference import analyze_file
from traditional_detector import FEATURE_COUNT, TraditionalDetector, utterance_features


class TraditionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = TraditionalDetector()
        t = np.arange(16000 * 5) / 16000
        cls.signal = (0.25 * np.sin(2 * np.pi * 221 * t)).astype(np.float32)

    def test_fitted_model_and_feature_contract(self):
        y = self.signal
        feature = utterance_features(y)
        self.assertEqual(feature.shape, (FEATURE_COUNT,))
        self.assertTrue(np.isfinite(feature).all())
        self.assertGreater(np.linalg.norm(self.detector.coef), 0)
        score, windows = self.detector.score_audio(y)
        self.assertEqual(len(windows), 2)
        self.assertTrue(0 <= score <= 1)
        expected = ((feature - self.detector.mean) / self.detector.scale
                    @ self.detector.coef + self.detector.intercept)
        self.assertAlmostEqual(score, 1 / (1 + np.exp(-expected)), places=5)

    def test_audio_entry_points_agree(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clip.wav"
            sf.write(path, self.signal, 16000)
            result = analyze_file(path, self.detector)
            client = create_app(self.detector).test_client()
            upload = client.post("/api/upload", data={"file": (io.BytesIO(path.read_bytes()), "clip.wav")})
            self.assertEqual(upload.status_code, 200)
            file_id = upload.get_json()["id"]
            response = client.post(f"/api/analyze/{file_id}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["score"], result["score"])
            self.assertEqual(response.get_json()["label"], result["label"])
            client.delete(f"/api/file/{file_id}")


if __name__ == "__main__":
    unittest.main()
