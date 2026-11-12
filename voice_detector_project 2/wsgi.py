from app import create_app
from inference import load_detector


class HostedDetector:
    normalize_input = False

    def __init__(self):
        self.detector = None

    def analyze(self, audio):
        if self.detector is None:
            import torch

            torch.set_num_threads(1)
            self.detector = load_detector("aasist")
        return self.detector.analyze(audio)


app = create_app(HostedDetector())
