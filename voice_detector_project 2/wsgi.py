from threadpoolctl import threadpool_limits

from app import create_app
from traditional_detector import TraditionalDetector


class HostedDetector(TraditionalDetector):
    def analyze(self, audio):
        with threadpool_limits(limits=1, user_api="blas"):
            return super().analyze(audio)


app = create_app(HostedDetector())
