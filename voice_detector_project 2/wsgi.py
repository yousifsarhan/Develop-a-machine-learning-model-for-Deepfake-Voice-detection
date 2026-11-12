from app import create_app
from inference import load_detector

app = create_app(load_detector("aasist"))
