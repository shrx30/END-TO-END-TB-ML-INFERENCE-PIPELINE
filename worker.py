from celery import Celery
from model import load_model
from inference import predict

celery_app = Celery(
    "worker",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)
model = load_model()

@celery_app.task
def run_inference(image_bytes):   # 🔥 NAME MUST MATCH
    return predict(model, image_bytes)