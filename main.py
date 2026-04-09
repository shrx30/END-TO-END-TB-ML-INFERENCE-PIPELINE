from fastapi import FastAPI, UploadFile
from worker import run_inference

app = FastAPI()

@app.post("/predict")
async def predict_api(file: UploadFile):
    image = await file.read()
    
    task = run_inference.delay(image)

    return {"task_id": task.id}
from celery.result import AsyncResult

from worker import celery_app
from celery.result import AsyncResult

@app.get("/result/{task_id}")
def get_result(task_id: str):
    result = AsyncResult(task_id, app=celery_app)  # ✅ FIX

    if result.ready():
        return {"result": result.result}

    return {"status": "processing"}