import onnxruntime as onxrt
from fastapi import FastAPI
from pydantic import BaseModel

class Review(BaseModel):
    text: str

sess = onxrt.InferenceSession("sgd_fake_review.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

app = FastAPI(title="Fake Review Detector")

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "OK"}

@app.post("/classify_review", tags=["Classify Review"])
async def classify_review(review: Review):
    pred_onx = sess.run([label_name], {input_name: [review.text]})[0]
    return {"prediction": int(pred_onx)}