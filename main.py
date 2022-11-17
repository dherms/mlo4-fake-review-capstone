import onnxruntime as onxrt
from fastapi import FastAPI
from pydantic import BaseModel

class Review(BaseModel):
    text: str

rf_sess = onxrt.InferenceSession("rf_fake_review.onnx")
rf_input_name = rf_sess.get_inputs()[0].name
rf_label_name = rf_sess.get_outputs()[0].name

sgd_sess = onxrt.InferenceSession("sgd_fake_review.onnx")
sgd_input_name = sgd_sess.get_inputs()[0].name
sgd_label_name = sgd_sess.get_outputs()[0].name

app = FastAPI(title="Fake Review Detector")

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "OK"}

@app.post("/classify_review", tags=["Classify Review - General"])
async def classify_review(review: Review):
    pred_onx = sgd_sess.run([sgd_label_name], {sgd_input_name: [review.text]})[0]
    return {"prediction": int(pred_onx)}

@app.post("/classify_review_weed", tags=["Classify Review - Weed.com"])
async def classify_review(review: Review):
    pred_onx = rf_sess.run([rf_label_name], {rf_input_name: [review.text]})[0]
    return {"prediction": int(pred_onx)}
