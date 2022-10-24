import joblib
from fastapi import FastAPI
from pydantic import BaseModel

class Review(BaseModel):
    text: str

model = joblib.load('model.joblib')

app = FastAPI(title="Fake Review Detector")

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "OK"}

@app.post("/classify_review")
async def classify_review(review: Review):
    prediction = model.predict([review.text])
    return {"prediction": int(prediction[0])}