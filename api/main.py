import json
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import AppendicitisPredictor

engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("\n--- Booting up CDSS Server ---")

    # This automatically handles checking, copying, and loading!
    engine = AppendicitisPredictor(
        pipeline_dir="../pipeline/Stage6",
        local_dir="./models"
    )

    yield  # Server is now LIVE and listening for requests

    print("🛑 Shutting down API and clearing memory...")
    engine = None


app = FastAPI(title="Pediatric Appendicitis CDSS", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_appendicitis(tabular_data: str = Form(...), image: UploadFile = File(...)):
    try:
        patient_features = json.loads(tabular_data)
        image_bytes = await image.read()
        report = engine.predict(patient_features, image_bytes)
        return {"status": "success", "patient_report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)