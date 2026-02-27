import json
import uvicorn
import os
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from inference import AppendicitisPredictor

# Global variable for our inference engine
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the models into RAM exactly once when the server boots up."""
    global engine
    print("\n" + "="*40)
    print("🚀 BOOTING UP CDSS SERVER")
    print("="*40)

    try:
        # Initialize the predictor with pipeline and local paths
        engine = AppendicitisPredictor(
            pipeline_dir=os.path.join("../", "pipeline", "data"),
            local_dir="models"
        )
    except Exception as e:
        print(f"CRITICAL: Failed to initialize engine: {e}")
        traceback.print_exc()

    yield

    print("\nSHUTTING DOWN API...")
    engine = None

app = FastAPI(
    title="Pediatric Appendicitis CDSS API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/predict")
async def predict_appendicitis(
    tabular_data: str = Form(..., description="JSON string of patient features"),
    image: UploadFile = File(..., description="Ultrasound image file")
):
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Engine is offline or failed to load.")

    try:
        # 1. Parse JSON safely
        # .strip() handles hidden characters often pasted from text editors
        patient_features = json.loads(tabular_data.strip())

        # 2. Read image bytes
        image_bytes = await image.read()

        # 3. Run Inference
        report = engine.predict(patient_features, image_bytes)

        return {"status": "success", "patient_report": report}

    except json.JSONDecodeError as je:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid JSON format: {str(je)}"}
        )
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/health")
async def health_check():
    return {
        "status": "online" if engine else "offline",
        "engine_ready": engine is not None
    }



# Ensure UI directory exists before mounting
ui_path = os.path.abspath("../ui")
if os.path.exists(ui_path):
    # Mounting the static files at the root
    app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")
    print(f"UI mounted successfully from: {ui_path}")
else:
    print(f"UI folder NOT FOUND at {ui_path}. Dashboard will be unavailable.")

if __name__ == "__main__":
    # Note: Use 'main:app' as string for reliable hot-reloading
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)