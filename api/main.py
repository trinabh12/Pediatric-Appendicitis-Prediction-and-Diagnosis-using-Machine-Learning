import json
import uvicorn
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # <--- This was likely missing
from fastapi.responses import RedirectResponse
from inference import AppendicitisPredictor

# Global variable for our inference engine
engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the models into RAM exactly once when the server boots up."""
    global engine
    print("\n--- Booting up CDSS Server ---")

    # We point to a LOCAL folder inside the 'api/' directory
    # The inference engine will automatically copy the required files here from the pipeline
    try:
        engine = AppendicitisPredictor(
            pipeline_dir="../pipeline/Stage6",  # Where to find the source files
            local_dir="./models"  # Where to put them locally for the API
        )
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        # We don't raise here so the server can still start (and show error on /health)

    yield  # The server runs while paused here

    # Clean up when the server stops
    print("🛑 Shutting down API and clearing memory...")
    engine = None


# Initialize the API using the lifespan manager
app = FastAPI(
    title="Pediatric Appendicitis CDSS API",
    description="Multi-Modal AI for Appendicitis Diagnosis and Ultrasound Finding Prediction",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware (Allows your frontend UI to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# 🩺 API Endpoints
# ------------------------------------------------------------------

@app.post("/predict")
async def predict_appendicitis(
        tabular_data: str = Form(..., description="JSON string containing the patient's tabular features"),
        image: UploadFile = File(..., description="Ultrasound BMP image file")
):
    """
    Receives patient vitals/labs and an ultrasound image, returns the CDSS report.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="AI Engine is not ready.")

    try:
        # 1. Parse the incoming JSON string into a Python dictionary
        patient_features = json.loads(tabular_data)

        # 2. Read the image file directly into memory as bytes
        image_bytes = await image.read()

        # 3. Pass to our Inference Engine
        report = engine.predict(patient_features, image_bytes)

        return {
            "status": "success",
            "patient_report": report
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="tabular_data must be a valid JSON string.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/health")
async def health_check():
    """Simple endpoint to verify the API is running."""
    status = "online" if engine else "offline"
    return {"status": status, "message": "Appendicitis API is ready."}


@app.get("/info")
async def get_model_info():
    """Returns the current active model version and metrics."""
    if engine:
        # If your inference class has a metadata method, use it.
        # If not, we return a basic status.
        return {
            "status": "active",
            "model_type": "Multi-Modal Fusion Network"
        }
    else:
        raise HTTPException(status_code=503, detail="Engine not loaded")


# ------------------------------------------------------------------
# 🖥️ Frontend Serving (Must be the last route!)
# ------------------------------------------------------------------

# Mount the 'ui' folder to serve index.html at the root URL
# This assumes your folder structure is:
# project/
#   ├── api/
#   │   └── main.py
#   └── ui/
#       └── index.html
if os.path.exists("../ui"):
    app.mount("/", StaticFiles(directory="../ui", html=True), name="ui")
else:
    print("⚠️ UI folder not found at '../ui'. Frontend will not be served.")

# Auto-run the server when executing `python main.py`
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)