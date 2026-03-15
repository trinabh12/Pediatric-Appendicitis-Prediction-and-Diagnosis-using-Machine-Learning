import json
import uvicorn
import os
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Ensure this matches your actual inference module
from inference import AppendicitisPredictor

engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("\n--- BOOTING UP CDSS SERVER ---")

    # Detects if running in Docker or Local
    current_file_dir = os.path.dirname(os.path.abspath(__file__))  # /app/api
    local_model_path = os.path.join(current_file_dir, "models")
    pipeline_src_path = os.path.abspath(os.path.join(current_file_dir, "../../pipeline/Stage6"))

    try:
        engine = AppendicitisPredictor(
            pipeline_dir=pipeline_src_path if os.path.exists(pipeline_src_path) else None,
            local_dir=local_model_path
        )
        print("Engine Initialized Successfully.")
    except Exception as e:
        print(f"Initialization Failed: {e}")
        traceback.print_exc()

    yield
    print("\n--- SHUTTING DOWN API ---")
    engine = None


app = FastAPI(title="Appendicitis CDSS API", lifespan=lifespan)

# Allow frontend to communicate with API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
#               API ENDPOINTS
# ==========================================

@app.post("/predict")
async def predict_appendicitis(
        tabular_data: str = Form(...),
        image: UploadFile = File(...)
):
    """
    Receives the JSON string of patient vitals and the ultrasound image,
    passes them to the multimodal fusion engine, and returns the diagnostic report.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine Offline")
    try:
        # The frontend JS packages the form data into a neat JSON string
        features = json.loads(tabular_data.strip())
        img_bytes = await image.read()

        # Pass to your Inference module
        report = engine.predict(features, img_bytes)

        return {"status": "success", "patient_report": report}
    except Exception as e:
        print(f"Prediction Error: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/health")
async def health():
    return {"status": "online" if engine else "offline"}


# ==========================================
#               UI MOUNTING
# ==========================================

current_dir = os.path.dirname(os.path.abspath(__file__))
potential_ui_paths = [
    os.path.abspath(os.path.join(current_dir, "../ui")),
    os.path.abspath(os.path.join(current_dir, "../../ui")),
    "/app/ui"  # Docker container path
]

ui_mounted = False
for path in potential_ui_paths:
    if os.path.exists(path):
        # Mount the UI folder at the root path "/"
        app.mount("/", StaticFiles(directory=path, html=True), name="ui")
        print(f"UI mounted successfully from: {path}")
        ui_mounted = True
        break

if not ui_mounted:
    print("UI folder not found. API only mode active. Check your directory structure if you expect a UI.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)