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
API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(API_DIR, ".."))

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
            pipeline_dir="../pipeline/Stage6",
            local_dir="./models"
        )
    except Exception as e:
        print(f"❌ CRITICAL: Failed to initialize engine: {e}")
        traceback.print_exc()

    yield

    print("\n🛑 SHUTTING DOWN API...")
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

# ------------------------------------------------------------------
# 🩺 API Endpoints
# ------------------------------------------------------------------

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


@app.get("/feature-schema")
async def feature_schema():
    """
    Builds UI input schema from prepared dataset metadata.
    """
    candidate_paths = [
        os.path.join(PROJECT_ROOT, "pipeline", "Stage1", "ingestion_data", "tabular", "feature_info.json"),
        os.path.join(PROJECT_ROOT, "pipeline", "Stage3", "prepared dataset", "tabular", "feature_info.json"),
        os.path.join(PROJECT_ROOT, "pipeline", "Stage2", "validation_and_profiling_data", "tabular", "validated_feature_info.json"),
    ]

    info_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            info_path = path
            break

    if info_path is None:
        raise HTTPException(status_code=404, detail="Feature metadata file not found.")

    with open(info_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    excluded = {
        "Diagnosis", "Severity", "Diagnosis_Presumptive",
        "Management", "Length_of_Stay", "US_Number",
        "Abscess_Location", "Lymph_Nodes_Location", "Gynecological_Findings"
    }

    numeric_features = []
    binary_features = []
    categorical_features = []

    for feature, meta in feature_info.items():
        if feature in excluded:
            continue

        if meta in ("Continuous", "Discrete"):
            numeric_features.append({
                "key": feature,
                "label": feature.replace("_", " "),
                "step": "0.1"
            })
            continue

        if isinstance(meta, dict) and "Binary" in meta:
            labels = [str(x).strip().lower() for x in meta["Binary"]]
            if set(labels) == {"yes", "no"}:
                binary_features.append({
                    "key": feature,
                    "label": feature.replace("_", " "),
                    "encoding": "yes_no"
                })
            elif set(labels) == {"male", "female"}:
                binary_features.append({
                    "key": feature,
                    "label": feature.replace("_", " "),
                    "encoding": "map",
                    "options": [
                        {"label": "Male", "value": "male"},
                        {"label": "Female", "value": "female"}
                    ],
                    "map": {"male": 1, "female": 0}
                })
            elif set(labels) == {"appendicitis", "no appendicitis"}:
                binary_features.append({
                    "key": feature,
                    "label": feature.replace("_", " "),
                    "encoding": "map",
                    "options": [
                        {"label": "Appendicitis", "value": "appendicitis"},
                        {"label": "No Appendicitis", "value": "no appendicitis"}
                    ],
                    "map": {"appendicitis": 1, "no appendicitis": 0}
                })
            elif set(labels) == {"complicated", "uncomplicated"}:
                binary_features.append({
                    "key": feature,
                    "label": feature.replace("_", " "),
                    "encoding": "map",
                    "options": [
                        {"label": "Complicated", "value": "complicated"},
                        {"label": "Uncomplicated", "value": "uncomplicated"}
                    ],
                    "map": {"complicated": 1, "uncomplicated": 0}
                })
            else:
                binary_features.append({
                    "key": feature,
                    "label": feature.replace("_", " "),
                    "encoding": "map",
                    "options": [{"label": v.title(), "value": v} for v in labels],
                    "map": {v: i for i, v in enumerate(labels)}
                })
            continue

        if isinstance(meta, dict) and "Categorical" in meta:
            options = [str(x).strip().lower() for x in meta["Categorical"]]
            categorical_features.append({
                "key": feature,
                "label": feature.replace("_", " "),
                "options": options
            })

    return {
        "source": info_path,
        "numeric_features": numeric_features,
        "binary_features": binary_features,
        "categorical_features": categorical_features
    }

# ------------------------------------------------------------------
# 🖥️ Frontend Serving
# ------------------------------------------------------------------

# Ensure UI directory exists before mounting
ui_path = os.path.join(PROJECT_ROOT, "ui")
if os.path.exists(ui_path):
    # Mounting the static files at the root
    app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")
    print(f"✅ UI mounted successfully from: {ui_path}")
else:
    print(f"⚠️ UI folder NOT FOUND at {ui_path}. Dashboard will be unavailable.")

if __name__ == "__main__":
    # Note: Use 'main:app' as string for reliable hot-reloading
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
