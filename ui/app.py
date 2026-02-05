import streamlit as st
from PIL import Image
import random

# Dummy ML model function (replace with actual model later)
def predict_appendicitis(symptoms, image=None):
    # For now, random prediction based on symptoms
    risk_score = random.uniform(0, 1)
    diagnosis = "Appendicitis" if risk_score > 0.5 else "No Appendicitis"
    confidence = round(risk_score * 100, 2) if diagnosis == "Appendicitis" else round((1 - risk_score) * 100, 2)
    return diagnosis, confidence, risk_score

# Streamlit app
st.set_page_config(page_title="Pediatric Appendicitis Prediction", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    :root {
        --bg: #f5f7fb;
        --card: #ffffff;
        --accent: #0b6efd;
        --accent-dark: #094db3;
        --text: #0f172a;
        --muted: #64748b;
        --border: #e2e8f0;
    }
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
    }
    .hero {
        background: linear-gradient(120deg, #0b6efd 0%, #0ea5e9 60%, #38bdf8 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
        margin-bottom: 18px;
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 30px;
        letter-spacing: 0.2px;
    }
    .hero p {
        margin: 0;
        opacity: 0.95;
        font-size: 15px;
    }
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 16px;
    }
    .card h3 {
        margin-top: 0;
        margin-bottom: 10px;
        color: var(--text);
        font-size: 18px;
    }
    .hint {
        color: var(--muted);
        font-size: 12px;
        margin-top: 4px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.2);
        font-size: 12px;
        margin-right: 8px;
    }
    .footer {
        color: var(--muted);
        font-size: 12px;
        text-align: center;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <span class="badge">Clinical Decision Support</span>
        <span class="badge">Pediatric Focus</span>
        <h1>Pediatric Appendicitis Prediction</h1>
        <p>Enter patient details and upload ultrasound images to receive a risk estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Patient Information</h3>", unsafe_allow_html=True)

    # Basic info
    age = st.number_input("Age (years)", min_value=0, max_value=18, value=10)
    gender = st.selectbox("Gender", ["", "Male", "Female"])

    st.markdown("<h3>Symptoms</h3>", unsafe_allow_html=True)
    col_symptoms1, col_symptoms2 = st.columns(2)

    with col_symptoms1:
        pain_duration = st.number_input("Pain Duration (hours)", min_value=0, value=0)
        migration_of_pain = st.selectbox("Migration of Pain", ["", "Yes", "No"])
        anorexia = st.selectbox("Anorexia", ["", "Yes", "No"])
        nausea = st.selectbox("Nausea", ["", "Yes", "No"])

    with col_symptoms2:
        vomiting = st.selectbox("Vomiting", ["", "Yes", "No"])
        right_lower_quadrant_pain = st.selectbox("Right Lower Quadrant Pain", ["", "Yes", "No"])
        fever = st.selectbox("Fever", ["", "Yes", "No"])
        rebound_tenderness = st.selectbox("Rebound Tenderness", ["", "Yes", "No"])

    st.markdown("<h3>Laboratory Values</h3>", unsafe_allow_html=True)
    col_lab1, col_lab2 = st.columns(2)

    with col_lab1:
        white_blood_cell_count = st.number_input(
            "White Blood Cell Count (x10^9/L)", min_value=0.0, value=0.0, step=0.1
        )
        neutrophil_percentage = st.number_input(
            "Neutrophil Percentage (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1
        )

    with col_lab2:
        c_reactive_protein = st.number_input("C-Reactive Protein (mg/L)", min_value=0.0, value=0.0, step=0.1)

    st.markdown("<h3>Ultrasound Images</h3>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload one or more ultrasound images",
        type=["png", "jpg", "jpeg", "gif"],
        accept_multiple_files=True,
    )
    st.markdown('<div class="hint">PNG, JPG, or GIF. Multiple files supported.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)

    if st.button("Predict Diagnosis", type="primary"):
        # Validate inputs
        if not gender:
            st.error("Please select gender.")
        elif not all(
            [
                migration_of_pain,
                anorexia,
                nausea,
                vomiting,
                right_lower_quadrant_pain,
                fever,
                rebound_tenderness,
            ]
        ):
            st.error("Please fill in all symptom fields.")
        else:
            # Prepare symptoms data
            symptoms = {
                "age": age,
                "gender": gender,
                "pain_duration": pain_duration,
                "migration_of_pain": migration_of_pain,
                "anorexia": anorexia,
                "nausea": nausea,
                "vomiting": vomiting,
                "right_lower_quadrant_pain": right_lower_quadrant_pain,
                "fever": fever,
                "rebound_tenderness": rebound_tenderness,
                "white_blood_cell_count": white_blood_cell_count,
                "neutrophil_percentage": neutrophil_percentage,
                "c_reactive_protein": c_reactive_protein,
            }

            # Handle image
            image = None
            if uploaded_files:
                image = Image.open(uploaded_files[0])

            # Make prediction
            diagnosis, confidence, risk_score = predict_appendicitis(symptoms, image)

            # Display result
            if diagnosis == "Appendicitis":
                st.error(f"Diagnosis: {diagnosis}")
                st.metric("Risk Score", f"{risk_score:.2f}")
            else:
                st.success(f"Diagnosis: {diagnosis}")
                st.metric("Confidence", f"{confidence}%")

            st.info(
                "Note: This is a preliminary prediction. Please consult with a medical professional for accurate diagnosis."
            )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>Uploaded Images</h3>", unsafe_allow_html=True)
        img_cols = st.columns(2)
        for idx, file in enumerate(uploaded_files):
            with img_cols[idx % 2]:
                image = Image.open(file)
                st.image(image, caption=file.name, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<div class="footer">Pediatric Appendicitis Prediction using Machine Learning</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">Disclaimer: This tool is for educational purposes only and should not replace professional medical advice.</div>',
    unsafe_allow_html=True,
)
