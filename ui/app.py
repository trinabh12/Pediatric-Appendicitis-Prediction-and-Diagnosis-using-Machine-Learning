import streamlit as st
import pandas as pd
from PIL import Image
import random
import os

# Dummy ML model function (replace with actual model later)
def predict_appendicitis(symptoms, image=None):
    # For now, random prediction based on symptoms
    risk_score = random.uniform(0, 1)
    diagnosis = "Appendicitis" if risk_score > 0.5 else "No Appendicitis"
    confidence = round(risk_score * 100, 2) if diagnosis == "Appendicitis" else round((1 - risk_score) * 100, 2)
    return diagnosis, confidence, risk_score

# Streamlit app
st.set_page_config(page_title="Pediatric Appendicitis Prediction", page_icon="🏥", layout="wide")

st.title("🏥 Pediatric Appendicitis Prediction")
st.markdown("Enter patient information and upload ultrasound image for diagnosis prediction.")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Patient Information")

    # Basic info
    age = st.number_input("Age (years)", min_value=0, max_value=18, value=10)
    gender = st.selectbox("Gender", ["", "Male", "Female"])

    st.header("Symptoms")
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

    st.header("Laboratory Values")
    col_lab1, col_lab2 = st.columns(2)

    with col_lab1:
        white_blood_cell_count = st.number_input("White Blood Cell Count (x10^9/L)", min_value=0.0, value=0.0, step=0.1)
        neutrophil_percentage = st.number_input("Neutrophil Percentage (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

    with col_lab2:
        c_reactive_protein = st.number_input("C-Reactive Protein (mg/L)", min_value=0.0, value=0.0, step=0.1)

    st.header("Ultrasound Image")
    uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["png", "jpg", "jpeg", "gif"])

with col2:
    st.header("Prediction Result")

    if st.button("Predict Diagnosis", type="primary"):
        # Validate inputs
        if not gender:
            st.error("Please select gender.")
        elif not all([migration_of_pain, anorexia, nausea, vomiting, right_lower_quadrant_pain, fever, rebound_tenderness]):
            st.error("Please fill in all symptom fields.")
        else:
            # Prepare symptoms data
            symptoms = {
                'age': age,
                'gender': gender,
                'pain_duration': pain_duration,
                'migration_of_pain': migration_of_pain,
                'anorexia': anorexia,
                'nausea': nausea,
                'vomiting': vomiting,
                'right_lower_quadrant_pain': right_lower_quadrant_pain,
                'fever': fever,
                'rebound_tenderness': rebound_tenderness,
                'white_blood_cell_count': white_blood_cell_count,
                'neutrophil_percentage': neutrophil_percentage,
                'c_reactive_protein': c_reactive_protein
            }

            # Handle image
            image = None
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Ultrasound Image", use_column_width=True)

            # Make prediction
            diagnosis, confidence, risk_score = predict_appendicitis(symptoms, image)

            # Display result
            if diagnosis == "Appendicitis":
                st.error(f"⚠️ **Diagnosis: {diagnosis}**")
                st.metric("Risk Score", f"{risk_score:.2f}")
            else:
                st.success(f"✅ **Diagnosis: {diagnosis}**")
                st.metric("Confidence", f"{confidence}%")

            st.info("**Note:** This is a preliminary prediction. Please consult with a medical professional for accurate diagnosis.")

# Footer
st.markdown("---")
st.markdown("*Pediatric Appendicitis Prediction using Machine Learning*")
st.markdown("*Disclaimer: This tool is for educational purposes only and should not replace professional medical advice.*")