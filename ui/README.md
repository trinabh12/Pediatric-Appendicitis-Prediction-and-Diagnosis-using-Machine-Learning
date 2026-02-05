# Pediatric Appendicitis Prediction Frontend

This is a Streamlit-based frontend for predicting pediatric appendicitis using machine learning.

## Features

- Interactive web interface for inputting patient information, symptoms, and laboratory values
- Ultrasound image upload and display
- Real-time prediction results with visual indicators
- Responsive design that works on different screen sizes

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   streamlit run app.py
   ```

3. The app will open in your default web browser automatically.

## Usage

1. Fill in the patient information, symptoms, and lab values in the left column.
2. Optionally upload an ultrasound image.
3. Click "Predict Diagnosis" to get the result.
4. View the prediction with risk score or confidence in the right column.

## Model Integration

Currently uses a dummy model for demonstration. To integrate your actual ML model:

1. Train and save your model in the pipeline stages.
2. Replace the `predict_appendicitis()` function in `app.py` with your model loading and prediction code.
3. Handle both tabular data and image inputs as needed.

## Technologies Used

- Streamlit for the web interface
- Pandas for data handling
- Pillow for image processing
