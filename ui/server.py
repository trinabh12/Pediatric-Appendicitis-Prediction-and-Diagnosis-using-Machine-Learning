from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import random  # For dummy prediction

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Dummy ML model function (replace with actual model later)
def predict_appendicitis(symptoms, image_path=None):
    # For now, random prediction based on symptoms
    risk_score = random.uniform(0, 1)
    diagnosis = "Appendicitis" if risk_score > 0.5 else "No Appendicitis"
    confidence = round(risk_score * 100, 2) if diagnosis == "Appendicitis" else round((1 - risk_score) * 100, 2)
    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "risk_score": risk_score
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    symptoms = {
        'age': request.form.get('age'),
        'gender': request.form.get('gender'),
        'pain_duration': request.form.get('pain_duration'),
        'migration_of_pain': request.form.get('migration_of_pain'),
        'anorexia': request.form.get('anorexia'),
        'nausea': request.form.get('nausea'),
        'vomiting': request.form.get('vomiting'),
        'right_lower_quadrant_pain': request.form.get('right_lower_quadrant_pain'),
        'fever': request.form.get('fever'),
        'rebound_tenderness': request.form.get('rebound_tenderness'),
        'white_blood_cell_count': request.form.get('white_blood_cell_count'),
        'neutrophil_percentage': request.form.get('neutrophil_percentage'),
        'c_reactive_protein': request.form.get('c_reactive_protein')
    }

    # Handle image upload
    image_path = None
    if 'ultrasound_image' in request.files:
        file = request.files['ultrasound_image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

    # Make prediction
    result = predict_appendicitis(symptoms, image_path)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
