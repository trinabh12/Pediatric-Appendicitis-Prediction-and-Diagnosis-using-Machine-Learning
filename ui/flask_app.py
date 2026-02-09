from __future__ import annotations

from flask import Flask, jsonify, render_template, request


app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    """
    Dummy endpoint for the existing frontend.
    Replace this logic with your real model inference.
    """
    # You can read form fields from `request.form` and files from `request.files`.
    # For now, return a fixed response the UI expects.
    return jsonify(
        {
            "diagnosis": "No Appendicitis",
            "confidence": 85,
            "risk_score": 0.15,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
