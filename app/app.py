from flask import Flask, render_template, request
import pandas as pd
from src.predict import forecast_and_predict  # You’ll wrap your logic into a function
import os
import json
from flask import render_template
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Run forecast and prediction logic
        results = forecast_and_predict()  # returns a DataFrame

        # Group by day and compute average AQI
        avg_aqi = results.groupby("day")["predicted_aqi"].mean().reset_index()
        avg_aqi = avg_aqi.rename(columns={"day": "date", "predicted_aqi": "avg_aqi"})

        # Convert day numbers to full future dates
        today = datetime.now()
        avg_aqi["date"] = avg_aqi["date"].apply(lambda d: (today + timedelta(days=int(d))).strftime("%B %d, %Y"))

        # Convert to dictionary for Jinja2
        forecast_data = avg_aqi.to_dict(orient='records')

        # Current date for display
        current_date = today.strftime("%B %d, %Y")

        return render_template(
            'predict.html',
            current_date=current_date,
            forecast_data=forecast_data
        )

    except Exception as e:
        return render_template(
            'predict.html',
            error=str(e),
            current_date=datetime.now().strftime("%B %d, %Y"),
            forecast_data=[]
        )

@app.route('/metrics')
def metrics():
    # Path to your metrics file
    artifact_dir = "artifacts"
    metrics_path = os.path.join(artifact_dir, "model_metrics.json")

    # Check if file exists
    if not os.path.exists(metrics_path):
        return render_template("metrics.html", metrics=None, error="Metrics file not found.")

    # Load metrics from JSON
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    return render_template("metrics.html", metrics=metrics_data)


if __name__ == '__main__':
    app.run(debug=True)
