# 🌤️ AQI Forecasting Pipeline

A fully automated **Air Quality Index (AQI) Forecasting System** that fetches live environmental data, trains a machine learning model, stores metrics, and serves predictions through a Flask web interface.

This project demonstrates a **complete MLOps-style workflow** — integrating data collection, preprocessing, model training, prediction, and deployment — all automated using **GitHub Actions**.

---

## 🚀 Features

✅ **Automated Data Fetching** — pulls daily air quality data and stores it in MongoDB.  
✅ **Model Training Pipeline** — automatically retrains and updates models at 10 PM (Pakistan time).  
✅ **Forecasting & Prediction API** — Flask app that provides AQI forecasts using the latest trained model.  
✅ **Metrics Dashboard** — displays model performance (MAE, RMSE, R², etc.) stored in `artifacts/model_metrics.json`.  
✅ **CI/CD Automation** — uses GitHub Actions to schedule daily data fetch and model training.

---

## 🧩 Project Structure

---

## ⚙️ Installation

### Clone the Repository

git clone https://github.com/yourusername/aqi-pipeline.git
cd aqi-pipeline

### Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # (Linux/macOS)
venv\Scripts\activate      # (Windows)

### Install Dependencies
pip install -r requirements.txt

### Set Environment Variables

Create a .env file in the root directory:

MONGO_URI=<your_mongo_connection_string>
DB_NAME=<your_database_name>
COLLECTION_NAME=<your_collection_name>

