import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import joblib
import os
from dotenv import load_dotenv
from src.mongo_functions import preprocess_features

# Load environment variables
load_dotenv()

def forecast_and_predict():
    
    latitude = 24.8608
    longitude = 67.0104

    # Use UTC safely
    today = datetime.now(timezone.utc).date()
    start_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (today + timedelta(days=3)).strftime("%Y-%m-%d")

    print(f"📅 Fetching forecast data for {start_date} → {end_date}...")

    try:
        # Build URLs
        # --- Build API URLs ---
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}&longitude={longitude}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )

        air_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={latitude}&longitude={longitude}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
        )


        # Fetch data
        wresp = requests.get(weather_url)
        aresp = requests.get(air_url)
        wresp.raise_for_status()
        aresp.raise_for_status()

        wdata = wresp.json()
        adata = aresp.json()

        # Convert to DataFrames
        wdf = pd.DataFrame(wdata["hourly"])
        adf = pd.DataFrame(adata["hourly"])

        # Merge
        df = pd.merge(wdf, adf, on="time", how="inner")

        # Convert timestamps
        df["time"] = pd.to_datetime(df["time"], utc=True)
        print(f"✅ Retrieved {len(df)} forecast records")

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching forecast data: {e}")
        df = pd.DataFrame()

    if df.empty:
        print("❌ No forecast data available. Exiting.")
        exit()

    # --- Handle missing 'us_aqi' for forecast data ---
    if "us_aqi" not in df.columns:
        df["us_aqi"] = None  # Add dummy target for compatibility

    # --- Preprocess Features ---
    processed = preprocess_features(df)
    print(processed.head())
    # --- Load Model ---
    artifact_dir = "artifacts"
    model_path = os.path.join(artifact_dir, "best_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

    model = joblib.load(model_path)

    # --- Predict AQI ---
    features = processed.drop(columns=["time","us_aqi"], errors="ignore")
    predictions = model.predict(features)

    print("🧩 Final feature columns used for prediction:", features.columns.tolist())

    # --- Combine results ---
    processed["predicted_aqi"] = predictions

    print("\n📈 AQI Forecast (Next 3 Days):")
    print(processed[["time", "predicted_aqi"]].head(15))

    # Optional: Save results
    """output_path = os.path.join(artifact_dir, "forecast_predictions.csv")
    processed.to_csv(output_path, index=False)
    print("\n💾 Saved forecast predictions to forecast_predictions.csv")"""

    return processed

