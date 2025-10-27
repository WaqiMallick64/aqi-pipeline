import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import pytz  # optional, ensures timezone correctness
from mongo_functions import preprocess_features,save_to_mongodb

# Load environment variables
load_dotenv()

# MongoDB setup
client = MongoClient(os.getenv("MONGO_URI"))
db_name = os.getenv("DB_NAME")
db = client[db_name]
collection_name = os.getenv("TEST_COLLECTION_NAME")

if not collection_name:
    raise ValueError("❌ TEST_COLLECTION_NAME not found in environment variables")

# Ensure collection exists
if collection_name not in db.list_collection_names():
    db.create_collection(collection_name)
collection = db[collection_name]

# Coordinates for Karachi
latitude = 24.8608
longitude = 67.0104


def fetch_data_for_today():
    """Fetch combined air quality and weather data for today's date."""
    # Get current date in UTC (you can change to local timezone if needed)
    today = datetime.now(pytz.UTC).strftime("%Y-%m-%d")
    
    print(f"📅 Fetching today's data ({today})")

    # Weather API
    weather_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={today}&end_date={today}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )

    # Air quality API
    air_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={today}&end_date={today}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
        f"sulphur_dioxide,ozone,us_aqi"
    )

    # Fetch data
    wresp = requests.get(weather_url).json()
    aresp = requests.get(air_url).json()

    # Validate responses
    if "hourly" not in wresp or "hourly" not in aresp:
        print("⚠️ Missing 'hourly' data in response.")
        return pd.DataFrame()

    # Convert to DataFrames
    wdf = pd.DataFrame(wresp["hourly"])
    adf = pd.DataFrame(aresp["hourly"])

    # Merge on time
    df = pd.merge(wdf, adf, on="time")

    # Convert timestamps to datetime (UTC)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    print(f"✅ Retrieved {len(df)} hourly records for {today}.")
    return df


if __name__ == "__main__":
    df = fetch_data_for_today()
    processed = preprocess_features(df)
    print(processed.head())
    #save_to_mongodb(processed,collection_name,collection)
