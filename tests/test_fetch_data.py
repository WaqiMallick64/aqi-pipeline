import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# Coordinates for Karachi
latitude = 24.8608
longitude = 67.0104

# === 1️⃣ Dynamic date range: from today back to one year ===
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=365)

# === 2️⃣ Create data folder if not exists ===
os.makedirs("data/processed", exist_ok=True)

# === 3️⃣ Helper functions ===
def fetch_weather(start, end):
    """Fetch hourly weather data from Open-Meteo API"""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start}&end_date={end}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_air(start, end):
    """Fetch hourly air quality data from Open-Meteo API"""
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start}&end_date={end}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


# === 4️⃣ Loop through in monthly chunks (30 days each) ===
delta = timedelta(days=30)
current_start = start_date

weather_all, air_all = [], []

while current_start < end_date:
    current_end = min(current_start + delta, end_date)
    s = current_start.strftime("%Y-%m-%d")
    e = current_end.strftime("%Y-%m-%d")

    try:
        print(f"Fetching data from {s} → {e}")
        wdf = fetch_weather(s, e)
        adf = fetch_air(s, e)
        weather_all.append(wdf)
        air_all.append(adf)
    except Exception as ex:
        print(f"❌ Failed {s} → {e}: {ex}")

    current_start = current_end  # Move to next window

# === 5️⃣ Merge and save ===
if weather_all and air_all:
    weather_df = pd.concat(weather_all, ignore_index=True)
    air_df = pd.concat(air_all, ignore_index=True)

    merged_df = pd.merge(weather_df, air_df, on="time")
    output_path = f"data/raw/karachi_weather_airquality_{start_date}_{end_date}.csv"
    merged_df.to_csv(output_path, index=False)

    print(f"\n✅ Saved merged dataset to: {output_path}")
    print(f"📊 Final shape: {merged_df.shape}")
    print(merged_df.head())
else:
    print("⚠️ No data fetched. Please check the API or date range.")
