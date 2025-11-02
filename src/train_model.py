import os
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import json
# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Custom import to fetch data
from mongo_functions import get_features

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME")
collection_name = os.getenv("TEST_COLLECTION_NAME")

# Fetch data
print("📥 Fetching features from MongoDB...")
df = get_features(db_name, collection_name, mongo_uri)

if df is None or df.empty:
    raise ValueError("❌ No data fetched from MongoDB. Please check your connection or data source.")

# Clean and prepare data
df = df.dropna(subset=["us_aqi"])  # drop rows with missing target
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

df = df.drop(columns=[c for c in ["time", "_id"] if c in df.columns])

X = df.drop(columns=["us_aqi"])
y = df["us_aqi"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "LinearRegression": LinearRegression()
}

results = {}
best_model = None
best_score = -np.inf

print("\n🚀 Training models...\n")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    print(f"{name} Results:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.3f}\n")

    # Track best model
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

artifact_dir = "artifacts"
os.makedirs(artifact_dir, exist_ok=True)

# Save model metrics
metrics_path = os.path.join(artifact_dir, "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"📊 Model metrics saved at '{metrics_path}'")

# Save best model
if best_model:
    best_model_path = os.path.join(artifact_dir, "best_model.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"✅ Best model: {best_model_name} (R² = {best_score:.3f}) saved at '{best_model_path}'")

    # Save scaler too (for use during prediction)
    scaler_path = os.path.join(artifact_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved at '{scaler_path}'")

else:
    print("❌ No model was successfully trained.")



