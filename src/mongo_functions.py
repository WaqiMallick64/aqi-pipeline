import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime


def get_mongo_client(MONGO_URI):
    """Connect to MongoDB Atlas and return the client."""
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command("ping")  # test connection
        print("✅ MongoDB connection successful.")
        return client
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise


def ensure_db_and_collection(client,DB_NAME,COLLECTION_NAME):
    """Ensure the database and collection exist."""
    db = client[DB_NAME]
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(COLLECTION_NAME)
        print(f"📦 Created collection '{COLLECTION_NAME}' in DB '{DB_NAME}'")
    return db[COLLECTION_NAME]

def preprocess_features(df):
    """Clean and enhance the fetched data for storage or modeling."""
    if df.empty:
        print("⚠️ Empty DataFrame — nothing to preprocess.")
        return df

    # Drop rows without AQI values
    df = df.dropna(subset=["us_aqi"])

    # Fill missing numeric values with median
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    # Add temporal features
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["dayofweek"] = df["time"].dt.dayofweek

    # Sort by time
    df = df.sort_values("time").reset_index(drop=True)

    print(f"🧹 Preprocessed {len(df)} records with {len(df.columns)} columns.")
    return df


def store_features(df: pd.DataFrame,COLLECTION_NAME):
    """
    Uploads processed feature dataframe to MongoDB.
    If records for the same 'time' exist, they are updated.
    """
    client = get_mongo_client()
    collection = ensure_db_and_collection(client)

    # Preprocess before storing
    df = preprocess_features(df)

    records = df.to_dict("records")
    for record in records:
        record["time"] = pd.to_datetime(record["time"]).to_pydatetime()
        collection.update_one(
            {"time": record["time"]},
            {"$set": record},
            upsert=True
        )

    print(f"✅ Stored/updated {len(records)} processed feature records in MongoDB collection '{COLLECTION_NAME}'")
    client.close()


def get_features(DB_NAME,COLLECTION_NAME, start_date=None, end_date=None ):
    """
    Fetch feature data from MongoDB between specific dates.
    If no range is given, return all.
    """
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    query = {}
    if start_date and end_date:
        query = {"time": {"$gte": start_date, "$lte": end_date}}

    cursor = collection.find(query)
    df = pd.DataFrame(list(cursor))

    if not df.empty:
        df.drop("_id", axis=1, inplace=True)

    client.close()
    print(f"📤 Retrieved {len(df)} records from MongoDB")
    return df


def save_to_mongodb(df,collection_name,collection):
    
    if df.empty:
        print("ℹ️ No data to insert.")
        return

    records = df.to_dict("records")
    for record in records:
        record["time"] = pd.to_datetime(record["time"]).to_pydatetime()
        collection.update_one({"time": record["time"]}, {"$set": record}, upsert=True)

    print(f"✅ Stored/updated {len(records)} record(s) in MongoDB ({collection_name})")