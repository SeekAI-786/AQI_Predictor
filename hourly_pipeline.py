# ============================================
# HOURLY INCREMENTAL AQI PIPELINE
# Fetch last 6 hours → Feature Engineer → Append to MongoDB
# ============================================

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------
# CONFIG
# -----------------------
LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION")

# -----------------------
# SAFE TIME RANGE
# -----------------------

now = datetime.utcnow()

# Always fetch completed hours only
end_time = now - timedelta(hours=1)
start_time = end_time - timedelta(hours=6)

start_date = start_time.date()
end_date = end_time.date()

# -----------------------
# RETRY SESSION (Production Safe)
# -----------------------

session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# -----------------------
# 1️⃣ Fetch Air Quality
# -----------------------

aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

aq_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "hourly": ",".join([
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "us_aqi"
    ]),
    "start_date": str(start_date),
    "end_date": str(end_date),
    "timezone": TIMEZONE
}

aq_resp = session.get(aq_url, params=aq_params, timeout=60)
aq_resp.raise_for_status()
aq_json = aq_resp.json()

if "hourly" not in aq_json:
    print("No AQ data returned.")
    exit(0)

aq_df = pd.DataFrame(aq_json["hourly"])
aq_df["datetime"] = pd.to_datetime(aq_df["time"])
aq_df.drop(columns=["time"], inplace=True)

# Filter exact 6-hour window
aq_df = aq_df[(aq_df["datetime"] >= start_time) & (aq_df["datetime"] <= end_time)]

# -----------------------
# 2️⃣ Fetch Weather
# -----------------------

weather_url = "https://archive-api.open-meteo.com/v1/archive"

weather_params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": str(start_date),
    "end_date": str(end_date),
    "hourly": ",".join([
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "cloud_cover"
    ]),
    "timezone": TIMEZONE
}

weather_resp = session.get(weather_url, params=weather_params, timeout=60)
weather_resp.raise_for_status()
weather_json = weather_resp.json()

weather_df = pd.DataFrame(weather_json["hourly"])
weather_df["datetime"] = pd.to_datetime(weather_df["time"])
weather_df.drop(columns=["time"], inplace=True)

weather_df = weather_df[(weather_df["datetime"] >= start_time) & (weather_df["datetime"] <= end_time)]

# -----------------------
# 3️⃣ Merge
# -----------------------

df = pd.merge(aq_df, weather_df, on="datetime", how="inner")

if df.empty:
    print("No new hourly rows available.")
    exit(0)

# -----------------------
# 4️⃣ Feature Engineering (Same as historical)
# -----------------------

df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["day_of_week"] = df["datetime"].dt.dayofweek

df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

df["wind_u"] = df["wind_speed_10m"] * np.cos(np.deg2rad(df["wind_direction_10m"]))
df["wind_v"] = df["wind_speed_10m"] * np.sin(np.deg2rad(df["wind_direction_10m"]))

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# -----------------------
# 5️⃣ MongoDB Append (No Duplicates)
# -----------------------

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Ensure unique index (only created once)
collection.create_index([("datetime", ASCENDING)], unique=True)

new_records = []

for record in df.to_dict("records"):
    if not collection.find_one({"datetime": record["datetime"]}):
        new_records.append(record)

if new_records:
    collection.insert_many(new_records)
    print(f"Inserted {len(new_records)} new hourly records.")
else:
    print("No new records to insert (all duplicates).")

print(f"Processed window: {start_time} → {end_time}")
