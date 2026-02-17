# -----------------------------
# Pearls AQI Predictor — Hourly Pipeline (v4)
# Clean, Retry-Safe, Production Ready
# -----------------------------

import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import ASCENDING, errors
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ──
LATITUDE = 24.8607
LONGITUDE = 67.0011
LOCATION = "Karachi"

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION")

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

LOOKBACK_HOURS = 3
HISTORY_HOURS = 48

MAX_API_RETRIES = 3
BASE_DELAY = 20

# ── VARIABLES ──
WEATHER_VARS = [
    "temperature_2m","relative_humidity_2m","dew_point_2m",
    "apparent_temperature","precipitation","rain","snowfall",
    "surface_pressure","pressure_msl",
    "cloud_cover","cloud_cover_low","cloud_cover_mid","cloud_cover_high",
    "windspeed_10m","winddirection_10m","wind_gusts_10m",
    "shortwave_radiation","vapour_pressure_deficit"
]

AQ_VARS = [
    "pm10","pm2_5","carbon_monoxide","nitrogen_dioxide",
    "sulphur_dioxide","ozone","us_aqi",
    "us_aqi_pm2_5","us_aqi_pm10","us_aqi_nitrogen_dioxide",
    "us_aqi_ozone","us_aqi_sulphur_dioxide","us_aqi_carbon_monoxide"
]

# ─────────────────────────────────────────────
# RETRY WRAPPER
# ─────────────────────────────────────────────
def fetch_with_retries(fetch_function):
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            print(f"[RETRY] Attempt {attempt}/{MAX_API_RETRIES}")
            df = fetch_function()
            if not df.empty:
                return df
            print("[RETRY] Empty result.")
        except Exception as e:
            print(f"[RETRY] Failed: {e}")

        if attempt < MAX_API_RETRIES:
            delay = BASE_DELAY * attempt
            print(f"[RETRY] Waiting {delay}s...")
            time.sleep(delay)

    print("[RETRY] All attempts failed — graceful fallback.")
    return pd.DataFrame()

# ─────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────
def fetch_recent_hours():
    print(f"[FETCH] Fetching last {LOOKBACK_HOURS} hours")

    # Weather
    w_resp = requests.get(
        WEATHER_FORECAST_URL,
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": WEATHER_VARS,
            "timezone": "auto",
            "forecast_days": 2
        },
        timeout=20
    )
    w_resp.raise_for_status()
    w_data = w_resp.json()

    weather_df = pd.DataFrame({"datetime": w_data["hourly"]["time"]})
    for var in WEATHER_VARS:
        weather_df[var] = w_data["hourly"].get(var)
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

    # AQ
    aq_resp = requests.get(
        AIR_QUALITY_URL,
        params={
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": AQ_VARS,
            "timezone": "auto",
            "forecast_days": 2
        },
        timeout=20
    )
    aq_resp.raise_for_status()
    aq_data = aq_resp.json()

    aq_df = pd.DataFrame({"datetime": aq_data["hourly"]["time"]})
    for var in AQ_VARS:
        aq_df[var] = aq_data["hourly"].get(var)
    aq_df["datetime"] = pd.to_datetime(aq_df["datetime"])

    merged = pd.merge(weather_df, aq_df, on="datetime", how="inner")

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    cutoff = now - timedelta(hours=LOOKBACK_HOURS - 1)

    recent = merged[
        (merged["datetime"] >= cutoff) &
        (merged["datetime"] <= now)
    ].sort_values("datetime").reset_index(drop=True)

    if recent.empty:
        print("[FETCH] No recent data.")
        return pd.DataFrame()

    print(f"[FETCH] Retrieved {len(recent)} rows.")
    return recent

# ─────────────────────────────────────────────
# MONGO HELPERS
# ─────────────────────────────────────────────
def get_mongo_client():
    client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
    client.admin.command("ping")
    return client

def get_existing_datetimes(datetimes):
    client = get_mongo_client()
    col = client[DB_NAME][COLLECTION_NAME]
    dt_list = [pd.Timestamp(dt).to_pydatetime() for dt in datetimes]
    cursor = col.find({"datetime": {"$in": dt_list}}, {"datetime": 1})
    existing = {pd.Timestamp(doc["datetime"]) for doc in cursor}
    client.close()
    return existing

def upload_single_record(record):
    client = get_mongo_client()
    col = client[DB_NAME][COLLECTION_NAME]
    col.create_index([("datetime", ASCENDING)], unique=True)

    try:
        col.insert_one(record)
        print("[MONGO] Inserted.")
        ok = True
    except errors.DuplicateKeyError:
        print("[MONGO] Duplicate skipped.")
        ok = False

    client.close()
    return ok

# ─────────────────────────────────────────────
# FEATURE ENGINEERING (FIXED 2D)
# ─────────────────────────────────────────────
def engineer_current_hour(raw_df):
    df = raw_df.copy()
    row = df.iloc[0]

    dt = row["datetime"]

    df.loc[df.index[0], "hour_sin"] = np.sin(2*np.pi*dt.hour/24)
    df.loc[df.index[0], "hour_cos"] = np.cos(2*np.pi*dt.hour/24)
    df.loc[df.index[0], "day_of_week_sin"] = np.sin(2*np.pi*dt.weekday()/7)
    df.loc[df.index[0], "day_of_week_cos"] = np.cos(2*np.pi*dt.weekday()/7)
    df.loc[df.index[0], "is_weekend"] = 1.0 if dt.weekday() >= 5 else 0.0

    return df

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_hourly_pipeline():
    print("="*70)
    print(" Pearls AQI Predictor — Hourly Pipeline")
    print(f" Location: {LOCATION}")
    print(f" Time: {datetime.now()}")
    print("="*70)

    raw_df = fetch_with_retries(fetch_recent_hours)

    if raw_df.empty:
        print("[PIPELINE] No data. Graceful exit.")
        return

    fetched_datetimes = raw_df["datetime"].tolist()
    existing = get_existing_datetimes(fetched_datetimes)

    if existing:
        raw_df = raw_df[~raw_df["datetime"].isin(existing)]

    if raw_df.empty:
        print("[PIPELINE] Nothing new.")
        return

    inserted = 0

    for _, row in raw_df.iterrows():
        engineered = engineer_current_hour(pd.DataFrame([row]))
        record = engineered.iloc[0].to_dict()

        if upload_single_record(record):
            inserted += 1

    print(f"[PIPELINE] Inserted {inserted} new records.")
    print("="*70)

if __name__ == "__main__":
    run_hourly_pipeline()
