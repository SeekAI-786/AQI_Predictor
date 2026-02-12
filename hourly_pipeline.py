# -----------------------------
# Pearls AQI Predictor — Hourly MLOps Pipeline
# Self-contained script: Fetch current hour -> Feature engineer -> Upload to MongoDB
# Designed to run every hour via CI/CD (GitHub Actions, cron, etc.)
# -----------------------------

import requests
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import ASCENDING, errors
from dotenv import load_dotenv
import os
import sys

# =============================================
# Configuration
# =============================================
load_dotenv()

LATITUDE = 24.8607
LONGITUDE = 67.0011
LOCATION = "Karachi"

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB", "aqi_feature_store")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "karachi_aqi_features")

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Only AQI-relevant variables
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m",
    "surface_pressure", "windspeed_10m", "winddirection_10m"
]
AQ_VARS = [
    "pm10", "pm2_5", "carbon_monoxide",
    "nitrogen_dioxide", "sulphur_dioxide", "ozone"
]


# =============================================
# 1. DATA FETCH — Current Hour from Forecast API
# =============================================

def fetch_current_hour_data():
    """Fetch today's forecast data and extract the current hour row."""
    print("[FETCH] Fetching current hour weather + air quality ...")

    # --- Weather ---
    w_resp = requests.get(WEATHER_FORECAST_URL, params={
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "hourly": WEATHER_VARS,
        "timezone": "auto", "forecast_days": 1
    }, timeout=30)
    w_resp.raise_for_status()
    w_data = w_resp.json()

    weather_df = pd.DataFrame({"datetime": w_data["hourly"]["time"]})
    for var in WEATHER_VARS:
        weather_df[var] = w_data["hourly"].get(var)
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

    # --- Air Quality ---
    aq_resp = requests.get(AIR_QUALITY_URL, params={
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "hourly": AQ_VARS,
        "timezone": "auto", "forecast_days": 1
    }, timeout=30)
    aq_resp.raise_for_status()
    aq_data = aq_resp.json()

    aq_df = pd.DataFrame({"datetime": aq_data["hourly"]["time"]})
    for var in AQ_VARS:
        aq_df[var] = aq_data["hourly"].get(var)
    aq_df["datetime"] = pd.to_datetime(aq_df["datetime"])

    # --- Merge and pick current hour ---
    merged = pd.merge(weather_df, aq_df, on="datetime", how="inner")
    
    # Use timezone-aware Pakistan time
    pkt = pytz.timezone("Asia/Karachi")
    now_pkt = datetime.now(pkt)
    
    # Make merged datetime timezone-aware for proper comparison
    if merged["datetime"].dt.tz is None:
        merged["datetime"] = merged["datetime"].dt.tz_localize("UTC")
    
    # Pick the most recent row not later than current PKT hour
    current_hour = merged[merged["datetime"] <= now_pkt].sort_values("datetime").tail(1)

    if current_hour.empty:
        print("[FETCH] No data available for the current hour yet.")
        return pd.DataFrame()

    ts = current_hour.iloc[0]["datetime"]
    print(f"[FETCH] Got data for: {ts} (Pakistan Time: {now_pkt.strftime('%Y-%m-%d %H:%M:%S %Z')})")
    return current_hour.reset_index(drop=True)


# =============================================
# 2. MONGODB HELPERS
# =============================================

def get_mongo_client():
    """Create and verify MongoDB connection."""
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI not set in .env file")
    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    client.admin.command('ping')
    print("[MONGO] Connected to MongoDB Atlas.")
    return client


def fetch_recent_history(hours=48):
    """
    Fetch the last N hours of engineered features from MongoDB.
    Needed to compute lag/rolling features for the new hour.
    """
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Use timezone-aware UTC time
    cutoff = datetime.now(pytz.UTC) - timedelta(hours=hours)
    cursor = collection.find(
        {"datetime": {"$gte": cutoff}},
        {"_id": 0}
    ).sort("datetime", 1)

    records = list(cursor)
    client.close()

    if not records:
        print(f"[MONGO] No recent history found (last {hours}h).")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    print(f"[MONGO] Fetched {len(df)} recent records from MongoDB.")
    return df


def upload_single_record(df_row):
    """Upload a single engineered feature row to MongoDB (skip if duplicate)."""
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.create_index([("datetime", ASCENDING)], unique=True)

    record = prepare_record(df_row)

    try:
        collection.insert_one(record)
        print(f"[MONGO] Inserted 1 new record.")
        result = True
    except errors.DuplicateKeyError:
        print(f"[MONGO] Record for this hour already exists — skipped.")
        result = False
    except Exception as e:
        print(f"[MONGO] Upload error: {e}")
        result = False

    total = collection.count_documents({})
    print(f"[MONGO] Total documents in collection: {total}")
    client.close()
    return result


def prepare_record(df_row):
    """Convert a single-row DataFrame to a MongoDB-safe dict."""
    df_row = df_row.copy()
    if df_row["datetime"].dt.tz is None:
        df_row["datetime"] = df_row["datetime"].dt.tz_localize("UTC")
    else:
        df_row["datetime"] = df_row["datetime"].dt.tz_convert("UTC")

    df_row = df_row.replace([np.inf, -np.inf], np.nan)
    record = df_row.iloc[0].to_dict()

    clean = {}
    for k, v in record.items():
        if isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = None if np.isnan(v) else float(v)
        elif isinstance(v, np.bool_):
            clean[k] = bool(v)
        elif pd.isna(v) if not isinstance(v, str) else False:
            clean[k] = None
        else:
            clean[k] = v
    return clean


# =============================================
# 3. FEATURE ENGINEERING (self-contained)
# =============================================

# --- EPA AQI Breakpoint Tables ---
PM25_BP = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)
]
PM10_BP = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300),
    (425, 504, 301, 400), (505, 604, 401, 500)
]
O3_BP = [
    (0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
    (86, 105, 151, 200), (106, 200, 201, 300)
]
NO2_BP = [
    (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
    (361, 649, 151, 200), (650, 1249, 201, 300)
]
SO2_BP = [
    (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
    (186, 304, 151, 200), (305, 604, 201, 300)
]
CO_BP = [
    (0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300)
]


def aqi_subindex(C, breakpoints):
    for Clow, Chigh, Ilow, Ihigh in breakpoints:
        if Clow <= C <= Chigh:
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (C - Clow) + Ilow
    return None


def compute_aqi(pm25, pm10, o3=None, no2=None, so2=None, co=None):
    """Compute US EPA AQI from pollutant concentrations (ug/m3)."""
    co_ppm = co / 1145.0 if co is not None and not np.isnan(co) else None
    no2_ppb = no2 / 1.88 if no2 is not None and not np.isnan(no2) else None
    so2_ppb = so2 / 2.62 if so2 is not None and not np.isnan(so2) else None
    o3_ppb = o3 / 2.0 if o3 is not None and not np.isnan(o3) else None

    subs = {}
    if pm25 is not None and not np.isnan(pm25):
        v = aqi_subindex(pm25, PM25_BP)
        if v is not None: subs["PM2.5"] = v
    if pm10 is not None and not np.isnan(pm10):
        v = aqi_subindex(pm10, PM10_BP)
        if v is not None: subs["PM10"] = v
    if o3_ppb is not None:
        v = aqi_subindex(o3_ppb, O3_BP)
        if v is not None: subs["O3"] = v
    if no2_ppb is not None:
        v = aqi_subindex(no2_ppb, NO2_BP)
        if v is not None: subs["NO2"] = v
    if so2_ppb is not None:
        v = aqi_subindex(so2_ppb, SO2_BP)
        if v is not None: subs["SO2"] = v
    if co_ppm is not None:
        v = aqi_subindex(co_ppm, CO_BP)
        if v is not None: subs["CO"] = v

    if not subs:
        return np.nan
    return round(max(subs.values()), 1)


def engineer_current_hour(raw_row, history_df):
    """
    Engineer features for the current hour using recent MongoDB history.
    
    1. Append raw_row to history
    2. Compute AQI, time, lag, rolling, wind features on full window
    3. Return only the last row (current hour) with all features
    """
    print("[FEATURES] Engineering features for current hour ...")

    # --- Build raw columns for the new row ---
    raw_row = raw_row.copy()

    # Clean pollutants
    pollutants = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    for col in pollutants:
        if col in raw_row.columns:
            raw_row[col] = pd.to_numeric(raw_row[col], errors="coerce")
            raw_row.loc[raw_row[col] < 0, col] = np.nan

    # Compute AQI for new row
    row = raw_row.iloc[0]
    raw_row["us_aqi"] = compute_aqi(
        pm25=row.get("pm2_5"), pm10=row.get("pm10"),
        o3=row.get("ozone"), no2=row.get("nitrogen_dioxide"),
        so2=row.get("sulphur_dioxide"), co=row.get("carbon_monoxide")
    )

    # --- Wind decomposition ---
    if "windspeed_10m" in raw_row.columns and "winddirection_10m" in raw_row.columns:
        raw_row["wind_u"] = raw_row["windspeed_10m"] * np.cos(np.radians(raw_row["winddirection_10m"]))
        raw_row["wind_v"] = raw_row["windspeed_10m"] * np.sin(np.radians(raw_row["winddirection_10m"]))

    # --- Time features ---
    dt = raw_row["datetime"].iloc[0]
    raw_row["is_weekend"] = int(dt.weekday() >= 5)
    raw_row["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
    raw_row["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)
    raw_row["weekday_sin"] = np.sin(2 * np.pi * dt.weekday() / 7)
    raw_row["weekday_cos"] = np.cos(2 * np.pi * dt.weekday() / 7)
    raw_row["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
    raw_row["month_cos"] = np.cos(2 * np.pi * dt.month / 12)

    # --- Compute lag/rolling/change features using history ---
    if not history_df.empty:
        # Combine history + new row for context
        # History already has engineered cols; we only need the core values
        core_cols = ["datetime", "us_aqi", "pm2_5", "pm10",
                     "carbon_monoxide", "nitrogen_dioxide", "ozone"]
        hist = history_df[[c for c in core_cols if c in history_df.columns]].copy()
        new = raw_row[[c for c in core_cols if c in raw_row.columns]].copy()
        combined = pd.concat([hist, new], ignore_index=True)
        combined = combined.sort_values("datetime").reset_index(drop=True)

        last_idx = len(combined) - 1

        # Change rates (diff from previous hour)
        change_cols = ["pm2_5", "pm10", "carbon_monoxide",
                       "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
        for col in change_cols:
            if col in combined.columns and len(combined) >= 2:
                raw_row[f"{col}_change_rate"] = float(
                    combined[col].iloc[-1] - combined[col].iloc[-2]
                ) if pd.notna(combined[col].iloc[-1]) and pd.notna(combined[col].iloc[-2]) else np.nan
            else:
                raw_row[f"{col}_change_rate"] = np.nan

        # Lag features
        lag_cols = ["us_aqi", "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
        lags = [1, 3, 6, 12, 24]
        for col in lag_cols:
            if col in combined.columns:
                for lag in lags:
                    idx = last_idx - lag
                    if idx >= 0 and pd.notna(combined[col].iloc[idx]):
                        raw_row[f"{col}_lag_{lag}h"] = float(combined[col].iloc[idx])
                    else:
                        raw_row[f"{col}_lag_{lag}h"] = np.nan

        # Rolling features
        for col in ["us_aqi", "pm2_5", "pm10"]:
            if col in combined.columns:
                for w in [6, 12, 24]:
                    window = combined[col].iloc[max(0, last_idx - w + 1):last_idx + 1]
                    raw_row[f"{col}_rolling_mean_{w}h"] = float(window.mean()) if len(window) > 0 else np.nan
                    raw_row[f"{col}_rolling_std_{w}h"] = float(window.std()) if len(window) > 1 else np.nan
    else:
        # No history — fill lag/rolling/change with NaN
        print("[FEATURES] No history available — lag/rolling features will be NaN.")
        change_cols = ["pm2_5", "pm10", "carbon_monoxide",
                       "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
        for col in change_cols:
            raw_row[f"{col}_change_rate"] = np.nan

        lag_cols = ["us_aqi", "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
        for lag in [1, 3, 6, 12, 24]:
            for col in lag_cols:
                raw_row[f"{col}_lag_{lag}h"] = np.nan

        for col in ["us_aqi", "pm2_5", "pm10"]:
            for w in [6, 12, 24]:
                raw_row[f"{col}_rolling_mean_{w}h"] = np.nan
                raw_row[f"{col}_rolling_std_{w}h"] = np.nan

    # --- Drop raw columns not needed in final output ---
    drop_cols = ["windspeed_10m", "winddirection_10m", "location", "latitude", "longitude"]
    raw_row = raw_row.drop(columns=[c for c in drop_cols if c in raw_row.columns])

    print(f"[FEATURES] Engineered {len(raw_row.columns)} columns for current hour.")
    return raw_row


# =============================================
# 4. MAIN PIPELINE
# =============================================

def run_hourly_pipeline():
    """
    Full hourly pipeline:
      1. Fetch current hour data from Open-Meteo forecast API
      2. Fetch last 48h history from MongoDB (for lags/rolling)
      3. Engineer features for the current hour
      4. Upload the single engineered row to MongoDB
    """
    print("=" * 60)
    print(" Pearls AQI Predictor — Hourly Pipeline")
    print(f" Location: {LOCATION} ({LATITUDE}, {LONGITUDE})")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: Fetch current hour raw data
    raw_row = fetch_current_hour_data()
    if raw_row.empty:
        print("ABORT: No current hour data available.")
        sys.exit(1)

    # Step 2: Fetch recent history from MongoDB
    print()
    history_df = fetch_recent_history(hours=48)

    # Step 3: Check for duplicate before processing
    current_ts = raw_row.iloc[0]["datetime"]
    
    # Convert current_ts to UTC for comparison with MongoDB data
    if current_ts.tz is None:
        current_ts_utc = current_ts.tz_localize("UTC")
    else:
        current_ts_utc = current_ts.tz_convert("UTC")
    
    # Compare using normalized datetime (remove microseconds for matching)
    if not history_df.empty:
        history_df["datetime_normalized"] = pd.to_datetime(history_df["datetime"]).dt.floor('H')
        current_normalized = current_ts_utc.floor('H')
        
        if current_normalized in history_df["datetime_normalized"].values:
            print(f"\n[SKIP] Data for {current_ts_utc} already exists in MongoDB. No action needed.")
            print("=" * 60)
            return
        
        history_df.drop(columns=["datetime_normalized"], inplace=True)

    # Step 4: Engineer features
    print()
    engineered_row = engineer_current_hour(raw_row, history_df)

    # Step 5: Upload to MongoDB
    print()
    upload_single_record(engineered_row)

    print("\n" + "=" * 60)
    print(" Hourly Pipeline Complete!")
    print(f"  Timestamp: {current_ts}")
    print(f"  AQI: {engineered_row.iloc[0].get('us_aqi', 'N/A')}")
    print(f"  Columns: {len(engineered_row.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    run_hourly_pipeline()
