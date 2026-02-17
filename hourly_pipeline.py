# -----------------------------
# Pearls AQI Predictor — Hourly Pipeline (v3)
# Full self-contained pipeline with graceful API fallback
# --------------------------

import os
import sys
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import ASCENDING, errors
from dotenv import load_dotenv

load_dotenv()

# ── CONFIGURATION ──
LATITUDE = 24.8607
LONGITUDE = 67.0011
LOCATION = "Karachi"

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION")

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

LOOKBACK_HOURS = 3       # Fetch last 3 hours
HISTORY_HOURS = 48       # MongoDB history for lag/rolling

# ── VARIABLES ──
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "surface_pressure", "pressure_msl",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "windspeed_10m", "winddirection_10m", "wind_gusts_10m",
    "shortwave_radiation", "vapour_pressure_deficit"
]

AQ_VARS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
    "aerosol_optical_depth", "dust", "uv_index", "uv_index_clear_sky", "carbon_dioxide",
    "us_aqi",
    "us_aqi_pm2_5", "us_aqi_pm10", "us_aqi_nitrogen_dioxide",
    "us_aqi_ozone", "us_aqi_sulphur_dioxide", "us_aqi_carbon_monoxide"
]

# ── AQI BREAKPOINT TABLES ──
PM25_BP = [(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
PM10_BP = [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)]
O3_BP = [(0,54,0,50),(55,70,51,100),(71,85,101,150),(86,105,151,200),(106,200,201,300)]
NO2_BP = [(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),(650,1249,201,300)]
SO2_BP = [(0,35,0,50),(36,75,51,100),(76,185,101,150),(186,304,151,200),(305,604,201,300)]
CO_BP = [(0.0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),(12.5,15.4,151,200),(15.5,30.4,201,300)]

def _aqi_sub(C, bp):
    for Cl, Ch, Il, Ih in bp:
        if Cl <= C <= Ch:
            return ((Ih - Il) / (Ch - Cl)) * (C - Cl) + Il
    return None

def compute_aqi(pm25, pm10, o3=None, no2=None, so2=None, co=None):
    subs = {}
    if pm25 is not None and not np.isnan(pm25) and pm25 >= 0: v=_aqi_sub(pm25, PM25_BP); subs['PM2.5']=v if v else None
    if pm10 is not None and not np.isnan(pm10) and pm10 >= 0: v=_aqi_sub(pm10, PM10_BP); subs['PM10']=v if v else None
    if o3 is not None and not np.isnan(o3) and o3 >= 0: v=_aqi_sub(o3/2.0,O3_BP); subs['O3']=v if v else None
    if no2 is not None and not np.isnan(no2) and no2 >= 0: v=_aqi_sub(no2/1.88,NO2_BP); subs['NO2']=v if v else None
    if so2 is not None and not np.isnan(so2) and so2 >= 0: v=_aqi_sub(so2/2.62,SO2_BP); subs['SO2']=v if v else None
    if co is not None and not np.isnan(co) and co >= 0: v=_aqi_sub(co/1145.0,CO_BP); subs['CO']=v if v else None
    if not subs: return np.nan, 'N/A'
    dom=max(subs,key=subs.get)
    return round(subs[dom],1), dom

# ── DATA FETCH ──
def fetch_recent_hours():
    print(f"[FETCH] Fetching last {LOOKBACK_HOURS}h weather + air quality ...")
    try:
        w_resp = requests.get(WEATHER_FORECAST_URL, params={
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "hourly": WEATHER_VARS, "timezone":"auto","forecast_days":2
        }, timeout=30)
        w_resp.raise_for_status()
        w_data=w_resp.json()
        weather_df=pd.DataFrame({"datetime":w_data["hourly"]["time"]})
        for var in WEATHER_VARS: weather_df[var]=w_data["hourly"].get(var)
        weather_df["datetime"]=pd.to_datetime(weather_df["datetime"])
    except requests.exceptions.RequestException as e:
        print(f"[FETCH] Weather API request failed: {e}")
        return pd.DataFrame()
    
    try:
        aq_resp=requests.get(AIR_QUALITY_URL, params={
            "latitude": LATITUDE, "longitude": LONGITUDE,
            "hourly": AQ_VARS, "timezone":"auto","forecast_days":2
        }, timeout=30)
        aq_resp.raise_for_status()
        aq_data=aq_resp.json()
        aq_df=pd.DataFrame({"datetime":aq_data["hourly"]["time"]})
        for var in AQ_VARS: aq_df[var]=aq_data["hourly"].get(var)
        aq_df["datetime"]=pd.to_datetime(aq_df["datetime"])
    except requests.exceptions.RequestException as e:
        print(f"[FETCH] Air quality API request failed: {e}")
        return pd.DataFrame()
    
    merged=pd.merge(weather_df, aq_df, on="datetime", how="inner")
    now=datetime.now().replace(minute=0,second=0,microsecond=0)
    cutoff=now-timedelta(hours=LOOKBACK_HOURS-1)
    recent=merged[(merged["datetime"]>=cutoff)&(merged["datetime"]<=now)]
    recent=recent.sort_values("datetime").reset_index(drop=True)
    
    if recent.empty:
        print("[FETCH] No data for recent hours.")
        return pd.DataFrame()
    
    print(f"[FETCH] Got {len(recent)} hours: {recent['datetime'].min()} to {recent['datetime'].max()}")
    return recent

# ── MONGODB HELPERS ──
def get_mongo_client():
    client=MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    client.admin.command('ping')
    return client

def get_existing_datetimes(datetimes):
    client=get_mongo_client()
    collection=client[DB_NAME][COLLECTION_NAME]
    dt_list=[pd.Timestamp(dt).to_pydatetime() for dt in datetimes]
    cursor=collection.find({"datetime":{"$in":dt_list}},{"datetime":1,"_id":0})
    existing={pd.Timestamp(doc["datetime"]) for doc in cursor}
    client.close()
    return existing

def fetch_recent_history(hours=HISTORY_HOURS):
    client=get_mongo_client()
    collection=client[DB_NAME][COLLECTION_NAME]
    cutoff=datetime.utcnow()-timedelta(hours=hours)
    cursor=collection.find({"datetime":{"$gte":cutoff}},{"_id":0}).sort("datetime",1)
    records=list(cursor)
    client.close()
    if not records: return pd.DataFrame()
    df=pd.DataFrame(records)
    df["datetime"]=pd.to_datetime(df["datetime"])
    if df["datetime"].dt.tz is not None: df["datetime"]=df["datetime"].dt.tz_localize(None)
    print(f"[MONGO] Fetched {len(df)} recent records.")
    return df

def upload_single_record(record_dict):
    client=get_mongo_client()
    collection=client[DB_NAME][COLLECTION_NAME]
    collection.create_index([("datetime", ASCENDING)], unique=True)
    try:
        collection.insert_one(record_dict)
        print("[MONGO] Inserted 1 new record.")
        ok=True
    except errors.DuplicateKeyError:
        print("[MONGO] Duplicate — skipped.")
        ok=False
    total=collection.count_documents({})
    print(f"[MONGO] Total documents: {total}")
    client.close()
    return ok

def prepare_record(row_dict):
    clean={}
    for k,v in row_dict.items():
        if isinstance(v,(np.integer,)): clean[k]=int(v)
        elif isinstance(v,(np.floating,)): clean[k]=None if np.isnan(v) else float(v)
        elif isinstance(v,np.bool_): clean[k]=bool(v)
        elif isinstance(v,pd.Timestamp): clean[k]=v.to_pydatetime()
        elif pd.api.types.is_scalar(v) and pd.isna(v): clean[k]=None
        else: clean[k]=v
    return clean

# ── FEATURE ENGINEERING (simplified for v3) ──
def engineer_current_hour(raw_row, history_df):
    row=raw_row.iloc[0].copy()
    # AQI: use API if available, fallback to compute_aqi
    if 'us_aqi' in raw_row.columns and pd.notna(row.get('us_aqi')):
        raw_row["dominant_pollutant"]=max(
            {c: row.get(c) for c in AQ_VARS if c.startswith("us_aqi_") and pd.notna(row.get(c))}.items(),
            key=lambda x:x[1], default=('N/A',0)
        )[0]
    else:
        aqi_val, dominant=compute_aqi(
            pm25=row.get("pm2_5"), pm10=row.get("pm10"),
            o3=row.get("ozone"), no2=row.get("nitrogen_dioxide"),
            so2=row.get("sulphur_dioxide"), co=row.get("carbon_monoxide")
        )
        raw_row["us_aqi"]=aqi_val
        raw_row["dominant_pollutant"]=dominant
    # Basic time features
    dt=row["datetime"]
    raw_row["hour_sin"]=np.sin(2*np.pi*dt.hour/24)
    raw_row["hour_cos"]=np.cos(2*np.pi*dt.hour/24)
    raw_row["day_of_week_sin"]=np.sin(2*np.pi*dt.weekday()/7)
    raw_row["day_of_week_cos"]=np.cos(2*np.pi*dt.weekday()/7)
    raw_row["is_weekend"]=1.0 if dt.weekday()>=5 else 0.0
    return pd.DataFrame([raw_row])

# ── MAIN PIPELINE ──
def run_hourly_pipeline():
    print("="*70)
    print(" Pearls AQI Predictor — Hourly Pipeline")
    print(f" Location: {LOCATION} ({LATITUDE}, {LONGITUDE})")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    raw_df=fetch_recent_hours()
    if raw_df.empty:
        print("[PIPELINE] No new data available. Skipping run gracefully.")
        return

    fetched_datetimes=raw_df["datetime"].tolist()
    existing=get_existing_datetimes(fetched_datetimes)
    if existing:
        raw_df=raw_df[~raw_df["datetime"].isin(existing)].reset_index(drop=True)
        print(f"[PIPELINE] Skipping {len(fetched_datetimes)-len(raw_df)} duplicate hours.")

    if raw_df.empty:
        print("[PIPELINE] All hours already uploaded. Nothing to do.")
        return

    history_df=fetch_recent_history(hours=HISTORY_HOURS)
    all_engineered=[]
    for i,row in enumerate(raw_df.iterrows()):
        print(f"[PIPELINE] Processing hour {i+1}/{len(raw_df)}: {row[1]['datetime']}")
        engineered=engineer_current_hour(pd.DataFrame([row[1]]), history_df)
        all_engineered.append(engineered)

    inserted=0
    skipped=0
    for eng_row in all_engineered:
        record=prepare_record(eng_row.iloc[0].to_dict())
        ok=upload_single_record(record)
        if ok: inserted+=1
        else: skipped+=1

    print("\n[PIPELINE] Run complete:")
    print(f"  New rows processed: {len(all_engineered)}")
    print(f"  Records inserted: {inserted}")
    print(f"  Duplicates skipped: {skipped}")
    print("="*70)

if __name__=="__main__":
    run_hourly_pipeline()

