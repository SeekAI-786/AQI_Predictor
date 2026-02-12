# -----------------------------
# Pearls AQI Predictor — Data Fetch Module
# Fetches hourly weather + air quality data from Open-Meteo APIs
# -----------------------------

import requests
import pandas as pd
from datetime import datetime, timedelta
import calendar
import time

# Karachi coordinates
LATITUDE = 24.8607
LONGITUDE = 67.0011
LOCATION = "Karachi"

# Open-Meteo API endpoints
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Columns to fetch
WEATHER_HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation", "rain", "snowfall",
    "surface_pressure", "cloud_cover", "windspeed_10m", "winddirection_10m"
]

AIR_QUALITY_HOURLY_VARS = [
    "pm10", "pm2_5", "carbon_monoxide",
    "nitrogen_dioxide", "sulphur_dioxide", "ozone"
]


def fetch_weather_data(latitude, longitude, start_date, end_date):
    """Fetch hourly weather data from Open-Meteo Archive API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": WEATHER_HOURLY_VARS,
        "timezone": "auto"
    }
    resp = requests.get(WEATHER_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame({"datetime": data["hourly"]["time"]})
    for var in WEATHER_HOURLY_VARS:
        df[var] = data["hourly"].get(var)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def fetch_air_quality_data(latitude, longitude, start_date, end_date):
    """Fetch hourly air quality data from Open-Meteo Air Quality API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": AIR_QUALITY_HOURLY_VARS,
        "timezone": "auto"
    }
    resp = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame({"datetime": data["hourly"]["time"]})
    for var in AIR_QUALITY_HOURLY_VARS:
        df[var] = data["hourly"].get(var)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def fetch_monthly_data(latitude, longitude, start_date, end_date):
    """Fetch one month of merged weather + air quality data."""
    weather_df = fetch_weather_data(latitude, longitude, start_date, end_date)
    aq_df = fetch_air_quality_data(latitude, longitude, start_date, end_date)
    merged = pd.merge(weather_df, aq_df, on="datetime", how="inner")
    merged["location"] = LOCATION
    merged["latitude"] = latitude
    merged["longitude"] = longitude
    return merged


def fetch_range_data(latitude, longitude, start_year, start_month):
    """
    Fetch data month-by-month from a start month/year to today.
    Includes rate-limiting (1s sleep between API calls).
    """
    all_data = []
    end_date = datetime.utcnow()
    current_year = start_year
    current_month = start_month

    while (current_year < end_date.year) or \
          (current_year == end_date.year and current_month <= end_date.month):
        start = datetime(current_year, current_month, 1)
        last_day = calendar.monthrange(current_year, current_month)[1]
        end = datetime(current_year, current_month, last_day)
        if end > end_date:
            end = end_date

        s_str = start.strftime("%Y-%m-%d")
        e_str = end.strftime("%Y-%m-%d")
        print(f"  Fetching {s_str} to {e_str} ...")

        try:
            df = fetch_monthly_data(latitude, longitude, s_str, e_str)
            all_data.append(df)
            print(f"    -> {len(df)} records")
        except requests.exceptions.RequestException as e:
            print(f"    -> ERROR: {e}")

        # Rate limiting
        time.sleep(1)

        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def fetch_and_save(output_path="karachi_weather_pollutants_april2025_to_now.csv",
                   start_year=2025, start_month=4):
    """Main fetch entry point — fetches all data and saves to CSV."""
    print(f"Fetching weather + air quality data for {LOCATION} ...")
    print(f"  Coordinates: ({LATITUDE}, {LONGITUDE})")
    print(f"  Range: {start_year}-{start_month:02d} to today\n")

    df = fetch_range_data(LATITUDE, LONGITUDE, start_year, start_month)

    if df.empty:
        print("No data fetched.")
        return df

    print(f"\nTotal records fetched: {len(df)}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Columns: {list(df.columns)}\n")

    df.to_csv(output_path, index=False)
    print(f"Saved raw data to {output_path}")
    return df


# -------------------------------
if __name__ == "__main__":
    fetch_and_save()
