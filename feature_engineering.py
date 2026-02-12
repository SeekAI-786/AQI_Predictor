# -----------------------------
# Pearls AQI Predictor — Feature Engineering Module
# Computes US EPA AQI, time features, lag features,
# change rates, wind vectors, and cyclical encodings
# -----------------------------

import pandas as pd
import numpy as np


# =============================================
# US EPA AQI Calculation (Official Breakpoints)
# =============================================

# Breakpoint tables: (C_low, C_high, I_low, I_high)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500)
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500)
]

O3_BREAKPOINTS = [  # in ppb (8-hour)
    (0, 54, 0, 50),
    (55, 70, 51, 100),
    (71, 85, 101, 150),
    (86, 105, 151, 200),
    (106, 200, 201, 300)
]

NO2_BREAKPOINTS = [  # in ppb
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300)
]

SO2_BREAKPOINTS = [  # in ppb
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300)
]

CO_BREAKPOINTS = [  # in ppm
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300)
]


def calculate_aqi_subindex(C, breakpoints):
    """Calculate AQI sub-index for a single pollutant using EPA linear interpolation."""
    for Clow, Chigh, Ilow, Ihigh in breakpoints:
        if Clow <= C <= Chigh:
            return ((Ihigh - Ilow) / (Chigh - Clow)) * (C - Clow) + Ilow
    return None


def calculate_aqi(pm25, pm10, o3=None, no2=None, so2=None, co=None):
    """
    Calculate overall US EPA AQI from pollutant concentrations.

    Input units (from Open-Meteo):
      - pm25: ug/m3
      - pm10: ug/m3
      - o3: ug/m3  -> converted to ppb (/ 2.0)
      - no2: ug/m3 -> converted to ppb (/ 1.88)
      - so2: ug/m3 -> converted to ppb (/ 2.62)
      - co: ug/m3  -> converted to ppm (/ 1145)

    Returns: (aqi_value, dominant_pollutant)
    """
    # Unit conversions from ug/m3 to ppm/ppb
    co_ppm = co / 1145.0 if co is not None and not np.isnan(co) else None
    no2_ppb = no2 / 1.88 if no2 is not None and not np.isnan(no2) else None
    so2_ppb = so2 / 2.62 if so2 is not None and not np.isnan(so2) else None
    o3_ppb = o3 / 2.0 if o3 is not None and not np.isnan(o3) else None

    subindices = {}

    if pm25 is not None and not np.isnan(pm25):
        val = calculate_aqi_subindex(pm25, PM25_BREAKPOINTS)
        if val is not None:
            subindices["PM2.5"] = val

    if pm10 is not None and not np.isnan(pm10):
        val = calculate_aqi_subindex(pm10, PM10_BREAKPOINTS)
        if val is not None:
            subindices["PM10"] = val

    if o3_ppb is not None:
        val = calculate_aqi_subindex(o3_ppb, O3_BREAKPOINTS)
        if val is not None:
            subindices["O3"] = val

    if no2_ppb is not None:
        val = calculate_aqi_subindex(no2_ppb, NO2_BREAKPOINTS)
        if val is not None:
            subindices["NO2"] = val

    if so2_ppb is not None:
        val = calculate_aqi_subindex(so2_ppb, SO2_BREAKPOINTS)
        if val is not None:
            subindices["SO2"] = val

    if co_ppm is not None:
        val = calculate_aqi_subindex(co_ppm, CO_BREAKPOINTS)
        if val is not None:
            subindices["CO"] = val

    if not subindices:
        return np.nan, "N/A"

    dominant_pollutant = max(subindices, key=subindices.get)
    aqi_value = round(subindices[dominant_pollutant], 1)
    return aqi_value, dominant_pollutant


def get_aqi_category(aqi):
    """Return AQI category label based on EPA scale."""
    if pd.isna(aqi):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# =============================================
# Feature Engineering Functions
# =============================================

def add_aqi_features(df):
    """Compute US EPA AQI and dominant pollutant for each row."""
    print("  Computing EPA AQI ...")
    results = df.apply(
        lambda row: calculate_aqi(
            pm25=row.get("pm2_5"),
            pm10=row.get("pm10"),
            o3=row.get("ozone"),
            no2=row.get("nitrogen_dioxide"),
            so2=row.get("sulphur_dioxide"),
            co=row.get("carbon_monoxide")
        ), axis=1
    )
    df["us_aqi"] = results.apply(lambda x: x[0])
    df["dominant_pollutant"] = results.apply(lambda x: x[1])
    df["aqi_category"] = df["us_aqi"].apply(get_aqi_category)
    return df


def add_time_features(df):
    """Extract time-based features from datetime column."""
    print("  Adding time features ...")
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Cyclical encoding (helps ML models understand periodicity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_change_rate_features(df):
    """Compute hourly change rates for pollutants and AQI."""
    print("  Adding change rate features ...")
    change_cols = [
        "pm2_5", "pm10", "carbon_monoxide",
        "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"
    ]
    for col in change_cols:
        if col in df.columns:
            df[f"{col}_change_rate"] = df[col].diff()
    return df


def add_lag_features(df):
    """Add lag features for AQI and key pollutants (1h, 3h, 6h, 12h, 24h)."""
    print("  Adding lag features ...")
    lag_cols = ["us_aqi", "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "ozone"]
    lags = [1, 3, 6, 12, 24]
    for col in lag_cols:
        if col in df.columns:
            for lag in lags:
                df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(df):
    """Add rolling mean/std for AQI and PM2.5 (6h, 12h, 24h windows)."""
    print("  Adding rolling features ...")
    windows = [6, 12, 24]
    for col in ["us_aqi", "pm2_5", "pm10"]:
        if col in df.columns:
            for w in windows:
                df[f"{col}_rolling_mean_{w}h"] = df[col].rolling(window=w, min_periods=1).mean()
                df[f"{col}_rolling_std_{w}h"] = df[col].rolling(window=w, min_periods=1).std()
    return df


def add_wind_features(df):
    """Decompose wind into u/v components (useful for ML)."""
    print("  Adding wind vector features ...")
    if "windspeed_10m" in df.columns and "winddirection_10m" in df.columns:
        df["wind_u"] = df["windspeed_10m"] * np.cos(np.radians(df["winddirection_10m"]))
        df["wind_v"] = df["windspeed_10m"] * np.sin(np.radians(df["winddirection_10m"]))
    return df


def clean_data(df):
    """Clean raw data: cap impossible values, replace negatives with NaN."""
    print("  Cleaning data ...")
    pollutants = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    caps = {
        "pm2_5": 1000, "pm10": 1500, "carbon_monoxide": 50000,
        "nitrogen_dioxide": 1000, "sulphur_dioxide": 1000, "ozone": 500
    }

    for col in pollutants:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan
            if col in caps:
                df.loc[df[col] > caps[col], col] = np.nan
    return df


def select_final_columns(df):
    """Select only relevant columns for AQI prediction (drop raw/redundant/irrelevant cols)."""
    drop_cols = [
        # Raw weather cols already replaced by engineered features
        "dew_point_2m", "apparent_temperature", "cloud_cover",
        "snowfall", "rain", "precipitation",
        # Wind raw cols replaced by wind_u / wind_v decomposition
        "windspeed_10m", "winddirection_10m",
        # Not standard AQI inputs / redundant with PM measures
        "aerosol_optical_depth", "dust", "uv_index",
        # Categorical metadata — not numeric predictors
        "dominant_pollutant", "aqi_category",
        # Raw time cols replaced by cyclical sin/cos encodings
        "hour", "day", "month", "weekday", "day_of_year",
        # Constant for single-city dataset
        "location", "latitude", "longitude",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def engineer_features(df):
    """
    Full feature engineering pipeline.
    Input: raw merged weather + air quality DataFrame
    Output: feature-engineered DataFrame ready for MongoDB
    """
    print("\n--- Feature Engineering Pipeline ---")

    # Ensure datetime is proper type
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Step 1: Clean raw data
    df = clean_data(df)

    # Step 2: Compute AQI
    df = add_aqi_features(df)

    # Step 3: Time features
    df = add_time_features(df)

    # Step 4: Change rates
    df = add_change_rate_features(df)

    # Step 5: Lag features
    df = add_lag_features(df)

    # Step 6: Rolling statistics
    df = add_rolling_features(df)

    # Step 7: Wind decomposition
    df = add_wind_features(df)

    # Step 8: Select final columns
    df = select_final_columns(df)

    # Summary
    print(f"\n  Final shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  AQI NaN count: {df['us_aqi'].isna().sum()} / {len(df)}")
    print("--- Feature Engineering Complete ---\n")

    return df


def engineer_and_save(input_csv="karachi_weather_pollutants_april2025_to_now.csv",
                      output_csv="karachi_aqi_features_cleaned.csv"):
    """Load raw CSV, engineer features, save result."""
    print(f"Loading raw data from {input_csv} ...")
    df = pd.read_csv(input_csv, parse_dates=["datetime"])
    print(f"  Raw rows: {len(df)}")

    df = engineer_features(df)

    df.to_csv(output_csv, index=False)
    print(f"Saved engineered features to {output_csv}")
    return df


# -------------------------------
if __name__ == "__main__":
    engineer_and_save()
