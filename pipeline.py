# -----------------------------
# Pearls AQI Predictor — Full Feature Pipeline
# Fetches raw data -> Engineers features -> Uploads to MongoDB
# -----------------------------

import argparse
import sys
from datetime import datetime

from api_data_fetch import fetch_and_save, LATITUDE, LONGITUDE, LOCATION
from feature_engineering import engineer_features, engineer_and_save
from mongodb_upload import upload_to_mongodb, upload_from_csv
import pandas as pd


def run_full_pipeline(start_year=2025, start_month=4,
                      raw_csv="karachi_weather_pollutants_april2025_to_now.csv",
                      features_csv="karachi_aqi_features_cleaned.csv",
                      skip_fetch=False, skip_upload=False):
    """
    Run the complete feature pipeline:
      1. Fetch raw weather + air quality data from Open-Meteo
      2. Engineer features (EPA AQI, time, lags, rolling stats, wind)
      3. Upload engineered features to MongoDB Atlas
    """
    print("=" * 60)
    print(" Pearls AQI Predictor — Feature Pipeline")
    print(f" Location: {LOCATION} ({LATITUDE}, {LONGITUDE})")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ----- STEP 1: Fetch Raw Data -----
    if not skip_fetch:
        print("\n[STEP 1/3] Fetching raw data from Open-Meteo APIs ...")
        raw_df = fetch_and_save(output_path=raw_csv,
                                start_year=start_year,
                                start_month=start_month)
        if raw_df.empty:
            print("ERROR: No data fetched. Aborting pipeline.")
            sys.exit(1)
    else:
        print("\n[STEP 1/3] Skipping fetch — loading existing raw CSV ...")
        raw_df = pd.read_csv(raw_csv, parse_dates=["datetime"])
        print(f"  Loaded {len(raw_df)} rows from {raw_csv}")

    # ----- STEP 2: Feature Engineering -----
    print("\n[STEP 2/3] Running feature engineering ...")
    features_df = engineer_features(raw_df)
    features_df.to_csv(features_csv, index=False)
    print(f"  Saved {len(features_df)} engineered records to {features_csv}")

    # ----- STEP 3: Upload to MongoDB -----
    if not skip_upload:
        print("\n[STEP 3/3] Uploading to MongoDB Atlas ...")
        inserted, skipped, failed = upload_to_mongodb(features_df)
    else:
        print("\n[STEP 3/3] Skipping MongoDB upload (--skip-upload flag)")

    # ----- Summary -----
    print("\n" + "=" * 60)
    print(" Pipeline Complete!")
    print(f"  Raw records: {len(raw_df)}")
    print(f"  Engineered features: {features_df.shape[1]} columns")
    print(f"  Date range: {features_df['datetime'].min()} to {features_df['datetime'].max()}")
    if not skip_upload:
        print(f"  MongoDB: {inserted} inserted, {skipped} skipped, {failed} failed")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Pearls AQI Feature Pipeline")
    parser.add_argument("--start-year", type=int, default=2025,
                        help="Start year for data fetch (default: 2025)")
    parser.add_argument("--start-month", type=int, default=4,
                        help="Start month for data fetch (default: 4)")
    parser.add_argument("--raw-csv", type=str,
                        default="karachi_weather_pollutants_april2025_to_now.csv",
                        help="Path to raw data CSV")
    parser.add_argument("--features-csv", type=str,
                        default="karachi_aqi_features_cleaned.csv",
                        help="Path to output features CSV")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Skip API fetch, use existing raw CSV")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip MongoDB upload")

    args = parser.parse_args()

    run_full_pipeline(
        start_year=args.start_year,
        start_month=args.start_month,
        raw_csv=args.raw_csv,
        features_csv=args.features_csv,
        skip_fetch=args.skip_fetch,
        skip_upload=args.skip_upload
    )


if __name__ == "__main__":
    main()
