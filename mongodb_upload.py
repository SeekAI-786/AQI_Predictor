# -----------------------------
# Pearls AQI Predictor — MongoDB Upload Module
# Uploads only engineered features to MongoDB Atlas
# -----------------------------

import pandas as pd
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import ASCENDING, errors
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB", "aqi_feature_store")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "karachi_aqi_features")


def get_mongo_client():
    """Create and test MongoDB connection."""
    if not MONGODB_URI:
        raise ValueError("MONGODB_URI not set in .env file")

    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("Connected to MongoDB Atlas successfully.")
    except errors.ServerSelectionTimeoutError as e:
        raise ConnectionError(f"MongoDB server not reachable: {e}")
    except errors.OperationFailure as e:
        raise ConnectionError(f"MongoDB auth failed: {e}")
    return client


def prepare_records(df):
    """
    Convert DataFrame to list of MongoDB documents.
    - Converts datetime to UTC
    - Replaces NaN/inf with None (MongoDB-safe)
    - Converts numpy types to Python native types
    """
    df = df.copy()

    # Ensure datetime is timezone-aware (UTC)
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert("UTC")

    # Replace inf/-inf with NaN, then NaN with None
    df = df.replace([np.inf, -np.inf], np.nan)

    records = df.to_dict("records")

    # Convert numpy types to native Python types for MongoDB
    clean_records = []
    for record in records:
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
        clean_records.append(clean)

    return clean_records


def upload_to_mongodb(df, batch_size=1000):
    """
    Upload engineered feature DataFrame to MongoDB.
    - Creates unique index on datetime to prevent duplicates
    - Uses bulk insert_many with ordered=False (skips duplicates)
    - Processes in batches for efficiency
    """
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Create unique index on datetime
    collection.create_index([("datetime", ASCENDING)], unique=True)
    print(f"Target: {DB_NAME}.{COLLECTION_NAME}")
    print(f"Records to upload: {len(df)}")

    records = prepare_records(df)

    inserted = 0
    skipped = 0
    failed = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            result = collection.insert_many(batch, ordered=False)
            inserted += len(result.inserted_ids)
        except errors.BulkWriteError as bwe:
            # Some inserts succeeded, some were duplicates
            write_errors = bwe.details.get("writeErrors", [])
            n_inserted = bwe.details.get("nInserted", 0)
            inserted += n_inserted
            dup_count = sum(1 for e in write_errors if e.get("code") == 11000)
            skipped += dup_count
            failed += len(write_errors) - dup_count
        except Exception as e:
            failed += len(batch)
            print(f"  Batch error: {e}")

        progress = min(i + batch_size, len(records))
        print(f"  Progress: {progress}/{len(records)}")

    print(f"\nUpload complete:")
    print(f"  New records inserted: {inserted}")
    print(f"  Duplicates skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total documents in collection: {collection.count_documents({})}")

    client.close()
    return inserted, skipped, failed


def upload_from_csv(csv_path=None):
    """Load engineered features CSV and upload to MongoDB."""
    csv_path = csv_path or os.getenv("CSV_PATH", "karachi_aqi_features_cleaned.csv")
    print(f"Loading features from {csv_path} ...")

    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    print(f"  Loaded {len(df)} records")

    return upload_to_mongodb(df)


# -------------------------------
if __name__ == "__main__":
    upload_from_csv()
