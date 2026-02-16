# -------------------------------------------------------------
# Pearls AQI Predictor — Model Retrain Pipeline
# Runs every 12 hours via GitHub Actions
# Fetch features from MongoDB → Train 3-band models → Save to MongoDB
# Overwrites previous model, keeps training logs with it
# -------------------------------------------------------------

import os
import sys
import base64
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pymongo import MongoClient
from pymongo.server_api import ServerApi

warnings.filterwarnings("ignore")

# ── Configuration (from environment / GitHub Secrets) ──
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://mohammadaliaun7_db_user:fJjD83zeRYhJi3wc"
    "@aqi.yqustuk.mongodb.net/?appName=AQI"
)
FEATURE_DB = os.getenv("FEATURE_DB", "aqi_feature_store")
FEATURE_COL = os.getenv("FEATURE_COL", "karachi_aqi_features")
MODEL_DB = os.getenv("MODEL_DB", "aqi_model_store")
MODEL_COL = os.getenv("MODEL_COL", "pearls_72h_models")

RAW_POLLUTANTS = {
    "pm2_5", "pm10", "ozone", "nitrogen_dioxide",
    "sulphur_dioxide", "carbon_monoxide",
}

MAX_H = 72

BANDS = {
    "short": list(range(1, 9)),
    "medium": [9, 12, 15, 18, 21, 24],
    "long": [25, 30, 36, 42, 48, 54, 60, 66, 72],
}


# ═══════════════════════════════════════════════════════════════
# STEP 1: FETCH DATA FROM MONGODB
# ═══════════════════════════════════════════════════════════════

def fetch_features():
    """Fetch all feature-engineered records from MongoDB."""
    print("[1/4] FETCHING DATA FROM MONGODB")
    print("-" * 50)

    client = MongoClient(MONGODB_URI, server_api=ServerApi("1"),
                         serverSelectionTimeoutMS=15000)
    client.admin.command("ping")
    print("  Connected to MongoDB")

    df = pd.DataFrame(list(client[FEATURE_DB][FEATURE_COL].find()))
    client.close()

    if "_id" in df.columns:
        df.drop("_id", axis=1, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Records: {len(df):,}")
    print(f"  Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  Columns: {len(df.columns)}")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2: DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

def prepare_data(df):
    """Clean data, select features, split, scale."""
    print("\n[2/4] DATA PREPARATION")
    print("-" * 50)

    # Clean
    df = df.dropna(subset=["us_aqi"])
    df = df.ffill().bfill()

    # Feature selection — exclude raw pollutants, target, metadata, non-numeric cols
    exclude = RAW_POLLUTANTS | {
        "us_aqi", "datetime", "location", "latitude", "longitude",
        "dominant_pollutant",
    }
    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols].values
    y = df["us_aqi"].values
    dates = df["datetime"].values

    # 80/20 temporal split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = dates[split:]

    # Scale
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  AQI range: [{y.min():.0f}, {y.max():.0f}]")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "dates_test": dates_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 3: TRAIN 3-BAND MODELS
# ═══════════════════════════════════════════════════════════════

def build_ar(y, i):
    """Build autoregressive features at index i."""
    return np.array([
        y[i],
        y[i - 6] if i >= 6 else y[i],
        y[i - 12] if i >= 12 else y[i],
        np.mean(y[max(0, i - 24):i + 1]),
        np.std(y[max(0, i - 24):i + 1]),
    ])


def train_models(data):
    """Train short/medium/long band LightGBM models."""
    print("\n[3/4] MODEL TRAINING")
    print("-" * 50)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    models = {}
    band_scalers = {}
    band_metrics = {}

    for name, horizons in BANDS.items():
        print(f"\n  Training {name} band (h={horizons[0]}-{horizons[-1]}) ...")

        # Build training samples
        rows, targets = [], []
        for h in horizons:
            for t in range(24, len(X_train) - h):
                ar = build_ar(y_train, t)
                rows.append(np.concatenate([X_train[t], ar, [h / MAX_H]]))
                targets.append(y_train[t + h])

        rows = np.array(rows)
        targets = np.array(targets)

        # Scale band inputs
        scaler_b = RobustScaler()
        rows = scaler_b.fit_transform(rows)

        # Train
        model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(rows, targets)

        models[name] = model
        band_scalers[name] = scaler_b

        # Evaluate on test set
        test_rows, test_targets = [], []
        for h in horizons:
            for t in range(24, len(X_test) - h):
                ar = build_ar(y_test, t)
                test_rows.append(np.concatenate([X_test[t], ar, [h / MAX_H]]))
                test_targets.append(y_test[t + h])

        if test_rows:
            test_rows = scaler_b.transform(np.array(test_rows))
            test_targets = np.array(test_targets)
            preds = model.predict(test_rows)
            rmse = float(np.sqrt(mean_squared_error(test_targets, preds)))
            mae = float(mean_absolute_error(test_targets, preds))
            r2 = float(r2_score(test_targets, preds))
        else:
            rmse, mae, r2 = 0.0, 0.0, 0.0

        band_metrics[name] = {"rmse": rmse, "mae": mae, "r2": r2, "samples": len(rows)}
        print(f"    Samples: {len(rows):,} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")

    return models, band_scalers, band_metrics


# ═══════════════════════════════════════════════════════════════
# STEP 4: SAVE MODELS TO MONGODB (overwrite previous)
# ═══════════════════════════════════════════════════════════════

def save_models(models, band_scalers, band_metrics, data):
    """Serialize models + scalers and overwrite in MongoDB with training logs."""
    print("\n[4/4] SAVING MODELS TO MONGODB")
    print("-" * 50)

    client = MongoClient(MONGODB_URI, server_api=ServerApi("1"),
                         serverSelectionTimeoutMS=15000)
    model_collection = client[MODEL_DB][MODEL_COL]

    now = datetime.utcnow()

    for name in models:
        # Serialize model + band scaler together
        blob = pickle.dumps({
            "model": models[name],
            "scaler": band_scalers[name],
        })

        training_log = {
            "trained_at": now,
            "horizons": BANDS[name],
            "metrics": band_metrics[name],
            "feature_count": len(data["feature_cols"]),
            "train_size": len(data["X_train"]),
            "test_size": len(data["X_test"]),
        }

        # Overwrite previous model document for this band
        model_collection.update_one(
            {"band": name},
            {"$set": {
                "band": name,
                "model_blob": base64.b64encode(blob).decode(),
                "feature_cols": data["feature_cols"],
                "training_log": training_log,
                "created_at": now,
            }},
            upsert=True,
        )
        print(f"  {name}: saved (R²={band_metrics[name]['r2']:.4f})")

    # Save the main feature scaler alongside models
    scaler_blob = pickle.dumps(data["scaler"])
    model_collection.update_one(
        {"band": "_scaler"},
        {"$set": {
            "band": "_scaler",
            "model_blob": base64.b64encode(scaler_blob).decode(),
            "feature_cols": data["feature_cols"],
            "created_at": now,
        }},
        upsert=True,
    )
    print("  _scaler: saved")

    client.close()
    print(f"\n  Models written to {MODEL_DB}.{MODEL_COL}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_retrain():
    """Full retrain pipeline: Fetch → Prepare → Train → Save."""
    print("=" * 70)
    print(" Pearls AQI Predictor — Model Retrain Pipeline")
    print(f" Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 70)

    # 1. Fetch
    df = fetch_features()

    # 2. Prepare
    data = prepare_data(df)

    # 3. Train
    models, band_scalers, band_metrics = train_models(data)

    # 4. Save
    save_models(models, band_scalers, band_metrics, data)

    # Summary
    print("\n" + "=" * 70)
    print(" RETRAIN COMPLETE")
    print("=" * 70)
    for name in BANDS:
        m = band_metrics[name]
        print(f"  {name:>8}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R²={m['r2']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_retrain()
