import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import base64
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from pymongo import MongoClient
from pymongo.server_api import ServerApi


# ==============================
# CONFIG (ENV-BASED)
# ==============================

MONGODB_URI = os.getenv("MONGODB_URI")

FEATURE_DB = os.getenv("MONGODB_DB")
FEATURE_COL = os.getenv("MONGODB_COLLECTION")

MODEL_DB = os.getenv("MODEL_DB")
MODEL_COL = os.getenv("MODEL_COL")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable not set")

RAW_POLLUTANTS = {
    'pm2_5','pm10','ozone','nitrogen_dioxide',
    'sulphur_dioxide','carbon_monoxide'
}

MAX_H = 72


# ==============================
# HELPER FUNCTIONS
# ==============================

def build_ar(y, i):
    return np.array([
        y[i],
        y[i-6] if i>=6 else y[i],
        y[i-12] if i>=12 else y[i],
        np.mean(y[max(0,i-24):i+1]),
        np.std(y[max(0,i-24):i+1])
    ])


# ==============================
# 1. FETCH DATA
# ==============================

start_time = datetime.utcnow()

client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
df = pd.DataFrame(list(client[FEATURE_DB][FEATURE_COL].find()))
client.close()

if "_id" in df.columns:
    df = df.drop("_id", axis=1)

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

df = df.dropna(subset=["us_aqi"])
df = df.ffill().bfill()

n_records = len(df)


# ==============================
# 2. FEATURE PREPARATION
# ==============================

feature_cols = [
    c for c in df.columns
    if c not in RAW_POLLUTANTS
    and c not in ["us_aqi","datetime","location","latitude","longitude"]
    and df[c].dtype != "object"
]

X = df[feature_cols].values
y = df["us_aqi"].values

split = int(len(X)*0.8)

X_train = X[:split]
y_train = y[:split]

scaler_global = RobustScaler()
X_train = scaler_global.fit_transform(X_train)


# ==============================
# 3. TRAIN 3-BAND MODELS
# ==============================

bands = {
    "short": range(1,9),
    "medium": [9,12,15,18,21,24],
    "long": [25,30,36,42,48,54,60,66,72]
}

models = {}
band_scalers = {}
band_metrics = {}

for name, horizons in bands.items():

    rows, targets = [], []

    for h in horizons:
        for t in range(24, len(X_train)-h):
            ar = build_ar(y_train, t)
            rows.append(np.concatenate([X_train[t], ar, [h/MAX_H]]))
            targets.append(y_train[t+h])

    rows = np.array(rows)
    targets = np.array(targets)

    scaler_b = RobustScaler()
    rows = scaler_b.fit_transform(rows)

    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(rows, targets)

    preds = model.predict(rows)
    r2 = r2_score(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))

    models[name] = model
    band_scalers[name] = scaler_b
    band_metrics[name] = {
        "r2_train": float(r2),
        "rmse_train": float(rmse),
        "samples": int(len(rows))
    }


# ==============================
# 4. SAVE MODEL (OVERWRITE)
# ==============================

model_blob = pickle.dumps({
    "models": models,
    "band_scalers": band_scalers,
    "global_scaler": scaler_global,
    "features": feature_cols
})

encoded_model = base64.b64encode(model_blob).decode()

training_logs = {
    "records_used": n_records,
    "train_split": 0.8,
    "bands": band_metrics,
    "trained_at_utc": datetime.utcnow(),
    "training_duration_sec": (datetime.utcnow() - start_time).total_seconds()
}

client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
collection = client[MODEL_DB][MODEL_COL]

collection.update_one(
    {"model_name": "pearls_72h"},
    {
        "$set": {
            "model_name": "pearls_72h",
            "model_blob": encoded_model,
            "training_logs": training_logs
        }
    },
    upsert=True
)

client.close()

print("Model retrained and overwritten successfully.")
print("Training Logs:", training_logs)
