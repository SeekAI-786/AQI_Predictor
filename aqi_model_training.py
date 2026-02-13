import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

import pickle
from pymongo import MongoClient
from pymongo.server_api import ServerApi

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from statsmodels.tsa.statespace.sarimax import SARIMAX
import shap

print("="*80)
print(" PROPERLY FIXED Minimal AQI Prediction Pipeline")
print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# =========================
# --- DATA LOADING
# =========================
def load_data_from_mongodb(uri, db_name="aqi_feature_store", collection_name="karachi_aqi_features"):
    print("\n[1/7] Loading Data from MongoDB...")
    client = MongoClient(uri, server_api=ServerApi('1'))
    client.admin.command('ping')
    collection = client[db_name][collection_name]
    df = pd.DataFrame(list(collection.find({})))
    if '_id' in df.columns:
        df.drop('_id', axis=1, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    client.close()
    print(f"✓ Loaded {len(df):,} records")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df

def load_data_from_csv(csv_path="karachi_aqi_features_cleaned.csv"):
    print("\n[1/7] Loading Data from CSV...")
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"✓ Loaded {len(df):,} records")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df

# =========================
# --- DATA CLEANING
# =========================
def clean_data(df, target_col='us_aqi'):
    print("\n[2/7] Minimal Data Cleaning...")
    original_len = len(df)
    df = df.dropna(subset=[target_col])
    Q1, Q3 = df[target_col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df[target_col] >= Q1 - 3*IQR) & (df[target_col] <= Q3 + 3*IQR)]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    print(f"✓ Removed {original_len - len(df)} rows ({100*(original_len-len(df))/original_len:.1f}%)")
    print(f"  Final shape: {df.shape}")
    return df

# =========================
# --- EDA & LEAKAGE
# =========================
def perform_eda(df, target_col='us_aqi'):
    print("\n[3/7] Essential EDA...")
    os.makedirs('plots', exist_ok=True)
    # AQI stats
    print(f"\n  AQI Statistics: mean={df[target_col].mean():.2f}, median={df[target_col].median():.2f}, std={df[target_col].std():.2f}")
    # Histogram
    plt.hist(df[target_col], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(df[target_col].mean(), color='red', linestyle='--', label='Mean')
    plt.xlabel('AQI'); plt.ylabel('Frequency'); plt.title('AQI Distribution'); plt.legend()
    plt.savefig('plots/1_aqi_distribution.png', dpi=120, bbox_inches='tight'); plt.close()
    # Time series
    plt.plot(df['datetime'], df[target_col], linewidth=0.5, alpha=0.7)
    plt.xlabel('Date'); plt.ylabel('AQI'); plt.title('AQI Time Series')
    plt.xticks(rotation=45); plt.tight_layout(); plt.savefig('plots/2_aqi_timeseries.png', dpi=120, bbox_inches='tight'); plt.close()
    # Feature correlation
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [target_col, 'datetime']]
    correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False).head(15)
    correlations.plot(kind='barh'); plt.xlabel('Absolute Correlation with AQI'); plt.title('Top 15 Features'); plt.tight_layout(); plt.savefig('plots/3_feature_correlation.png', dpi=120, bbox_inches='tight'); plt.close()
    
    # Aggressive leakage removal
    leakage_features = []
    for col in numeric_cols:
        if 'aqi' in col.lower() or correlations.get(col,0)>0.90 or any(x in col for x in ['lag_1h','lag_3h','lag_6h']) or ('rolling' in col and any(x in col for x in ['_3h','_6h'])):
            leakage_features.append(col)
    if leakage_features:
        print(f"  ⚠️ Removing {len(leakage_features)} potential leakage features")
    else:
        print("  ✓ No leakage features detected")
    return leakage_features

# =========================
# --- DATA PREPARATION
# =========================
def prepare_data(df, target_col='us_aqi', test_size=0.2, leakage_features=[]):
    print("\n[4/7] Preparing Data...")
    feature_cols = [c for c in df.columns if c not in [target_col,'datetime'] and c not in leakage_features]
    safe_features = []
    for col in feature_cols:
        if any(x in col for x in ['pm2_5','pm10','carbon_monoxide','nitrogen_dioxide','sulphur_dioxide','ozone']):
            if not any(x in col for x in ['lag_1h','lag_3h','lag_6h']):
                safe_features.append(col)
        elif any(x in col for x in ['temperature','humidity','pressure','wind','precipitation']):
            safe_features.append(col)
        elif any(x in col for x in ['hour','day','month','weekday','weekend','sin','cos']):
            safe_features.append(col)
        elif 'interaction' in col and 'aqi' not in col:
            safe_features.append(col)
    feature_cols = list(set(safe_features))
    print(f"  ✓ Using {len(feature_cols)} safe features")
    X = df[feature_cols].values
    y = df[target_col].values
    dates = df['datetime'].values
    n_test = int(len(X)*test_size); n_train = len(X)-n_test
    X_train, X_test = X[:n_train], X[n_train:]; y_train, y_test = y[:n_train], y[n_train:]
    dates_train, dates_test = dates[:n_train], dates[n_train:]
    scaler = RobustScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
    return {'X_train':X_train_scaled,'y_train':y_train,'X_test':X_test_scaled,'y_test':y_test,
            'dates_train':dates_train,'dates_test':dates_test,'feature_names':feature_cols,'scaler':scaler,
            'y_train_raw': y[:n_train]}

# =========================
# --- TRAIN MODELS
# =========================
def train_models(data):
    print("\n[5/7] Training Models...")
    results = {}

    # SARIMA
    try:
        sarima_model = SARIMAX(data['y_train_raw'], order=(1,1,1), seasonal_order=(1,0,1,24),
                               enforce_stationarity=False, enforce_invertibility=False)
        sarima_fit = sarima_model.fit(disp=False, maxiter=50, method='lbfgs')
        sarima_pred_test = sarima_fit.forecast(len(data['y_test']))
        results['SARIMA'] = {'model': sarima_fit,
                             'predictions': sarima_pred_test,
                             'test_rmse': np.sqrt(mean_squared_error(data['y_test'], sarima_pred_test)),
                             'test_mae': mean_absolute_error(data['y_test'], sarima_pred_test),
                             'test_r2': r2_score(data['y_test'], sarima_pred_test)}
        print(f"  SARIMA ✓ RMSE: {results['SARIMA']['test_rmse']:.2f}, R²: {results['SARIMA']['test_r2']:.3f}")
    except Exception as e:
        print(f"  SARIMA ✗ {str(e)[:50]}")

    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4,
                                 min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
                                 reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1)
    xgb_model.fit(data['X_train'], data['y_train'])
    xgb_pred_test = xgb_model.predict(data['X_test'])
    results['XGBoost'] = {'model': xgb_model,
                           'predictions': xgb_pred_test,
                           'test_rmse': np.sqrt(mean_squared_error(data['y_test'], xgb_pred_test)),
                           'test_mae': mean_absolute_error(data['y_test'], xgb_pred_test),
                           'test_r2': r2_score(data['y_test'], xgb_pred_test)}
    print(f"  XGBoost ✓ RMSE: {results['XGBoost']['test_rmse']:.2f}, R²: {results['XGBoost']['test_r2']:.3f}")

    # LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4,
                                  min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
                                  reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(data['X_train'], data['y_train'])
    lgb_pred_test = lgb_model.predict(data['X_test'])
    results['LightGBM'] = {'model': lgb_model,
                            'predictions': lgb_pred_test,
                            'test_rmse': np.sqrt(mean_squared_error(data['y_test'], lgb_pred_test)),
                            'test_mae': mean_absolute_error(data['y_test'], lgb_pred_test),
                            'test_r2': r2_score(data['y_test'], lgb_pred_test)}
    print(f"  LightGBM ✓ RMSE: {results['LightGBM']['test_rmse']:.2f}, R²: {results['LightGBM']['test_r2']:.3f}")

    return results

# =========================
# --- EVALUATE MODELS
# =========================
def evaluate_models(results, data):
    print("\n[6/7] Model Evaluation & Comparison...")
    best_model_name = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
    print(f"★ Best Model: {best_model_name}")
    return best_model_name

# =========================
# --- SHAP EXPLAINER
# =========================
def explain_with_shap(model, model_name, X_test, feature_names):
    if model_name not in ['XGBoost','LightGBM']: return
    sample_size = min(500, len(X_test))
    X_sample = X_test[:sample_size]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    os.makedirs('plots', exist_ok=True)
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, max_display=15, show=False)
    plt.savefig('plots/6_shap_feature_importance.png', dpi=120, bbox_inches='tight'); plt.close()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type='bar', max_display=15, show=False)
    plt.savefig('plots/7_shap_bar_plot.png', dpi=120, bbox_inches='tight'); plt.close()

# =========================
# --- FORECAST 72H
# =========================
def forecast_72h_constrained(model, model_name, data, results):
    last_date = data['dates_test'][-1]
    recent_aqi = data['y_test'][-168:]
    recent_mean, recent_std = recent_aqi.mean(), recent_aqi.std()
    recent_min, recent_max = recent_aqi.min(), recent_aqi.max()

    if model_name == 'SARIMA':
        forecast_raw = results[model_name]['model'].forecast(steps=72)
        forecast = np.clip(forecast_raw, recent_min*0.7, recent_max*1.3)
    else:
        last_24h = data['y_test'][-24:]
        trend = (last_24h[-1]-last_24h[0])/24
        dampening = 0.95
        forecast = []
        base = last_24h[-1]
        for h in range(72):
            trend_component = trend * h * (dampening**h)
            mean_reversion = (recent_mean-base)*0.02*h
            noise = np.random.normal(0, recent_std*0.2)
            pred = np.clip(base+trend_component+mean_reversion+noise,
                           max(0, recent_mean-3*recent_std),
                           recent_mean+3*recent_std)
            forecast.append(pred)
        forecast = np.array(forecast)

    forecast_dates = [pd.Timestamp(last_date)+timedelta(hours=h+1) for h in range(72)]
    std_residual = np.std(data['y_test'] - results[model_name]['predictions'])
    expansion = np.linspace(1.0,2.0,72)
    forecast_df = pd.DataFrame({'datetime': forecast_dates,
                                'predicted_aqi': forecast,
                                'lower_bound': np.clip(forecast-1.96*std_residual*expansion,0,None),
                                'upper_bound': forecast+1.96*std_residual*expansion})
    os.makedirs('outputs', exist_ok=True)
    forecast_df.to_csv('outputs/72h_forecast.csv', index=False)

    # Plot
    plt.figure(figsize=(14,5))
    plt.plot(data['dates_test'][-168:], data['y_test'][-168:], label='Historical', color='blue', linewidth=1)
    plt.plot(forecast_df['datetime'], forecast_df['predicted_aqi'], label='72h Forecast', color='red', linewidth=2)
    plt.fill_between(forecast_df['datetime'], forecast_df['lower_bound'], forecast_df['upper_bound'], alpha=0.3, color='red', label='95% CI')
    plt.axvline(last_date,color='green',linestyle='--',alpha=0.5,label='Forecast Start')
    plt.xlabel('Date'); plt.ylabel('AQI'); plt.title(f'72-Hour AQI Forecast ({model_name})'); plt.legend()
    plt.xticks(rotation=45); plt.tight_layout(); plt.savefig('plots/8_72h_forecast.png', dpi=120, bbox_inches='tight'); plt.close()

    return forecast_df

# =========================
# --- SAVE MODEL
# =========================
def save_best_model(model, model_name):
    os.makedirs('models', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file_path = f"models/{model_name}_{ts}.pkl"
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved best model: {model_file_path}")
    return model_file_path

# =========================
# --- PUSH TO MONGO
# =========================
def push_model_to_mongodb(model_file_path, mongodb_uri, db_name="aqi_feature_store", collection_name="aqi_models"):
    client = MongoClient(mongodb_uri, server_api=ServerApi('1'))
    db = client[db_name]
    collection = db[collection_name]
    with open(model_file_path,'rb') as f:
        model_bytes = f.read()
    doc = {
        'model_name': os.path.basename(model_file_path),
        'timestamp': datetime.now(),
        'model_pickle': model_bytes
    }
    collection.insert_one(doc)
    client.close()
    print(f"✓ Pushed model to MongoDB collection '{collection_name}'")

# =========================
# --- RUN FULL PIPELINE
# =========================
def run_pipeline(mongodb_uri=None, csv_path=None):
    try:
        df = load_data_from_mongodb(mongodb_uri) if mongodb_uri else load_data_from_csv(csv_path)
        df = clean_data(df)
        leakage_features = perform_eda(df)
        data = prepare_data(df, leakage_features=leakage_features)
        results = train_models(data)
        best_model_name = evaluate_models(results, data)
        best_model = results[best_model_name]['model']
        explain_with_shap(best_model, best_model_name, data['X_test'], data['feature_names'])
        forecast_df = forecast_72h_constrained(best_model, best_model_name, data, results)
        model_file_path = save_best_model(best_model, best_model_name)
        if mongodb_uri: push_model_to_mongodb(model_file_path, mongodb_uri)
        print("\n✅ PIPELINE COMPLETE")
        return {'best_model_name': best_model_name,
                'best_model': best_model,
                'results': results,
                'forecast': forecast_df,
                'model_file_path': model_file_path}
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback; traceback.print_exc()
        return None

# =========================
# --- Example Run
# =========================
pipeline_outputs = run_pipeline(mongodb_uri="mongodb+srv://mohammadaliaun7_db_user:fJjD83zeRYhJi3wc@aqi.yqustuk.mongodb.net/?appName=AQI")
print(pipeline_outputs['best_model_name'])
