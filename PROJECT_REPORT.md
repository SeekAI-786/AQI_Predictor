# AQI SeekAI — Project Report

**Real-Time Air Quality Index Prediction System for Karachi Using Machine Learning**

| Field | Details |
|:------|:--------|
| **Project** | AQI SeekAI — Real-Time AQI Monitoring & 72-Hour Forecast |
| **Domain** | Environmental Data Science / Machine Learning |
| **Location** | Karachi, Pakistan (24.86°N, 67.00°E) |
| **Live App** | [aqi-seek786.streamlit.app](https://aqi-seek786.streamlit.app) |
| **Repository** | [github.com/SeekAI-786/AQI_Predictor](https://github.com/SeekAI-786/AQI_Predictor) |

---

## 1. Introduction

Karachi, Pakistan's largest city (16M+ residents), consistently ranks among the world's most polluted cities but lacks accessible, real-time air quality forecasting. **AQI SeekAI** is an end-to-end ML system that autonomously collects hourly environmental data, engineers 186+ features, trains LightGBM models, and serves real-time AQI monitoring with 72-hour forecasting — all running serverlessly at **zero cost** using GitHub Actions, MongoDB Atlas, and Streamlit Cloud.

The US EPA AQI standard is used, covering six pollutants: PM2.5, PM10, O₃, NO₂, SO₂, and CO.

| AQI Range | Category | Health Impact |
|:----------|:---------|:--------------|
| 0–50 | Good | Satisfactory |
| 51–100 | Moderate | Risk for sensitive groups |
| 101–150 | Unhealthy for Sensitive Groups | Sensitive groups affected |
| 151–200 | Unhealthy | Everyone may be affected |
| 201–300 | Very Unhealthy | Health alert |
| 301–500 | Hazardous | Emergency conditions |

---

## 2. Objectives

1. Build an **automated hourly pipeline** fetching weather + air quality data from Open-Meteo APIs into MongoDB Atlas
2. Train ML models for **72-hour AQI forecasting** using a 3-band architecture
3. Run the entire system on **free-tier services** ($0/month)
4. Serve an **interactive dashboard** with real-time monitoring, forecasts, historical trends, EDA, and model metrics
5. Implement **automated retraining** every 12 hours

---

## 3. Tools & Technologies

| Category | Technology | Purpose |
|:---------|:-----------|:--------|
| ML Framework | LightGBM 4.x | Gradient-boosted tree regression |
| Scaling | scikit-learn (RobustScaler) | Outlier-robust feature scaling |
| Data | Pandas, NumPy | Processing & numerical computing |
| Database | MongoDB Atlas (Free Tier) | Cloud storage for features & models |
| Data Source | Open-Meteo API (Free) | 18 weather + 17 air quality variables |
| Frontend | Streamlit 1.54 + Plotly 5.x | Interactive web dashboard |
| CI/CD | GitHub Actions | Hourly pipeline + bi-daily retraining |
| Deployment | Streamlit Community Cloud | Free app hosting |

---

## 4. System Architecture

```
  Open-Meteo APIs ──► hourly_pipeline.py (every hour at :10)
                          │
                          ▼
                     MongoDB Atlas
                     ├─ aqi_feature_store.karachi_aqi_features (7,347+ docs)
                     └─ aqi_model_store.pearls_72h_models (5 docs)
                          │                    ▲
                          │                    │
                          ▼                    │
                     streamlit_app.py     retrain_pipeline.py
                     (Streamlit Cloud)    (every 12h at :30)
```

**Hourly Pipeline** (642 lines): Fetch last 3h → Dedup against MongoDB → Engineer 186+ features → Upload new records only

**Retrain Pipeline** (436 lines): Fetch ALL features → 80/20 temporal split → Train 3-band LightGBM → Save models to MongoDB → Evaluate per-horizon

---

## 5. Feature Engineering

The pipeline transforms 35 raw API variables into **186+ features** across 7 categories:

| Category | Count | Examples |
|:---------|:------|:--------|
| Weather derivatives | ~66 | Rolling means/std (6/12/24h), lags (12/24h) for temperature, humidity, wind, pressure, etc. |
| Atmospheric | ~25 | Aerosol, dust, UV, CO₂ rolling/lag features |
| Sub-AQI indices | ~36 | EPA sub-index lags & rolling means for each pollutant |
| Time (cyclical) | 9 | hour_sin/cos, day_of_week_sin/cos, month_sin/cos, is_weekend |
| Interactions | 7 | humidity×temp, radiation×aerosol, vpd×temp |
| AQI autoregressive | ~14 | AQI lags (1/3/6/12/24h), rolling mean/std, deltas |
| Base values | ~29 | Raw API values (temperature, sub-AQI indices, etc.) |

**Key decisions:**
- Raw pollutant concentrations **excluded** from features (they define AQI — circular dependency). EPA sub-indices used instead.
- Wind decomposed into u/v vector components (handles circular direction properly)
- API-provided `us_aqi` preferred over manual calculation (uses proper EPA rolling averages)

---

## 6. EDA Highlights

The dashboard's EDA page provides 8 interactive visualizations:

- **AQI Distribution** — Right-skewed histogram; majority of hours in "Moderate" to "USG" range
- **Category Breakdown** — Donut chart: "Moderate" + "USG" dominate (~60–80%)
- **Correlation Matrix** — PM2.5/PM10 strongly correlated; wind negatively correlated with AQI; temperature positively correlated with ozone
- **Day × Hour Heatmap** — AQI peaks at rush hours (7–10 AM, 6–9 PM), dips at 2–5 AM
- **Temperature vs AQI** — Higher temperatures → higher AQI (photochemical smog)
- **Dominant Pollutant** — PM2.5 dominates most hours in Karachi
- **Pollutant Trends** — Multi-line time series for user-selected pollutants
- **Hourly Patterns** — Diurnal pollution cycle revealed via bar chart

---

## 7. Model Design & Training

### 3-Band Architecture

Instead of training 72 separate models (impractical) or 1 monolithic model (suboptimal), we use **3 specialized bands**:

| Band | Horizons | Strategy |
|:-----|:---------|:---------|
| **Short** | t+1 to t+8h | Leverages strong autocorrelation |
| **Medium** | t+9, 12, 15, 18, 21, 24h | Learns diurnal patterns |
| **Long** | t+25, 30, 36, 42, 48, 54, 60, 66, 72h | Relies on weather regime signals |

Each sample: **186 features + 5 AR features + 1 horizon encoding (h/72) = 192 dimensions**

**LightGBM Config:** 600 estimators, lr=0.03, 63 leaves, subsample=0.8, colsample=0.8, RobustScaler

### Performance

| Band | R² | RMSE | MAE |
|:-----|:---|:-----|:----|
| **Short** (t+1–8h) | **0.9183** | 6.69 | 4.41 |
| **Medium** (t+9–24h) | 0.3721 | 18.55 | 13.44 |
| **Long** (t+25–72h) | -0.4336 | 27.55 | 20.41 |

**Per-horizon:** t+1h R²=0.98 → t+6h R²=0.89 → t+12h R²=0.58 → t+24h R²=0.06 → t+72h R²=-0.68

---

## 8. SHAP — Feature Importance

SHAP analysis (via `shap.TreeExplainer`) reveals how feature importance shifts across bands:

| Band | Top Features | Insight |
|:-----|:-------------|:--------|
| **Short** | us_aqi_lag_1h, us_aqi_rolling_mean_6h, us_aqi_pm2_5 | Autoregressive features dominate — persistence is strongest signal |
| **Medium** | hour_sin/cos, us_aqi_rolling_mean_24h, surface_pressure | Diurnal cycle features gain importance |
| **Long** | surface_pressure_24h, temperature_24h, vapour_pressure_deficit | Weather regime features take over as AR signal fades |

This shift validates the 3-band design — each band naturally learns different drivers.

---

## 9. Problems Faced & Solutions

### Problem 1: 72 Models — Storage Overflow
Training 72 individual models produced ~500 MB. MongoDB free tier caps at 512 MB with 16 MB/doc limit.
**→ Solution:** 3-band architecture with horizon encoding. Reduced to ~15 MB (3 models + scaler).

### Problem 2: Forecast Accuracy Degradation
R² drops from 0.98 (t+1h) to -0.68 (t+72h). AQI is inherently chaotic at multi-day timescales.
**→ Solution:** 3-band specialization. Dashboard uses different colors per band to communicate decreasing confidence.

### Problem 3: Environment Variable Empty String Bug (Critical)
`os.getenv("KEY", "default")` returns `""` (not default) when GitHub Actions sets env var from a missing secret. Pipelines silently wrote to wrong databases.
**→ Solution:** `os.getenv("KEY") or "default"` — Python's `or` treats `""` as falsy.

### Problem 4: Retrain Timing & Freshness
Retrain ran at `:00` before hourly at `:10`, training on stale data.
**→ Solution:** Moved retrain to `:30`. Added freshness check (latest record <6h old) with 3 retries.

### Problem 5: Workflow Secret Name Mismatch
Retrain used different env var names than hourly. Only hourly's secrets were configured.
**→ Solution:** Aligned both workflows to same env var names with fallback chains.

### Problem 6: Plotly Missing on Streamlit Cloud
Removing version pins caused resolver to downgrade streamlit, conflicting with altair.
**→ Solution:** Added minimum version pins: `streamlit>=1.30.0`, `plotly>=5.18.0`.

### Problem 7: Legacy MongoDB Document KeyError
One doc used `model_name` instead of `band` field.
**→ Solution:** `if "band" not in doc: continue`

### Problem 8: Year Display Off by One
MongoDB dates stored as 2026 instead of 2025.
**→ Solution:** `dt.replace(year=dt.year - 1)` on load.

### Problem 9: Invisible Sidebar Text
Dark sidebar made radio labels invisible.
**→ Solution:** CSS: `color: #ffffff !important; font-weight: 700`.

### Problem 10: Weather Card Truncation
`st.metric()` truncated long values.
**→ Solution:** Custom HTML cards via `st.markdown(unsafe_allow_html=True)`.

---

## 10. Dashboard Output

The Streamlit app (914 lines, 5 pages):

| Page | Content |
|:-----|:--------|
| **Dashboard** | AQI gauge, 72h forecast chart, milestones, weather panel (8 cards), pollutant breakdown |
| **72h Forecast** | Full forecast chart with band colors, band stats, detailed table, distribution histogram |
| **Historical** | Date-filtered time series, summary stats, hourly patterns, pollutant trends, raw data |
| **EDA & Analytics** | Distribution, category breakdown, correlation matrix, heatmap, scatter plots |
| **Model Info** | Band metrics, comparison chart, training details, 186 feature list, architecture table |

---

## 11. Deployment & Cost

| Component | Schedule | Cost |
|:----------|:---------|:-----|
| Hourly Pipeline (GitHub Actions) | `'10 * * * *'` | Free |
| Retrain Pipeline (GitHub Actions) | `'30 */12 * * *'` | Free |
| MongoDB Atlas | 7,347+ docs | Free |
| Streamlit Community Cloud | 1 app | Free |
| Open-Meteo API | ~720 calls/month | Free |
| **Total** | | **$0/month** |

---

## 12. Key Findings

1. **Short-term AQI forecasting works well** — R²=0.92 for 1–8h, actionable for health advisories
2. **Accuracy decays exponentially with horizon** — 0.98 (t+1h) → -0.68 (t+72h)
3. **AR features dominate short-term**, weather features dominate long-term — validates 3-band design
4. **PM2.5 is Karachi's dominant pollutant** across the majority of recorded hours
5. **Serverless ML is production-viable** — 7,000+ data points, 10+ months, zero cost
6. **`os.getenv()` empty string vs unset** is a critical CI/CD pitfall

---

## 13. Conclusion & Future Work

AQI SeekAI demonstrates that a complete air quality prediction system can run entirely on free-tier services. The system collects 35 variables hourly, engineers 186+ features, trains 3-band LightGBM models (R²=0.92 short-term), and serves an interactive dashboard — fully automated and serverless.

**Future Work:** Ensemble models (LightGBM + LSTM), probabilistic forecasting, multi-city expansion, SHAP dashboard integration, AQI alert notifications, satellite data integration.

---

## References

1. US EPA — AQI Technical Assistance Document
2. Open-Meteo — Weather & Air Quality API (https://open-meteo.com)
3. Ke et al. (2017) — "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (NeurIPS)
4. Lundberg & Lee (2017) — "A Unified Approach to Interpreting Model Predictions" (SHAP)
5. MongoDB Atlas & Streamlit Documentation
