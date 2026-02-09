import pandas as pd

# -------------------------------
# Load raw AQI dataset
df = pd.read_csv("karachi_aqi_april2025_to_now.csv", parse_dates=["datetime"])

# -------------------------------
# 1️⃣ Time-based features
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["weekday"] = df["datetime"].dt.weekday

# -------------------------------
# 2️⃣ Change rates for key pollutants and AQI
relevant_pollutants = ["pm2_5","pm10","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide","ozone","us_aqi"]

for col in relevant_pollutants:
    df[f"{col}_change_rate"] = df[col].diff()

# -------------------------------
# 3️⃣ Select final relevant columns
final_columns = [
    "datetime",
    "location","latitude","longitude",
    "pm2_5","pm10","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide","ozone",
    "us_aqi","us_aqi_pm2_5","us_aqi_pm10",
    "pm2_5_change_rate","pm10_change_rate","carbon_monoxide_change_rate",
    "nitrogen_dioxide_change_rate","sulphur_dioxide_change_rate","ozone_change_rate",
    "us_aqi_change_rate",
    "hour","day","month","weekday"
]

# Keep only columns that exist in df (safety)
final_columns = [col for col in final_columns if col in df.columns]
df_final = df[final_columns]

# -------------------------------
# Save feature-engineered dataset
output_file = "karachi_aqi_features_cleaned.csv"
df_final.to_csv(output_file, index=False)

print(f"✅ Cleaned AQI features saved to '{output_file}'")
print(df_final.head())
print(f"Total rows: {len(df_final)}")
