import requests
import pandas as pd
from datetime import datetime
import calendar

# Karachi coordinates
latitude = 24.8607
longitude = 67.0011

# Hourly AQI/pollutant variables
hourly_vars = [
    "pm10", "pm2_5", "carbon_monoxide",
    "nitrogen_dioxide", "sulphur_dioxide", "ozone",
    "aerosol_optical_depth", "dust", "uv_index",
    "us_aqi", "us_aqi_pm2_5", "us_aqi_pm10"
]

# -------------------------------
def fetch_aqi_month(latitude, longitude, year, month):
    """Fetch AQI data for a single month."""
    start = f"{year}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    end = f"{year}-{month:02d}-{last_day:02d}"

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(hourly_vars),
        "timezone": "auto"
    }

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"⚠️  No data for {year}-{month:02d}, skipping...")
        return pd.DataFrame()

    data = resp.json()
    if "hourly" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["hourly"])
    df["datetime"] = pd.to_datetime(df["time"])
    df["latitude"] = latitude
    df["longitude"] = longitude
    df["location"] = "Karachi"
    df = df.drop(columns=["time"])
    return df

# -------------------------------
# Loop from April 2025 to today
start_year = 2025
start_month = 4
end_date = datetime.utcnow()
end_year = end_date.year
end_month = end_date.month

all_data = []
current_year = start_year
current_month = start_month

while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
    print(f"Fetching AQI for {current_year}-{current_month:02d}...")
    df_month = fetch_aqi_month(latitude, longitude, current_year, current_month)
    if not df_month.empty:
        all_data.append(df_month)

    # Move to next month
    current_month += 1
    if current_month > 12:
        current_month = 1
        current_year += 1

# -------------------------------
# Combine all months
if all_data:
    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.sort_values("datetime").reset_index(drop=True)

    # Add derived change rates
    for col in ["pm2_5","pm10","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide","ozone","us_aqi"]:
        if col in df_all.columns:
            df_all[f"{col}_change_rate"] = df_all[col].diff()
else:
    df_all = pd.DataFrame()
    print("⚠️  No AQI data fetched!")

# -------------------------------
# Save CSV
output_file = "karachi_aqi_april2025_to_now.csv"
df_all.to_csv(output_file, index=False)
print(f"\n✅ Saved AQI data to '{output_file}'")
print(df_all.head())
print(f"Total rows: {len(df_all)}")
