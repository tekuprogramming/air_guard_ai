import requests
import pandas as pd
from datetime import datetime, timedelta

LAT = 50.0755
LON = 14.4378

start = datetime(2025, 1, 1)
end = datetime(2026, 1, 1)

all_data = []

current = start

while current < end:
    next_day = current + timedelta(days=1)

    date_str = current.strftime("%Y-%m-%d")

    weather_url = "https://archive-api.open-meteo.com/v1/archive"

    weather_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,cloud_cover",
        "timezone": "Europe/Prague"
    }

    w = requests.get(weather_url, params=weather_params).json()

    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    air_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "pm10,pm2_5",
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "Europe/Prague"
    }

    a = requests.get(air_url, params=air_params).json()

    if "hourly" in w and "hourly" in a:

        times = w["hourly"]["time"]

        for i in range(len(times)):
            all_data.append({
                "timestamp": times[i],

                "temperature": w["hourly"]["temperature_2m"][i],
                "humidity": w["hourly"]["relative_humidity_2m"][i],
                "pressure": w["hourly"]["pressure_msl"][i],
                "wind_speed": w["hourly"]["wind_speed_10m"][i],
                "clouds": w["hourly"]["cloud_cover"][i],

                "pm10": a["hourly"]["pm10"][i],
                "pm25": a["hourly"]["pm2_5"][i],
            })

    print(f"Downloaded: {date_str}")

    current = next_day

df = pd.DataFrame(all_data)

if df.empty:
    print("No data downloaded!")
    exit()

df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.weekday
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["month"] = df["timestamp"].dt.month

df["aqi_risk"] = (df["pm25"] > 25).astype(int)

df.to_csv("historical_data.csv", index=False)

print("DONE:", len(df), "rows saved")