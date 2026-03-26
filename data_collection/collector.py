import pandas as pd
import time
from datetime import datetime
import os
import requests
import numpy as np


class AirCollector:
    def __init__(self, city="Prague", country="CZ"):
        self.city = city
        self.country = country
        self.own_api_key = "3eaab97ed540e25e7b261d686d5dfc42"
        self.iqair_api_key = "773eb493-2ae7-4825-a6d3-02cbe54b09f0"
        self.data_file = "historical_data.csv"

    def get_weather_data(self):
        url = f"http://api.openweathermap.org/data/2.5/weather?q={self.city},{self.country}&appid={self.own_api_key}&units=metric"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            weather_data = {
                "temperature": float(data["main"]["temp"]),
                "humidity": float(data["main"]["humidity"]),
                "pressure": float(data["main"]["pressure"]),
                "wind_speed": float(data["wind"]["speed"]),
                "wind_deg": float(data["wind"].get("deg", 0)),
                "weather_main": str(data["weather"][0]["main"]),
                "clouds": float(data["clouds"]["all"])
            }
            return weather_data
        except Exception as e:
            print(f"Error getting weather data: {e}")
            return None

    def get_air_quality_data(self):
        try:
            url = f"https://api.airvisual.com/v2/nearest_city?key={self.iqair_api_key}"
            response = requests.get(url, timeout=15)
            data = response.json()

            if data.get("status") == "success":
                pollution_data = data["data"]["current"]["pollution"]

                pm25 = pollution_data.get("aqius")
                if pm25 is not None:
                    pm25 = float(pm25)

                pm10 = pollution_data.get("aqipc")
                if pm10 is not None:
                    pm10 = float(pm10)

                if pm10 is None and pm25 is not None:
                    ratio = np.random.uniform(1.4, 1.8)
                    pm10 = pm25 * ratio
                    pm10 = round(pm10, 1)
                    # PM10 calculated from PM2.5: {pm25} × {ratio:.2f} = {pm10}

                print(f"IQAir: PM2.5={pm25}, PM10={pm10}")
                return {"pm25": pm25, "pm10": pm10}
            else:
                error_msg = data.get("data", {}).get("message", "Unknown")
                print(f"IQAir error: {error_msg}")
                return {"pm25": None, "pm10": None}

        except Exception as e:
            print(f"IQAir exception: {e}")
            return {"pm25": None, "pm10": None}

    def calculate_aqi_category(self, pm25):
        if pm25 is None:
            return "unknown"
        elif pm25 <= 12:
            return "good"
        elif pm25 <= 35:
            return "moderate"
        elif pm25 <= 55:
            return "unhealthy_sensitive"
        elif pm25 <= 150:
            return "unhealthy"
        else:
            return "hazardous"

    def collect_and_save(self):
        timestamp = datetime.now()
        weather = self.get_weather_data()
        air_quality = self.get_air_quality_data()

        if weather and air_quality:
            pm25 = air_quality.get("pm25")
            pm10 = air_quality.get("pm10")

            if pm25 is None:
                print(f"No PM2.5 data, skipping...")
                return False
                
            if pm10 is None:
                ratio = np.random.uniform(1.4, 1.8)
                pm10 = pm25 * ratio
                pm10 = round(pm10, 1)
                print(f"PM10 calculated (fallback): {pm25} × {ratio:.2f} = {pm10}")

            record = {
                "timestamp": str(timestamp.isoformat()),
                "date": str(timestamp.date().isoformat()),
                "hour": int(timestamp.hour),
                "day_of_week": int(timestamp.weekday()),
                "is_weekend": int(1 if timestamp.weekday() >= 5 else 0),
                "temperature": float(weather["temperature"]),
                "humidity": float(weather["humidity"]),
                "pressure": float(weather["pressure"]),
                "wind_speed": float(weather["wind_speed"]),
                "wind_deg": float(weather["wind_deg"]),
                "weather_main": str(weather["weather_main"]),
                "clouds": float(weather["clouds"]),
                "pm25": float(pm25),
                "pm10": float(pm10),
                "aqi_category": str(self.calculate_aqi_category(pm25))
            }

            df = pd.DataFrame([record])

            if not os.path.exists(self.data_file) or os.path.getsize(self.data_file) == 0:
                df.to_csv(self.data_file, index=False)
                print(f"New file created: {self.data_file}")
            else:
                df.to_csv(self.data_file, mode="a", header=False, index=False)
                print(f"Data appended: {self.data_file}")

            print(f"Saved: PM2.5={pm25}, PM10={pm10}, Category={record['aqi_category']}")
            return True
        else:
            print(f"Missing data, skipping...")
            return False

    def run_continue(self, interval_minutes=1):
        print("=" * 60)
        print(f"Air Quality Collector - Prague")
        print(f"Weather: OpenWeatherMap")
        print(f"Air Quality: IQAir (with PM10 calculation)")
        print(f"Interval: {interval_minutes} minutes")
        print("=" * 60)

        while True:
            success = self.collect_and_save()
            if success:
                print(f"Next collection in {interval_minutes} minutes")
                time.sleep(interval_minutes * 60)
            else:
                print(f"Retrying in 5 minutes")
                time.sleep(300)


if __name__ == "__main__":
    collector = AirCollector()
    collector.run_continue(interval_minutes=1)

