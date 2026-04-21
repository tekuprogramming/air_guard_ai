import pandas as pd
import time
from datetime import datetime
import os
import requests
import logging
logging.basicConfig(level=logging.INFO)

class AirCollector:
    def __init__(self, city="Prague", country="CZ"):
        self.city = city
        self.country = country
        self.own_api_key = os.getenv("OPENWEATHER_API_KEY")
        self.iqair_api_key = os.getenv("IQAIR_API_KEY")
        if not self.own_api_key or not self.iqair_api_key:
            raise ValueError("Missing API keys in environment variables")
        self.data_file = "historical_data.csv"

    def get_weather_data(self):
        url = f"http://api.openweathermap.org/data/2.5/weather?q={self.city},{self.country}&appid={self.own_api_key}&units=metric"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            main = data.get("main", {})
            wind = data.get("wind", {})
            weather = data.get("weather", [{}])
            clouds = data.get("clouds", {})
            weather_data = {
                "temperature": float(main.get("temp", 0)),
                "humidity": float(main.get("humidity", 0)),
                "pressure": float(main.get("pressure", 0)),
                "wind_speed": float(wind.get("speed", 0)),
                "wind_deg": float(wind.get("deg", 0)),
                "weather_main": str(weather[0].get("main", "unknown")),
                "clouds": float(clouds.get("all", 0))
            }
            return weather_data
        except Exception as e:
            print(f"Error getting weather data: {e}")
            return None

    def get_air_quality_data(self):
        try:
            url = f"https://api.airvisual.com/v2/nearest_city?key={self.iqair_api_key}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "success":
                pollution = data["data"]["current"]["pollution"]

                aqi_us = pollution.get("aqius")

                logging.info("AQI US=%s", aqi_us)

                return {
                    "aqi_us": float(aqi_us) if aqi_us is not None else None
                }
                
            return {"aqi_us": None, "pm25": None, "pm10": None}
        
        except Exception as e:
            logging.error("IQAir error: %s", e)
            return {"aqi_us": None}

    def calculate_aqi_category(self, aqi):
        if aqi is None:
            return "unknown"
        elif aqi <= 50:
            return "good"
        elif aqi <= 100:
            return "moderate"
        elif aqi <= 150:
            return "unhealthy_sensitive"
        elif aqi <= 200:
            return "unhealthy"
        else:
            return "hazardous"

    def collect_and_save(self):
        timestamp = datetime.now()
        weather = self.get_weather_data()
        air_quality = self.get_air_quality_data()
        
        if weather is None or air_quality is None:
            logging.warning("Skipping cycle due to missing data")
            return False

        aqi = air_quality.get("aqi_us")

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

            "aqi_us": float(aqi) if aqi is not None else None,
            "aqi_category": self.calculate_aqi_category(aqi)
        }

        df = pd.DataFrame([record])

        if not os.path.exists(self.data_file) or os.path.getsize(self.data_file) == 0:
            df.to_csv(self.data_file, index=False)
            print(f"New file created: {self.data_file}")
        else:
            df.to_csv(self.data_file, mode="a", header=False, index=False)
            print(f"Data appended: {self.data_file}")

        print(f"Saved: AQI={aqi}, Category={record['aqi_category']}")
        return True

    def run_continue(self, interval_minutes=1):
        try:
            while True:
                self.collect_and_save()
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("Stopped safely")

if __name__ == "__main__":
    collector = AirCollector()
    collector.run_continue(interval_minutes=1)

