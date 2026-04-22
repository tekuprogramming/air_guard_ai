import pandas as pd
import time
from datetime import datetime
import os
import requests
import logging
import json
from pathlib import Path

config_path = Path(__file__).parent.parent / "config.json"

with open(config_path, "r") as f:
    config = json.load(f)

logging.basicConfig(level=logging.INFO)


class AirCollector:
    """
    Class responsible for collecting weather and air quality data
    from external APIs and saving them into a CSV file.
    """

    def __init__(self, city="Prague", country="CZ"):
        self.city = city
        self.country = country

        self.own_api_key = config.get("openweather_api_key")
        self.iqair_api_key = config.get("iqair_api_key")

        if not self.own_api_key or not self.iqair_api_key:
            raise ValueError("Missing API keys in environment variables")

        self.data_file = "historical_data.csv"

    def get_weather_data(self):
        """
        Fetch current weather data from OpenWeather API.
        Returns dictionary with processed weather features.
        """
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
        """
        Fetch air quality data from IQAir API.
        Returns AQI (Air Quality Index).
        """
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
            return {"aqi_us": None, "pm25": None, "pm10": None}

    def calculate_aqi_category(self, aqi):
        """
        Convert numeric AQI value into categorical label.
        """
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

    def _safe_float(self, value):
        # Helper: safely convert value to float or return None
        return float(value) if value is not None else None

    def _build_record(self, timestamp, weather, aqi):
        """
        Build a single structured dataset row for saving.
        Combines timestamp, weather, and AQI data.
        """
        return {
            "timestamp": timestamp.isoformat(),
            "date": timestamp.date().isoformat(),
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "is_weekend": int(timestamp.weekday() >= 5),

            "temperature": self._safe_float(weather["temperature"]),
            "humidity": self._safe_float(weather["humidity"]),
            "pressure": self._safe_float(weather["pressure"]),
            "wind_speed": self._safe_float(weather["wind_speed"]),
            "wind_deg": self._safe_float(weather["wind_deg"]),
            "weather_main": weather["weather_main"],
            "clouds": self._safe_float(weather["clouds"]),

            "aqi_us": self._safe_float(aqi),
            "aqi_category": self.calculate_aqi_category(aqi)
        }

    def _save_record(self, df):
        """
        Save dataframe to CSV file (append mode if file exists).
        """
        file_exists = os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0

        df.to_csv(
            self.data_file,
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False
        )

    def collect_and_save(self):
        """
        Main pipeline:
        1. Fetch weather data
        2. Fetch air quality data
        3. Validate and combine data
        4. Save to CSV
        """
        timestamp = datetime.now()

        weather = self.get_weather_data()
        air_quality = self.get_air_quality_data()

        if weather is None or air_quality is None:
            logging.warning("Skipping cycle due to missing data")
            return False

        aqi = self.validate_aqi(air_quality.get("aqi_us"))

        record = self._build_record(timestamp, weather, aqi)
        df = pd.DataFrame([record])

        self._save_record(df)

        print(f"Saved: AQI={aqi}, Category={record['aqi_category']}")
        return True

    def run_continue(self, interval_minutes=1):
        """
        Continuous data collection loop.
        Runs every X minutes until interrupted.
        """
        try:
            while True:
                self.collect_and_save()
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("Stopped safely")

    def validate_aqi(self, aqi):
        """
        Validate AQI value:
        - must be numeric
        - must be in realistic range (0–500)
        """
        if aqi is None:
            return None

        try:
            aqi = float(aqi)
        except (ValueError, TypeError):
            logging.warning(f"AQI is not a number: {aqi}")
            return None

        if 0 <= aqi <= 500:
            return aqi

        logging.warning(f"Suspicious AQI value: {aqi}")
        return None


if __name__ == "__main__":
    collector = AirCollector()
    collector.run_continue(interval_minutes=1)
