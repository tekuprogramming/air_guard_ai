from visualizer import AirQualityVisualizer
import pandas as pd
import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from predictor import AirQualityPredictor
import requests
import numpy as np

# adding import path to project's core folder
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# a try of collecting data with module import
try:
    from data_collection.collector import AirCollector
    COLLECTOR_AVAILABLE = True
except ImportError:
    COLLECTOR_AVAILABLE = False
    print("Data collection module not found. Demo mode will be used")


class AirGuardApp:
    def __init__(self):
        print("\n" + "=" * 60)
        print("AirGuard Application Started")
        print("=" * 60)

        # loading models
        model_path = PROJECT_ROOT / "air_module_training" / "air_quality_model.pkl"
        self.predictor = AirQualityPredictor(str(model_path))
        self.visualizer = AirQualityVisualizer()

        # loading historical data
        self.historical_data = self.load_historical_data()
        self.current_data = None
        self.city = "Prague"
        self.country = "CZ"
        self.api_key = self.load_api_key()
        print("\nThe app was initialized")
        print(f"\nFolder for graphs: {self.visualizer.output_dir}")
        print("=" * 60)

    def load_api_key(self):
        config_parse = PROJECT_ROOT / "config.json"
        if config_parse.exists():
            try:
                with open(str(config_parse), "r") as f:
                    config = json.load(f)
                return config.get("openweather_api_key", "")
            except:
                pass

        # if file doesn't exist, try environmental variable
        return os.environ.get("OPENWEATHER_API_KEY", "")

    def load_historical_data(self):
        data_path = PROJECT_ROOT / "data_collection" / "historical_data.csv"
        if data_path.exists():
            try:
                df = pd.read_csv(data_path)
                print(f"Loaded {len(df)} records of historical data")
                return df
            except Exception as e:
                print("Error loading historical data: ", e)
        print("Historical data not found. Creating demo data")
        return self.create_demo_data()

    def create_demo_data(self):
        # creating 168 records (week)
        dates = [datetime.now() - timedelta(hours = i) for i in range(168, 0, -1)]

        # generating data with realistic patterns
        np.random.seed(42)
        data = []
        for dt in dates:
            hour = dt.hour
            is_weekend = 1 if dt.weekday() >= 5 else 0

            # morning and evening rush in week days
            rush_factor = 1.0
            if not is_weekend and (7 <= hour <= 9 or 16 <= hour <= 19):
                rush_factor = 1.5

            # pollution higher in winter
            month = dt.month
            season_factor = 1.3 if month in [11, 12, 1, 2] else 1.0
            base_pm25 = 20 * season_factor * rush_factor
            base_pm10 = 30 * season_factor * rush_factor
            record = {"timestamp": dt.isoformat(),
                      "date": dt.date().isoformat(),
                      "hour": hour,
                      "day_of_week": dt.weekday(),
                      "is_weekend": is_weekend,
                      "temperature": 10 + 5 * np.sin(hour * np.pi / 12) + np.random.normal(0, 2),
                      "humidity": 70 + 10 * np.random.randn(),
                      "treasure": 1013 + 5 * np.random.randn(),
                      "wind_speed": 3 + np.random.exponential(1),
                      "pm25": max(0, base_pm25 + np.random.normal(0, 5)),
                      "pm10":  max(0, base_pm10 + np.random.normal(0, 8)),
                      "aqi_category": self.calculate_aqi_category(base_pm25)}
            data.append(record)
        df = pd.DataFrame(data)
        print(f"Created {len(df)} records of demo data")
        return df

    # calculates the category of air quality by PM2.5
    def calculate_aqi_category(self, pm25):
        if pm25 <= 12:
            return "good"
        if pm25 <= 35:
            return "moderate"
        if pm25 <= 55:
            return "unhealthy_sensitive"
        if pm25 <= 150:
            return "unhealthy"
        else:
            return "hazardous"

    # getting current data from API
    def fetch_current_data(self):
        if not self.api_key:
            print("API key not found. Using most recent historic data")
            return self.get_last_historical_data()
        try:
            # getting weather
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={self.city},{self.country}&appid={self.api_key}&units=metric"
            weather_response = requests.get(weather_url, timeout = 10)
            weather_data = weather_response.json()
            if weather_response.status_code != 200:
                print(f"API weather error: {weather_data.get('message', 'Unknown error')}")
                return self.get_last_historical_data()
            pm25 = None
            pm10 = None
            try:
                air_url = f"https://api.openaq.org/v2/latest?city={self.city}&parameter=pm25&parameter=pm10"
                air_response = requests.get(air_url, timeout = 10)
                air_data = air_response.json()
                for result in air_data.get("results", []):
                    for measurement in result.get("measurements", []):
                        if measurement["parameter"] == "pm25":
                            pm25 = measurement["value"]
                        elif measurement["parameter"] == "pm10":
                            pm10 = measurement["value"]
            except:
                print("Couldn't get air quality data")
