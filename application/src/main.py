# Main application file for AirGuard
# Handles data loading, API communication, prediction, and user interaction

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

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from data_collection.collector import AirCollector

    COLLECTOR_AVAILABLE = True
except ImportError:
    COLLECTOR_AVAILABLE = False
    print("Data collection module not found. Demo mode will be used")


class AirGuardApp:
    # Main application class handling all logic
    def __init__(self):
        print("\n" + "=" * 60)
        print("AirGuard Application Started")
        print("=" * 60)

        model_path = PROJECT_ROOT / "module_training" / "trained_model.pkl"
        print(model_path)
        print(model_path.exists())
        self.predictor = AirQualityPredictor(str(model_path))
        self.visualizer = AirQualityVisualizer()

        self.historical_data = self.load_historical_data()
        self.current_data = None
        self.city = "Prague"
        self.country = "CZ"
        self.api_key = self.load_api_key()
        print("\nThe app was initialized")
        print(f"\nFolder for graphs: {self.visualizer.output_dir}")
        print("=" * 60)

    def load_api_key(self):
        # Load API key from config.json or environment variable
        config_path = PROJECT_ROOT / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                return config.get("openweather_api_key", "")
            except:
                pass

        return os.environ.get("OPENWEATHER_API_KEY", "")

    def load_historical_data(self):
        # Load historical data from CSV or create demo data if not available
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

    def _get_rush_factor(self, dt):
        # Increase pollution during rush hours on weekdays
        hour = dt.hour
        is_weekend = dt.weekday() >= 5
        if not is_weekend and (7 <= hour <= 9 or 16 <= hour <= 19):
            return 1.5
        return 1.0

    def _get_season_factor(self, dt):
        # Increase pollution during winter months
        return 1.3 if dt.month in [11, 12, 1, 2] else 1.0

    def create_demo_data(self):
        # Generate synthetic historical data for demo/testing
        dates = [datetime.now() - timedelta(hours=i) for i in range(168, 0, -1)]
        np.random.seed(42)

        data = []
        for dt in dates:
            rush = self._get_rush_factor(dt)
            season = self._get_season_factor(dt)

            base_pm25 = 20 * season * rush
            base_pm10 = 30 * season * rush

            data.append({
                "timestamp": dt.isoformat(),
                "date": dt.date().isoformat(),
                "hour": dt.hour,
                "day_of_week": dt.weekday(),
                "is_weekend": dt.weekday() >= 5,

                "temperature": 10 + 5 * np.sin(dt.hour * np.pi / 12) + np.random.normal(0, 2),
                "humidity": 70 + 10 * np.random.randn(),
                "pressure": 1013 + 5 * np.random.randn(),
                "wind_speed": 3 + np.random.exponential(1),
                "clouds": 50 + 20 * np.random.randn(),

                "aqi_category": self.calculate_aqi_category(base_pm25)
            })

        print(f"Created {len(data)} records of demo data")
        return pd.DataFrame(data)

    def calculate_aqi_category(self, pm25):
        # Determine AQI category based on PM2.5 value
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

    def get_weather(self):
        # Fetch current weather data from OpenWeather API
        url = f"http://api.openweathermap.org/data/2.5/weather?q={self.city},{self.country}&appid={self.api_key}&units=metric"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            raise Exception(r.json().get("message", "Weather API error"))
        return r.json()

    def get_air_quality(self):
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"

        params = {
            "latitude": 50.0755,
            "longitude": 14.4378,
            "hourly": "pm10,pm2_5"
        }

        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        pm25 = data["hourly"]["pm2_5"][-1]
        pm10 = data["hourly"]["pm10"][-1]

        return pm25, pm10

    def fetch_current_data(self):
        # Fetch current data from APIs or fallback to historical data
        if not self.api_key:
            print("API key not found. Using historic data")
            return self.get_last_historical_data()

        try:
            weather = self.get_weather()
            pm25, pm10 = self.get_air_quality()

            now = datetime.now()

            self.current_data = {
                "timestamp": now.isoformat(),
                "date": now.date().isoformat(),
                "hour": now.hour,
                "day_of_week": now.weekday(),
                "is_weekend": 1 if now.weekday() >= 5 else 0,

                "temperature": weather["main"]["temp"],
                "feels_like": weather["main"]["feels_like"],
                "humidity": weather["main"]["humidity"],
                "pressure": weather["main"]["pressure"],
                "wind_speed": weather["wind"]["speed"],
                "wind_deg": weather["wind"].get("deg", 0),
                "weather_main": weather["weather"][0]["main"],
                "weather_desc": weather["weather"][0]["description"],
                "clouds": weather["clouds"]["all"],

                "pm25": pm25,
                "pm10": pm10,
                "no2": 0,
                "aqi_category": self.calculate_aqi_category(pm25) if pm25 else "unknown"
            }

            print("Successfully received data from API")
            return self.current_data

        except Exception as e:
            print(f"Error: {e}")
            return self.get_last_historical_data()

    def get_last_historical_data(self):
        # Return latest available historical record or default values
        if self.historical_data is not None and len(self.historical_data) > 0:
            last_record = self.historical_data.iloc[-1].to_dict()
            last_record["timestamp"] = datetime.now().isoformat()
            self.current_data = last_record
            print("Using the last historical data")
            return self.current_data
        else:
            self.current_data = {
                "timestamp": datetime.now().isoformat(),
                "temperature": 12,
                "humidity": 70,
                "pressure": 1013,
                "wind_speed": 3.5,
                "pm25": 25,
                "pm10": 40,
                "aqi_category": "moderate",
                "day_of_week": datetime.now().weekday(),
                "is_weekend": 1 if datetime.now().weekday() >= 5 else 0,
                "weather_main": "Clear",
                "clouds": 50
            }
            return self.current_data

    def make_prediction(self):
        # Prepare features and get prediction from ML model
        if self.current_data is None:
            self.fetch_current_data()

        features = {
            "temperature": self.current_data.get("temperature", 0),
            "humidity": self.current_data.get("humidity", 0),
            "pressure": self.current_data.get("pressure", 1013),
            "wind_speed": self.current_data.get("wind_speed", 0),
            "clouds": self.current_data.get("clouds", 0),
            "hour_sin": np.sin(2 * np.pi * self.current_data.get("hour", 0) / 24),
            "hour_cos": np.cos(2 * np.pi * self.current_data.get("hour", 0) / 24),
            "day_sin": np.sin(2 * np.pi * self.current_data.get("day_of_week", 0) / 7),
            "day_cos": np.cos(2 * np.pi * self.current_data.get("day_of_week", 0) / 7),
            "weather_main_encoded": hash(self.current_data.get("weather_main", "Clear")) % 10,
            "is_weekend": self.current_data.get("is_weekend", 0),
            "pm25": self.current_data.get("pm25", 0),
            "pm10": self.current_data.get("pm10", 0)
        }

        prediction = self.predictor.predict(features)
        return prediction

    def encode_weather(self, weather_main):
        # Encode weather condition into numerical value for model
        try:
            return self.predictor.label_encoder.transform([weather_main])[0]
        except Exception:
            return 0

    def is_czech_holiday(self, date):
        # Check if given date is a Czech public holiday
        czech_holidays = [
            "2026-01-01", "2026-04-01", "2026-05-01", "2026-05-08",
            "2026-07-05", "2026-07-06", "2026-09-28", "2026-10-28",
            "2026-12-24", "2026-12-25", "2026-12-26"
        ]
        return 1 if date.strftime("%Y-%m-%d") in czech_holidays else 0

    def is_rush_hour(self, dt):
        # Check if current time is rush hour
        hour = dt.hour
        is_week_day = dt.weekday() < 5
        return 1 if (is_week_day and (7 <= hour <= 9 or 15 <= hour <= 17)) else 0

    def get_recommendation(self, category):
        # Return recommendation based on air quality category
        recommendations = {
            "good": {
                "en": "perfect air quality\n"
                      "- you can organize classes outside\n"
                      "- ventilate the classrooms\n"
                      "- suitable for sports"
            },
            "moderate": {
                "en": "good air quality\n"
                      "- classes outside are allowed\n"
                      "- people with asthma should be careful"
            },
            "unhealthy_sensitive": {
                "en": "satisfying air quality\n"
                      "- limit long activities\n"
                      "- sensitive people should stay inside"
            },
            "unhealthy": {
                "en": "bad air quality\n"
                      "- cancel classes outside\n"
                      "- remain in classrooms"
            },
            "hazardous": {
                "en": "dangerous air quality\n"
                      "- everyone should remain inside\n"
                      "- don't ventilate"
            }
        }
        return recommendations.get(category, recommendations["moderate"])

    def show_menu(self):
        # Display main menu
        print("\n" + "=" * 60)
        print("Main menu")
        print("=" * 60)
        print("1. Current air quality prediction")
        print("2. 24h trend graph")
        print("3. Weekly statistics")
        print("4. Full dashboard")
        print("5. Update data")
        print("6. About application")
        print("7. Exit")
        print("-" * 60)

    def get_header(self, title):
        # Format section header
        return "\n" + "=" * 60 + f"\n{title}\n" + "=" * 60

    def print_current_data(self):
        # Print current weather and pollution data
        print(f"\nLocation: {self.city}, {self.country}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        print("\nWeather now:")
        print(f"Temperature: {self.current_data.get('temperature', 'N/A')}°C")
        print(f"Humidity: {self.current_data.get('humidity', 'N/A')}%")
        print(f"Wind: {self.current_data.get('wind_speed', 'N/A')}m/s")
        print(f"Pressure: {self.current_data.get('pressure', 'N/A')}hPa")

        print("\nPollution:")
        print(f"PM2.5: {self.current_data.get('pm25', 'N/A')}μg/m³")
        print(f"PM10: {self.current_data.get('pm10', 'N/A')}μg/m³")

    def print_prediction(self, prediction):
        # Print model prediction and probabilities
        print("\nModel prediction:")
        print(f"Category: {prediction['category'].upper()}")
        print(f"Confidence: {prediction['confidence'] * 100:.1f}%")

        rec = self.get_recommendation(prediction["category"])
        print("\nRecommendation:")
        print(rec["en"])

        if "probabilities" in prediction:
            print("\nProbability distribution")
            for cat, prob in prediction["probabilities"].items():
                bar = "[]" * int(prob * 20)
                print(f"{cat:20} {bar} {prob * 100:.1f}%")

    def show_current_prediction(self):
        # Display current data, prediction and graph
        print(self.get_header("Current air quality prediction"))

        if self.current_data is None:
            self.fetch_current_data()

        prediction = self.make_prediction()

        self.print_current_data()
        self.print_prediction(prediction)

        self.visualizer.plot_current_metrics(self.current_data)

    def show_trend_chart(self):
        # Display 24-hour trend chart
        print("\n" + "=" * 60)
        print("24h trend")
        print("=" * 60)
        if self.historical_data is not None and len(self.historical_data) > 0:
            recent_data = self.historical_data.tail(24)
            self.visualizer.plot_24h_trend(recent_data)
            print("Trend graph was created")
        else:
            print("No historical data to make a graph")

    def show_weekly_stats(self):
        # Display weekly statistics
        print("\n" + "=" * 60)
        print("Weekly statistics")
        print("=" * 60)
        if self.historical_data is not None and len(self.historical_data) >= 168:
            week_data = self.historical_data.tail(168)
            self.visualizer.plot_weekly_stats(week_data)
            print("Weekly statistics was created")
        else:
            print("Not enough data for weekly statistics")
            print(
                f"There are {len(self.historical_data) if self.historical_data is not None else 0} records. 168 needed")

    def show_dashboard(self):
        # Display full dashboard with all metrics
        print("\n" + "=" * 60)
        print("Full dashboard")
        print("=" * 60)
        if self.current_data is None:
            self.fetch_current_data()
        if self.historical_data is not None and len(self.historical_data) > 0:
            self.visualizer.create_dashboard(self.current_data, self.historical_data)
            print("Dashboard was created")
        else:
            print("No data to make a dashboard")

    def update_data(self):
        # Fetch new data and optionally save it using collector
        print("\n" + "=" * 60)
        print("Updating data")
        print("=" * 60)
        self.fetch_current_data()

        if COLLECTOR_AVAILABLE:
            try:
                collector = AirCollector()
                collector.collect_and_save()
                self.historical_data = self.load_historical_data()
                print("Data was collected and saved")
            except Exception as e:
                print("Error while collecting data: ", e)

    def show_about(self):
        # Display application info
        print("\n" + "=" * 60)
        print("About the application")
        print("=" * 60)
        print("AirGuard 1.0")
        print("\nUse of application: air quality prediction with use of machine learning")
        print("\nStructure:")
        print("data_collection/ - data collection")
        print("module_training/ - model training")
        print("application/ - application")
        print("\nAuthor: Milana Poljanskova")
        print("=" * 60)

    def run(self):
        # Main application loop (menu navigation)
        while True:
            self.show_menu()
            action = input("Choose an action from 1 to 7: ").strip()
            if action == "1":
                self.show_current_prediction()
            elif action == "2":
                self.show_trend_chart()
            elif action == "3":
                self.show_weekly_stats()
            elif action == "4":
                self.show_dashboard()
            elif action == "5":
                self.update_data()
            elif action == "6":
                self.show_about()
            elif action == "7":
                print("Bye!")
                break
            else:
                print("Invalid action. Please choose from 1 to 7")


def main():
    try:
        app = AirGuardApp()
        app.run()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print("Error: ", e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
