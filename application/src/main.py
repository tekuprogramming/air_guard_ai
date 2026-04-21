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
        model_path = PROJECT_ROOT / "module_training" / "trained_model_compact.pkl"
        print(model_path)
        print(model_path.exists())
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
        config_path = PROJECT_ROOT / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
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
        dates = [datetime.now() - timedelta(hours=i) for i in range(168, 0, -1)]

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
            record = {
                "timestamp": dt.isoformat(),
                "date": dt.date().isoformat(),
                "hour": hour,
                "day_of_week": dt.weekday(),
                "is_weekend": is_weekend,
                "temperature": 10 + 5 * np.sin(hour * np.pi / 12) + np.random.normal(0, 2),
                "humidity": 70 + 10 * np.random.randn(),
                "pressure": 1013 + 5 * np.random.randn(),
                "wind_speed": 3 + np.random.exponential(1),
                "clouds": 50 + 20 * np.random.randn(),
                "pm25": max(0, base_pm25 + np.random.normal(0, 5)),
                "pm10": max(0, base_pm10 + np.random.normal(0, 8)),
                "aqi_category": self.calculate_aqi_category(base_pm25)
            }
            data.append(record)
        df = pd.DataFrame(data)
        print(f"Created {len(df)} records of demo data")
        return df

    # calculates the category of air quality by PM2.5
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

    # getting current data from API
    def fetch_current_data(self):
        if not self.api_key:
            print("API key not found. Using most recent historic data")
            return self.get_last_historical_data()
        try:
            # getting weather
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={self.city},{self.country}&appid={self.api_key}&units=metric"
            weather_response = requests.get(weather_url, timeout=10)
            weather_data = weather_response.json()
            if weather_response.status_code != 200:
                print(f"API weather error: {weather_data.get('message', 'Unknown error')}")
                return self.get_last_historical_data()

            pm25 = None
            pm10 = None
            try:
                air_url = f"https://api.openaq.org/v2/latest?city={self.city}&parameter=pm25&parameter=pm10"
                air_response = requests.get(air_url, timeout=10)
                air_data = air_response.json()
                for result in air_data.get("results", []):
                    for measurement in result.get("measurements", []):
                        if measurement["parameter"] == "pm25":
                            pm25 = measurement["value"]
                        elif measurement["parameter"] == "pm10":
                            pm10 = measurement["value"]
            except:
                print("Couldn't get air quality data")

            # forming current data
            self.current_data = {
                "timestamp": datetime.now().isoformat(),
                "date": datetime.now().date().isoformat(),
                "hour": datetime.now().hour,
                "day_of_week": datetime.now().weekday(),
                "is_weekend": 1 if datetime.now().weekday() >= 5 else 0,
                "temperature": weather_data["main"]["temp"],
                "feels_like": weather_data["main"]["feels_like"],
                "humidity": weather_data["main"]["humidity"],
                "pressure": weather_data["main"]["pressure"],
                "wind_speed": weather_data["wind"]["speed"],
                "wind_deg": weather_data["wind"].get("deg", 0),
                "weather_main": weather_data["weather"][0]["main"],
                "weather_desc": weather_data["weather"][0]["description"],
                "clouds": weather_data["clouds"]["all"],
                "pm25": pm25,
                "pm10": pm10,
                "no2": 0,
                "aqi_category": self.calculate_aqi_category(pm25) if pm25 is not None else "unknown"
            }
            print("Successfully received data from API")
            return self.current_data

        except Exception as e:
            print(f"Couldn't get air quality data: {e}")
            return self.get_last_historical_data()

    # getting the last record from historical data
    def get_last_historical_data(self):
        if self.historical_data is not None and len(self.historical_data) > 0:
            last_record = self.historical_data.iloc[-1].to_dict()
            last_record["timestamp"] = datetime.now().isoformat()
            self.current_data = last_record
            print("Using the last historical data")
            return self.current_data
        # if historical data doesn't exist -> create own data
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

    # creating prediction based on current data
    def make_prediction(self):
        if self.current_data is None:
            self.fetch_current_data()

        # preparing features for the model
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
            "weather_main_encoded": self.encode_weather(self.current_data.get("weather_main", "Clear")),
            "is_weekend": self.current_data.get("is_weekend", 0),
            "pm25": self.current_data.get("pm25", 0),
            "pm10": self.current_data.get("pm10", 0)
        }

        # getting predictions from the model
        prediction = self.predictor.predict(features)
        return prediction

    # encoding weather type for model
    def encode_weather(self, weather_main):
        try:
            return self.predictor.label_encoder.transform([weather_main])[0]
        except Exception:
            return 0

    # checking if the date is a holiday
    def is_czech_holiday(self, date):
        czech_holidays = [
            "2026-01-01", "2026-04-01", "2026-05-01", "2026-05-08",
            "2026-07-05", "2026-07-06", "2026-09-28", "2026-10-28",
            "2026-12-24", "2026-12-25", "2026-12-26"
        ]
        return 1 if date.strftime("%Y-%m-%d") in czech_holidays else 0

    # checking if the current time is rush hour
    def is_rush_hour(self, dt):
        hour = dt.hour
        is_week_day = dt.weekday() < 5
        return 1 if (is_week_day and (7 <= hour <= 9 or 15 <= hour <= 17)) else 0

    # getting recommendation based on air quality category
    def get_recommendation(self, category):
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

    # shows main menu
    def show_menu(self):
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

    # shows current prediction
    def show_current_prediction(self):
        print("\n" + "=" * 60)
        print("Current air quality prediction")
        print("=" * 60)

        # getting current data
        if self.current_data is None:
            self.fetch_current_data()

        # making a prediction
        prediction = self.make_prediction()

        # printing information
        print(f"\nLocation: {self.city}, {self.country}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("\nWeather now: ")
        print(f"Temperature: {self.current_data.get('temperature', 'N/A')}°C")
        print(f"Humidity: {self.current_data.get('humidity', 'N/A')}%")
        print(f"Wind: {self.current_data.get('wind_speed', 'N/A')}m/s")
        print(f"Pressure: {self.current_data.get('pressure', 'N/A')}hPa")
        print(f"\nPollution: ")
        print(f"PM2.5: {self.current_data.get('pm25', 'N/A')}μg/m³")
        print(f"PM10: {self.current_data.get('pm10', 'N/A')}μg/m³")
        print("\nModel prediction: ")
        print(f"Category: {prediction['category'].upper()}")
        print(f"Confidence: {prediction['confidence'] * 100:.1f}%")
        print("\nRecommendation: ")
        rec = self.get_recommendation(prediction["category"])
        print(rec['en'])

        if "probabilities" in prediction:
            print("\nProbability distribution")
            for cat, prob in prediction["probabilities"].items():
                bar = '[]' * int(prob * 20)
                print(f"{cat:20} {bar} {prob*100:.1f}%")

        self.visualizer.plot_current_metrics(self.current_data)

    # shows 24h trend graph
    def show_trend_chart(self):
        print("\n" + "=" * 60)
        print("24h trend")
        print("=" * 60)
        if self.historical_data is not None and len(self.historical_data) > 0:
            recent_data = self.historical_data.tail(24)
            self.visualizer.plot_24h_trend(recent_data)
            print("Trend graph was created")
        else:
            print("No historical data to make a graph")

    # shows weekly statistics
    def show_weekly_stats(self):
        print("\n" + "=" * 60)
        print("Weekly statistics")
        print("=" * 60)
        if self.historical_data is not None and len(self.historical_data) >= 168:
            week_data = self.historical_data.tail(168)
            self.visualizer.plot_weekly_stats(week_data)
            print("Weekly statistics was created")
        else:
            print("Not enough data for weekly statistics")
            print(f"There are {len(self.historical_data) if self.historical_data is not None else 0} records. 168 needed")

    # shows full dashboard
    def show_dashboard(self):
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

    # updates current data
    def update_data(self):
        print("\n" + "=" * 60)
        print("Updating data")
        print("=" * 60)
        self.fetch_current_data()

        # if collector exists, starts collecting
        if COLLECTOR_AVAILABLE:
            try:
                collector = AirCollector()
                collector.collect_and_save()
                self.historical_data = self.load_historical_data()
                print("Data was collected and saved")
            except Exception as e:
                print("Error while collecting data: ", e)

    # application information
    def show_about(self):
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

    # main application cycle
    def run(self):
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

# main application entrance point
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

