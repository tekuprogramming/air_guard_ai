import request
import pandas as pd
import time
from datetime import datetime
import json
import os

from django.contrib.sites import requests


class AirCollector:
    def __init__(self, city="Prague", country="EN"):
        self.city = city
        self.country = country
        self.own_api_key = "3eaab97ed540e25e7b261d686d5dfc42" # openweathermap.org
        self.data_file = "historical_data.csv"

    # getting current weather data
    def get_weather_data(self):
        url = f"http://api.openweathermap.org/data/2.5/weather?q={self.city},{self.country}&appid={self.own_api_key}&units=metric"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            weather_data = {"temperature": data["main"]["temp"],
                            "humidity": data["main"]["humidity"],
                            "pressure": data["main"]["pressure"],
                            "wind_speed": data["wind"]["speed"],
                            "wind_deg": data["wind"].get("deg", 0),
                            "weather_main": data["weather"][0]["main"],
                            "clouds": data["clouds"]["all"]}
            return weather_data
        except Exception as e:
            print(f"Error getting weather data: \n{e}")
            return None

    def get_air_quality_data(self):
        url = "https://api.openaq.org/v2/latest?limit=100&city=Prague&parameter=pm25&parameter=pm10&parameter=no2"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            pm25 = None
            pm10 = None
            for result in data.get("results", []):
                for measurement in result.get("measurements", []):
                    if measurement["parameter"] == "pm25":
                        pm25 = measurement["value"]
                    elif measurement["parameter"] == "pm10":
                        pm10 = measurement["value"]
            return {"pm25": pm25, "pm10": pm10}
        except Exception as e:
            print(f"Error getting air quality data: \n{e}")
            return None

    def collect_and_save(self):
        timestamp = datetime.now()
        weather = self.get_weather_data()
        air_quality = self.get_air_quality_data()
        if weather and air_quality:
            record = {"timestamp": timestamp.isoformat(),
                      "data": timestamp.date().isoformat(),
                      "hour": timestamp.hour,
                      "day_of_week": 1 if timestamp.weekday() >= 5 else 0,
                      "temperature": weather["temperature"],
                      "humidity": weather["humidity"],
                      "pressure": weather["pressure"],
                      "wind_speed": weather["wind_speed"],
                      "wind_deg": weather["wind_deg"],
                      "weather_main": weather["weather_main"],
                      "clouds": weather["clouds"],
                      "pm25": air_quality["pm25"],
                      "pm10": air_quality["pm10"],
                      "air_category": self.calculate_aqi_category(air_quality["pm25"])}
            df = pd.DataFrame([record])
            if os.path.exists(self.data_file):
                df.to_csv(self.data_file, mode="a", header=False, index=False)
            else:
                df.to_csv(self.data_file, index = False)

            print("Data saved to file:", timestamp)
            return True
        return False

    # calculates the category of air quality for pm25
    def calculate_aqi_category(self, pm25):
        if pm25 is None:
            return "Unknown"
        elif pm25 <= 12:
            return "Good"
        elif pm25 <= 35:
            return "Moderate"
        elif pm25 <= 55:
            return "Sensitive"
        elif pm25 <= 150:
            return "Unhealthy"
        else:
            return "Hazardous"

    # start of indefinite data collection
    def run_continue(self, interval_hours = 2):
        print("Start of data collection for: ", self.city)
        print("Data will be collected every 2 hours")
        while True:
            success = self.collect_and_save()
            if success:
                print(f"Data collected successfully, next collection in {interval_hours} hours")
            else:
                print(f"Data collection failed, next collection in 30 minutes")
                time.sleep(1800) # 30 min
            time.sleep(interval_hours * 3600) # interval between data collections

if __name__ == "__main__":
    collector = AirCollector()
    collector.collect_and_save()
    


