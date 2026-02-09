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
        # try:
            # response = requests.get(url, timeout=10)
            # data = response.json()
            # pm25 = None
            # pm10 = None
            # for result in data.get("results", []):
                # for measurement in result.get("measurements", []):
                    # if measurement.

