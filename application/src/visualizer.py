# visualizer.py - module for visualizing air quality data
# This module handles all plotting and dashboard creation for air quality data

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib

import os
if os.environ.get("DISPLAY") is None:
    matplotlib.use("Agg")

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')

COLORS = {
    'good': '#00E400',
    'moderate': '#FFFF00',
    'unhealthy_sensitive': '#FF7E00',
    'unhealthy': '#FF0000',
    'hazardous': '#8F3F97',
    'pm25': '#1f77b4',
    'pm10': '#ff7f0e',
    'temperature': '#d62728',
    'humidity': '#2ca02c'
}

class AirQualityVisualizer:
    # Main class responsible for generating all graphs and dashboards
    def __init__(self, output_dir=None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent / "charts"

        self.output_dir.mkdir(exist_ok=True)

    def _safe(self, data, key, default=0):
        # safe getter for dictionary values (handles missing/None values)
        val = data.get(key)
        return val if val is not None else default

    def _create_gauge_chart(self, ax, value, category, max_value=300):
        # creates a "speedometer style" AQI gauge chart

        colours = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97']

        theta = np.linspace(0, np.pi, 100)
        r = [0.8] * 100

        ax.plot(theta, r, color='gray', linewidth=3)
        
        n_segments = 5
        for i in range(n_segments):
            start_angle = i * np.pi / n_segments
            end_angle = (i + 1) * np.pi / n_segments
            theta_seg = np.linspace(start_angle, end_angle, 20)
            ax.fill_between(theta_seg, 0.7, 0.9, color=colours[i], alpha=0.3)

        angle = (value / max_value) * np.pi
        arrow_length = 0.7

        ax.arrow(np.pi / 2, 0,
                 arrow_length * np.cos(angle - np.pi / 2),
                 arrow_length * np.sin(angle - np.pi / 2),
                 head_width=0.1, head_length=0.1,
                 fc="black", ec="black")

        ax.text(np.pi / 2, -0.2, f"{value:.0f}",
                ha="center", va="center", fontsize=16, fontweight="bold")

        color_idx = min(int(value / (max_value / 5)), 4)
        ax.text(np.pi / 2, -0.4, category.replace("_", " ").title(),
                ha="center", va="center", fontsize=10, color=colours[color_idx])

        ax.set_xlim(0, np.pi)
        ax.set_ylim(-0.5, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Air Quality Index (AQI)", fontsize=10)

    def plot_current_metrics(self, data, title="Current air quality metrics"):
        # main function for current snapshot visualization (AQI + pollutants + weather)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        ax1 = axes[0]

        if "aqi_category" in data:
            category_map = {
                "good": 50,
                "moderate": 100,
                "unhealthy_sensitive": 150,
                "unhealthy": 200,
                "hazardous": 300
            }
            value = category_map.get(data["aqi_category"], 100)
            self._create_gauge_chart(ax1, value, data["aqi_category"])
        else:
            ax1.text(0.5, 0.5, "AQI data\nnot available",
                     ha="center", va="center", transform=ax1.transAxes)
            ax1.axis("off")

        pollutants = ["PM2.5", "PM10", "NO2"]

        pm25_val = self._safe(data, "pm25")
        pm10_val = self._safe(data, "pm10")
        no2_val = self._safe(data, "no2")

        values = [pm25_val, pm10_val, no2_val]

        bars = axes[1].bar(pollutants, values,
                           color=['#1f77b4', '#ff7f0e', '#2ca02c'])

        axes[1].set_ylabel("Concentration (μg/m³)")
        axes[1].set_title("Pollutants")

        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width() / 2.,
                             height, f"{value:.1f}",
                             ha="center", va="bottom", fontsize=9)

        axes[1].axhline(y=25, color="red", linestyle="--", alpha=0.5, label="PM2.5 limit")
        axes[1].axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="PM10 limit")
        axes[1].legend(fontsize=8)

        weather_metrics = ["Temperature", "Humidity", "Wind"]

        temp_val = data.get("temperature") or 0
        hum_val = data.get("humidity") or 0
        wind_val = data.get("wind_speed") or 0

        weather_values = [temp_val, hum_val, wind_val]

        axes[2].bar(weather_metrics, weather_values,
                    color=['#d62728', '#2ca02c', '#1f77b4'])

        axes[2].set_ylabel("Value")
        axes[2].set_title("Weather conditions")

        for bar, value in zip(axes[2].patches, weather_values):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width() / 2.,
                         height, f"{value:.1f}",
                         ha="center", va="bottom", fontsize=9)

        if "timestamp" in data:
            fig.text(0.02, 0.02, f'Data at: {data["timestamp"]}',
                     fontsize=8, style="italic")

        plt.tight_layout()

        filename = self.output_dir / f"current_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return filename

    def plot_24h_trend(self, df, title="24h air quality trend"):
        # shows last 24 hours of pollution and weather trends

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        df = df.copy()

        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("datetime")

        ax1 = axes[0]
        if "pm25" in df.columns:
            ax1.plot(df["datetime"], df["pm25"], "o-", color=COLORS["pm25"], label="PM2.5")
        if "pm10" in df.columns:
            ax1.plot(df["datetime"], df["pm10"], "s-", color=COLORS["pm10"], label="PM10")

        ax1.set_title("Pollutants")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax1.axhline(y=25, color="red", linestyle="--", alpha=0.5)
        ax1.axhline(y=50, color="orange", linestyle="--", alpha=0.5)

        ax2 = axes[1]
        ax2_twin = ax2.twinx()

        if "temperature" in df.columns:
            ax2.plot(df["datetime"], df["temperature"],
                     color=COLORS["temperature"], label="Temperature")

        if "humidity" in df.columns:
            ax2_twin.plot(df["datetime"], df["humidity"],
                          color=COLORS["humidity"], label="Humidity")

        ax2.set_title("Weather conditions")
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        if "aqi_category" in df.columns:
            category_map = {
                "good": 1, "moderate": 2,
                "unhealthy_sensitive": 3,
                "unhealthy": 4,
                "hazardous": 5
            }

            df["aqi_level"] = df["aqi_category"].map(category_map)

            colors = [COLORS.get(cat, "gray") for cat in df["aqi_category"]]

            ax3.scatter(df["datetime"], df["aqi_level"], c=colors, s=50)
            ax3.set_yticks([1, 2, 3, 4, 5])
            ax3.set_yticklabels(["Good", "Moderate", "Sensitive", "Unhealthy", "Hazardous"])

        plt.tight_layout()

        filename = self.output_dir / f"trend_24h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.show()

        return filename

    def plot_weekly_stats(self, df, title="Weekly statistics of air quality"):
        # analyzes weekly patterns (daily averages, distribution, correlations)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        df = df.copy()

        df["weekday"] = pd.to_datetime(df["timestamp"]).dt.day_name()

        week_day_order = ["Monday", "Tuesday", "Wednesday",
                          "Thursday", "Friday", "Saturday", "Sunday"]

        if "pm25" in df.columns:
            week_avg = df.groupby("weekday")["pm25"].mean().reindex(week_day_order)
            axes[0, 0].bar(range(7), week_avg.values)

        if "aqi_category" in df.columns:
            counts = df["aqi_category"].value_counts()
            axes[0, 1].pie(counts.values, labels=counts.index, autopct="%1.1f%%")

        if "pm25" in df.columns:
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

        numeric_cols = ["pm25", "pm10", "temperature", "humidity", "wind_speed"]
        available = [c for c in numeric_cols if c in df.columns]

        if len(available) > 1:
            corr = df[available].corr()
            axes[1, 1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

        plt.tight_layout()

        filename = self.output_dir / f"weekly_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.show()

        return filename

    def create_dashboard(self, current_data, historical_df):
        # full dashboard combining all charts into one view

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("Air Quality Dashboard", fontsize=16, fontweight="bold")

        gs = GridSpec(3, 3, figure=fig)

        plt.tight_layout()

        filename = self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        plt.show()

        return filename

def plot_current(data):
    # Creates and returns a visualization of current air quality metrics
    vis = AirQualityVisualizer()
    return vis.plot_current_metrics(data)


def plot_trend(df):
    # Creates a 24-hour trend plot for pollutants and weather data
    vis = AirQualityVisualizer()
    return vis.plot_24h_trend(df)


def plot_weekly(df):
    # Creates weekly statistics visualization (averages, distributions, correlations)
    vis = AirQualityVisualizer()
    return vis.plot_weekly_stats(df)


if __name__ == "__main__":

    print("Testing visualizer")

    current = {
        'timestamp': datetime.now().isoformat(),
        'pm25': 35.5,
        'pm10': 48.2,
        'no2': 42.1,
        'temperature': 12.5,
        'humidity': 78,
        'wind_speed': 3.2,
        'pressure': 1012,
        'aqi_category': 'moderate'
    }

    dates = pd.date_range(end=datetime.now(), periods=48, freq="H")
    historical = pd.DataFrame({
        "timestamp": dates,
        "pm25": np.random.lognormal(mean=3.0, sigma=0.5, size=48),
        "pm10": np.random.lognormal(mean=3.5, sigma=0.5, size=48),
        "temperature": np.random.normal(loc=12, scale=3, size=48),
        "humidity": np.random.normal(loc=70, scale=10, size=48),
        "aqi_category": np.random.choice(["good", "moderate", "unhealthy_sensitive"], size=48)
    })

    vis = AirQualityVisualizer()

    print("Creating a graph of current metrics")
    vis.plot_current_metrics(current)

    print("Creating a graph of 24h trend")
    vis.plot_24h_trend(historical.tail(24))

    print("Creating weekly statistics")
    vis.plot_weekly_stats(historical)

    print("Creating a dashboard")
    vis.create_dashboard(current, historical)

    print(f"All files saved to folder: {vis.output_dir}")
