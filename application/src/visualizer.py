# visualizer.py - module for visualizing air quality data

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path

# setting the style

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'good': '#00E400',        # green
    'moderate': '#FFFF00',    # yellow
    'unhealthy_sensitive': '#FF7E00',  # orange
    'unhealthy': '#FF0000',   # red
    'hazardous': '#8F3F97',   # purple
    'pm25': '#1f77b4',        # blue
    'pm10': '#ff7f0e',        # orange
    'temperature': '#d62728', # red
    'humidity': '#2ca02c'     # green
}

class AirQualityVisualizer:
    def __init__(self, output_dir = None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent / "charts"

        # creating folder for graph
        self.output_dir.mkdir(exist_ok=True)

    def plot_current_metrics(self, data, title = "current air quality metrics"):
        fig, axis = plt.subplots(1,3, figsize = (15,5))
        fig.suptitle(title, fontsize = 14, fontweight = 'bold')

        # air quality indicator
        if "aqi_score" in data:
            self._create_gauge_chart(self, ax = axis[0], value = data["aqi_score"], category = data.get("aqi_category", "unknown"))

            # primary polluters
            polluters = ["pm2.5", "pm10", "NO₂"]
            values = [data.get("pm25", 0), data.get("pm10", 0), data.get("no2", 0)]
            bars = axis[1].bar(polluters, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axis[1].set_ylabel("concentration (μg/m³)")
            axis[1].set_title("polluters")

            # adding values above columns
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axis[1].text(bar.get_x() + bar.get_width()/2., height, f"{value:.1f}", ha="center", va="bottom")

                # adding lines for edge values
                axis[1].axhline(y=25, color="red", linestyle="--", alpha=0.5, label="edge pm2.5")
                axis[1].axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="edge pm10")
                axis[1].legend(fontsize = 8)

                # weather conditions
                weather_metrics = ["temperature", "humidity", "wind_speed"]
                weather_values = [data.get("temperature", 0), data.get("humidity", 0), data.get("wind_speed", 0)]
                weather_colors = ['#d62728', '#2ca02c', '#1f77b4']
                bars = axis[2].set_ylabel("value")
                axis[2].set_title("weather conditions")

                # adding measuring units
                units = ['°C', '%', 'м/с']
                for bar, value, unit in zip(bars, weather_values, units):
                    height = bar.get_height()
                    axis[2].text(bar.get_x() + bar.get_width()/2., height, f"{value:{unit}}", ha = "center", va = "bottom")

                    # adding timestamp
                    if "timestamp" in data:
                        fig.text(0.02, 0.02, f'Data in: {data["timestamp"]}', fontsize=8, style="italic")
                        plt.tight_layout()

                        # saving
                        filename = self.output_dir / f"current_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png"
                        plt.savefig(filename, dpi = 100, bbox_inches = "tight")
                        plt.show()
                        return filename

    def _create_gauge_chart(self, ax, value, category, max_value = 300):
        # colours for categories
        colours = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97']

        # creating a speedometer
        theta = np.linspace(0, np.pi, 100)
        r = [0.8] * 100

        # drawing an arch
        n_segments = 5
        for i in range(n_segments):
            start_angle = i * np.pi / n_segments
            end_angle = (i + 1) * np.pi / n_segments
            theta_seg = np.linspace(start_angle, end_angle, 20)
            ax.fill_between(theta_seg, 0.7, 0.9, color=colours[i], alpha = 0.3)

        # drawing an arrow
        angle = (value / max_value) * np.pi
        arrow_length = 0.7
        ax.arrow(np.pi / 2, 0, arrow_length * np.cos(angle - np.pi / 2),
                 arrow_length * np.sin(angle - np.pi / 2),
                 head_width = 0.1,
                 head_length = 0.1,
                 fc = "black", ec = "black")

        # adding a sign
        ax.text(np.pi / 2, -0.2, f"{value:.0f}", ha = "center", va = "center", fontsize = 16, fontweight = "bold")
        ax.text(np.pi / 2, -0.4, category.replace("_", " ").title(), ha="center", va="center", fontsize=10, color=colours[min(int(value/(max_value/5)), 4)])
        ax.set_xlim(0, np.pi)
        ax.set_ylim(-0.5, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("air quality index AQI", fontsize = 16)

    def plot_24h_trend(self, df, title = "24h air quality trend"):
        fig, ax = plt.subplots(3,1, figsize = (12,10))
        fig.suptitle(title, fontsize = 14, fontweight = 'bold')

        # changing timestamp
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("datetime")

            # trend pm2.5 and pm10
            ax1 = ax[0]
            if "pm25" in df.columns:
                ax1.plot(df["datetime"], df["pm25"], "o-", color = COLORS["pm25"], label = "pm2.5", linewidth = 2, markersize = 4)
            if "pm10" in df.columns:
                ax1.plot(df["datetime"], df["pm10"], "s-", color = COLORS["pm10"], label = "pm10", linewidth = 2, markersize = 4)
            ax1.set_ylabel("concentration (μg/m³)")
            ax1.set_title("polluters")
            ax.legend(loc = "upper right")
            ax1.grid(True, alpha = 0.3)

            # adding WHO norms
            ax1.axhline(y = 25, color="red", linestyle="--", alpha=0.5, label="pm2.5 norm")
            ax1.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="pm10 norm")
