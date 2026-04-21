# visualizer.py - module for visualizing air quality data

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
    def __init__(self, output_dir=None):
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent / "charts"

        # creating folder for graph
        self.output_dir.mkdir(exist_ok=True)

    def _safe(self, data, key, default=0):
        val = data.get(key)
        return val if val is not None else default

    def _create_gauge_chart(self, ax, value, category, max_value=300):
        # colours for categories
        colours = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97']

        # creating a speedometer
        theta = np.linspace(0, np.pi, 100)
        r = [0.8] * 100
        
        # draw the arc background
        ax.plot(theta, r, color='gray', linewidth=3)

        # drawing colored segments
        n_segments = 5
        for i in range(n_segments):
            start_angle = i * np.pi / n_segments
            end_angle = (i + 1) * np.pi / n_segments
            theta_seg = np.linspace(start_angle, end_angle, 20)
            ax.fill_between(theta_seg, 0.7, 0.9, color=colours[i], alpha=0.3)

        # drawing an arrow
        angle = (value / max_value) * np.pi
        arrow_length = 0.7
        ax.arrow(np.pi / 2, 0, 
                 arrow_length * np.cos(angle - np.pi / 2),
                 arrow_length * np.sin(angle - np.pi / 2),
                 head_width=0.1, head_length=0.1,
                 fc="black", ec="black")

        # adding value text
        ax.text(np.pi / 2, -0.2, f"{value:.0f}", ha="center", va="center", fontsize=16, fontweight="bold")
        
        # Determine color index safely
        color_idx = min(int(value / (max_value / 5)), 4)
        ax.text(np.pi / 2, -0.4, category.replace("_", " ").title(), 
                ha="center", va="center", fontsize=10, color=colours[color_idx])
        
        ax.set_xlim(0, np.pi)
        ax.set_ylim(-0.5, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Air Quality Index (AQI)", fontsize=10)

    def plot_current_metrics(self, data, title="Current air quality metrics"):
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
            ax1.text(0.5, 0.5, "AQI data\nnot available", ha="center", va="center", transform=ax1.transAxes)
            ax1.axis("off")

        # primary polluters - with None handling
        pollutants = ["PM2.5", "PM10", "NO2"]

        # Get values and replace None with 0
        pm25_val = self._safe(data, "pm25")
        pm10_val = self._safe(data, "pm10")
        no2_val = self._safe(data, "no2")

        pm25_val = pm25_val if pm25_val is not None else 0
        pm10_val = pm10_val if pm10_val is not None else 0
        no2_val = no2_val if no2_val is not None else 0

        values = [pm25_val, pm10_val, no2_val]

        bars = axes[1].bar(pollutants, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1].set_ylabel("Concentration (μg/m³)")
        axes[1].set_title("Pollutants")

        # adding values above columns
        for bar, value in zip(bars, values):
            if value > 0:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width() / 2., height, f"{value:.1f}",
                             ha="center", va="bottom", fontsize=9)

        # adding lines for limit values
        axes[1].axhline(y=25, color="red", linestyle="--", alpha=0.5, label="PM2.5 limit")
        axes[1].axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="PM10 limit")
        axes[1].legend(fontsize=8)

        # weather conditions - with None handling
        weather_metrics = ["Temperature", "Humidity", "Wind"]

        temp_val = data.get("temperature")
        hum_val = data.get("humidity")
        wind_val = data.get("wind_speed")

        temp_val = temp_val if temp_val is not None else 0
        hum_val = hum_val if hum_val is not None else 0
        wind_val = wind_val if wind_val is not None else 0

        weather_values = [temp_val, hum_val, wind_val]
        weather_colors = ['#d62728', '#2ca02c', '#1f77b4']
        weather_bars = axes[2].bar(weather_metrics, weather_values, color=weather_colors)
        axes[2].set_ylabel("Value")
        axes[2].set_title("Weather conditions")

        # adding measuring units
        units = ['°C', '%', 'm/s']
        for bar, value, unit in zip(weather_bars, weather_values, units):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width() / 2., height, f"{value:.1f}{unit}",
                         ha="center", va="bottom", fontsize=9)

        # adding timestamp
        if "timestamp" in data:
            fig.text(0.02, 0.02, f'Data at: {data["timestamp"]}', fontsize=8, style="italic")

        try:
            plt.tight_layout()
        except:
            plt.subplots_adjust(wspace=0.3)

        # saving
        filename = self.output_dir / f"current_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return filename

    def plot_24h_trend(self, df, title="24h air quality trend"):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # make a copy to avoid modifying original
        df = df.copy()
        
        # convert timestamp
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("datetime")

        # trend pm2.5 and pm10
        ax1 = axes[0]
        if "pm25" in df.columns:
            ax1.plot(df["datetime"], df["pm25"], "o-", color=COLORS["pm25"], 
                    label="PM2.5", linewidth=2, markersize=4)
        if "pm10" in df.columns:
            ax1.plot(df["datetime"], df["pm10"], "s-", color=COLORS["pm10"], 
                    label="PM10", linewidth=2, markersize=4)
        ax1.set_ylabel("Concentration (μg/m³)")
        ax1.set_title("Pollutants")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # adding WHO norms
        ax1.axhline(y=25, color="red", linestyle="--", alpha=0.5, label="PM2.5 norm")
        ax1.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="PM10 norm")

        # temperature and humidity
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        lines = []
        
        if "temperature" in df.columns:
            line1 = ax2.plot(df["datetime"], df["temperature"], "o-", color=COLORS['temperature'],
                            label="Temperature", linewidth=2, markersize=4)
            ax2.set_ylabel("Temperature (°C)", color=COLORS["temperature"])
            ax2.tick_params(axis='y', labelcolor=COLORS["temperature"])
            lines.extend(line1)
            
        if "humidity" in df.columns:
            line2 = ax2_twin.plot(df["datetime"], df["humidity"], "s-", color=COLORS['humidity'],
                                  label="Humidity", linewidth=2, markersize=4)
            ax2_twin.set_ylabel("Humidity (%)", color=COLORS["humidity"])
            ax2_twin.tick_params(axis='y', labelcolor=COLORS["humidity"])
            lines.extend(line2)
            
        ax2.set_title("Weather conditions")
        ax2.grid(True, alpha=0.3)
        
        # combine legends
        if lines:
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc="upper right")

        # AQI categories
        ax3 = axes[2]
        if "aqi_category" in df.columns:
            # convert categories to numbers for plotting
            category_map = {"good": 1, "moderate": 2, "unhealthy_sensitive": 3, 
                           "unhealthy": 4, "hazardous": 5}
            df["aqi_level"] = df["aqi_category"].map(category_map)

            # colors for dots
            colors = [COLORS.get(cat, "gray") for cat in df["aqi_category"]]
            ax3.scatter(df["datetime"], df["aqi_level"], c=colors, s=50, alpha=0.7)
            ax3.plot(df["datetime"], df["aqi_level"], "gray", alpha=0.3, linewidth=1)
            ax3.set_yticks([1, 2, 3, 4, 5])
            ax3.set_yticklabels(["Good", "Moderate", "Unhealthy\nSensitive", 
                                "Unhealthy", "Hazardous"])
            ax3.set_ylabel("Quality category")
            ax3.set_title("Changes of air quality category")
            ax3.grid(True, alpha=0.3)

        # format time axis for all subplots
        for ax in axes:
            if "datetime" in df.columns and len(df) > 0:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.set_xlabel("Time")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # saving
        filename = self.output_dir / f'trend_24h_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.show()
        return filename

    def plot_weekly_stats(self, df, title="Weekly statistics of air quality"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # make a copy to avoid modifying original
        df = df.copy()
        
        # adding day of week
        df["weekday"] = pd.to_datetime(df["timestamp"]).dt.day_name()
        week_day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                         "Friday", "Saturday", "Sunday"]
        short_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        # average pm2.5 for week days
        ax1 = axes[0, 0]
        if "pm25" in df.columns:
            week_day_pm25 = df.groupby("weekday")["pm25"].mean().reindex(week_day_order)
            bars = ax1.bar(range(7), week_day_pm25.values, color="steelblue")
            ax1.set_xticks(range(7))
            ax1.set_xticklabels(short_days)
            ax1.set_ylabel("Average PM2.5 (μg/m³)")
            ax1.set_title("Average PM2.5 by weekday")

            # adding values
            for bar, value in zip(bars, week_day_pm25.values):
                if not np.isnan(value):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height, f"{value:.1f}", 
                            ha="center", va="bottom", fontsize=8)

        # category distribution
        ax2 = axes[0, 1]
        if "aqi_category" in df.columns:
            category_counts = df["aqi_category"].value_counts()
            colors = [COLORS.get(cat, "gray") for cat in category_counts.index]
            wedges, texts, autotexts = ax2.pie(
                category_counts.values, 
                labels=category_counts.index, 
                colors=colors, 
                autopct='%1.1f%%', 
                startangle=90
            )
            ax2.set_title("AQI category distribution")

        # box plot by hour
        ax3 = axes[1, 0]
        if "pm25" in df.columns:
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
            hourly_data = [df[df["hour"] == h]["pm25"].dropna().values for h in range(24)]
            # filter out empty arrays
            hourly_data = [data for data in hourly_data if len(data) > 0]
            positions = [h for h in range(24) if len(df[df["hour"] == h]["pm25"].dropna()) > 0]
            
            if hourly_data:
                box = ax3.boxplot(hourly_data, positions=positions, patch_artist=True)
                ax3.set_xlabel("Hour of day")
                ax3.set_ylabel("PM2.5 (μg/m³)")
                ax3.set_title("PM2.5 distribution by hour")
                ax3.grid(True, alpha=0.3)
                
                # color the boxes
                for box_item in box["boxes"]:
                    box_item.set_facecolor("lightblue")

        # correlations between parameters
        ax4 = axes[1, 1]
        numeric_cols = ["pm25", "pm10", "temperature", "humidity", "wind_speed", "pressure"]
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()
            im = ax4.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            ax4.set_xticks(range(len(available_cols)))
            ax4.set_yticks(range(len(available_cols)))
            ax4.set_xticklabels(available_cols, rotation=45, ha="right")
            ax4.set_yticklabels(available_cols)
            ax4.set_title("Parameter correlations")

            # adding values into cells
            for i in range(len(available_cols)):
                for j in range(len(available_cols)):
                    val = corr_matrix.iloc[i, j]
                    text_color = 'white' if abs(val) > 0.5 else 'black'
                    ax4.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color)
                    
            plt.colorbar(im, ax=ax4)

        plt.tight_layout()

        # saving
        filename = self.output_dir / f'weekly_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.show()
        return filename

    # comparison of current values with the norm
    def plot_comparison_with_norms(self, data, norms=None):
        if norms is None:
            norms = {
                'PM2.5': {'who': 25},
                'PM10': {'who': 50},
                'NO2': {'who': 40}
            }

        fig, ax = plt.subplots(figsize=(10, 6))

        pollutants = []
        current_values = []
        who_norms = []
        colors = []

        # Get values with None handling
        pm25_val = data.get('pm25')
        pm10_val = data.get('pm10')
        no2_val = data.get('no2')

        pm25_val = pm25_val if pm25_val is not None else 0
        pm10_val = pm10_val if pm10_val is not None else 0
        no2_val = no2_val if no2_val is not None else 0

        for pollutant, value in [('PM2.5', pm25_val),
                                 ('PM10', pm10_val),
                                 ('NO2', no2_val)]:
            if value > 0:
                pollutants.append(pollutant)
                current_values.append(value)
                who_norms.append(norms[pollutant]['who'])
                colors.append("red" if value > norms[pollutant]['who'] else "green")

        if not pollutants:
            ax.text(0.5, 0.5, "No pollutant data available", ha='center', va='center', transform=ax.transAxes)
        else:
            x = np.arange(len(pollutants))
            width = 0.35

            bars1 = ax.bar(x - width / 2, current_values, width, label="Current values",
                           color=colors, edgecolor="black")
            bars2 = ax.bar(x + width / 2, who_norms, width, label="WHO norms",
                           color="gray", alpha=0.5, edgecolor="black")

            ax.set_xlabel("Pollutant")
            ax.set_ylabel("Concentration (μg/m³)")
            ax.set_title("Comparison of current values with WHO norms")
            ax.set_xticks(x)
            ax.set_xticklabels(pollutants)
            ax.legend()

            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height, f"{height:.1f}",
                        ha="center", va="bottom", fontsize=9)
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height, f"{height:.1f}",
                        ha="center", va="bottom", fontsize=9)

            ax.grid(True, alpha=0.3, axis="y")

        try:
            plt.tight_layout()
        except:
            plt.subplots_adjust()

        filename = self.output_dir / f'comparison_norms_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.show()
        return filename

    # creates a full dashboard with multiple graphs
    def create_dashboard(self, current_data, historical_df):
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"Air Quality Dashboard - Prague\n{datetime.now().strftime('%Y-%m-%d %H:%M')}",
                     fontsize=16, fontweight="bold")

        # creating a grid for the plots with adjusted spacing
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

        # ROW 1

        # 1. current AQI (speedometer)
        ax1 = fig.add_subplot(gs[0, 0])
        if "aqi_category" in current_data:
            category_map = {
                "good": 50,
                "moderate": 100,
                "unhealthy_sensitive": 150,
                "unhealthy": 200,
                "hazardous": 300
            }
            value = category_map.get(current_data["aqi_category"], 100)
            self._create_gauge_chart(ax1, value, current_data["aqi_category"])
        else:
            ax1.text(0.5, 0.5, "No AQI data", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_title("Air Quality Index (AQI)")
            ax1.axis("off")

        # 2. current pollutants - with None handling
        ax2 = fig.add_subplot(gs[0, 1])

        # Safe extraction of values (handle None)
        pm25_val = current_data.get("pm25")
        pm10_val = current_data.get("pm10")
        no2_val = current_data.get("no2")

        # Replace None with 0
        pm25_val = pm25_val if pm25_val is not None else 0
        pm10_val = pm10_val if pm10_val is not None else 0
        no2_val = no2_val if no2_val is not None else 0

        pollutants = ["PM2.5", "PM10", "NO2"]
        values = [pm25_val, pm10_val, no2_val]

        bars = ax2.bar(pollutants, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title("Current Pollutants")
        ax2.set_ylabel("μg/m³")

        # Add values on bars
        for bar, val in zip(bars, values):
            if val > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height, f"{val:.1f}",
                         ha="center", va="bottom", fontsize=8)

        # 3. 24h trend
        ax3 = fig.add_subplot(gs[0, 2])
        if len(historical_df) > 0 and isinstance(historical_df, pd.DataFrame):
            recent = historical_df.tail(24)
            if "pm25" in recent.columns and recent["pm25"].notna().any():
                ax3.plot(range(len(recent)), recent["pm25"].values, color=COLORS["pm25"],
                         label="PM2.5", marker='o', markersize=4, linewidth=1.5)
            if "pm10" in recent.columns and recent["pm10"].notna().any():
                ax3.plot(range(len(recent)), recent["pm10"].values, color=COLORS["pm10"],
                         label="PM10", marker='s', markersize=4, linewidth=1.5)
            ax3.set_title("24h Trend")
            ax3.legend(fontsize=8)
            ax3.set_xlabel("Hours ago")
            ax3.set_ylabel("μg/m³")
            ax3.set_xticks(range(0, 24, 4))
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No trend data", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("24h Trend")

        # ROW 2

        # 4. category distribution
        ax4 = fig.add_subplot(gs[1, 0])
        if "aqi_category" in historical_df.columns and len(historical_df) > 0:
            cat_counts = historical_df["aqi_category"].value_counts()
            if len(cat_counts) > 0:
                colors = [COLORS.get(cat, "gray") for cat in cat_counts.index]
                ax4.pie(cat_counts.values, labels=cat_counts.index, colors=colors, autopct='%1.1f%%')
                ax4.set_title("AQI Category Distribution")
            else:
                ax4.text(0.5, 0.5, "No category data", ha="center", va="center", transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, "No category data", ha="center", va="center", transform=ax4.transAxes)

        # 5. hourly statistics
        ax5 = fig.add_subplot(gs[1, 1])
        if "timestamp" in historical_df.columns and len(historical_df) > 0:
            df_hourly = historical_df.copy()
            df_hourly["hour"] = pd.to_datetime(df_hourly["timestamp"]).dt.hour
            hourly_avg = df_hourly.groupby("hour")["pm25"].mean()
            if len(hourly_avg) > 0 and hourly_avg.notna().any():
                ax5.plot(hourly_avg.index, hourly_avg.values, "o-", color=COLORS["pm25"], markersize=4)
                ax5.set_title("Average PM2.5 by Hour")
                ax5.set_xlabel("Hour")
                ax5.set_ylabel("PM2.5 (μg/m³)")
                ax5.grid(True, alpha=0.3)
                ax5.set_xticks(range(0, 24, 3))
                ax5.set_xlim(0, 23)
            else:
                ax5.text(0.5, 0.5, "No hourly data", ha="center", va="center", transform=ax5.transAxes)
        else:
            ax5.text(0.5, 0.5, "No hourly data", ha="center", va="center", transform=ax5.transAxes)

        # 6. weather conditions
        ax6 = fig.add_subplot(gs[1, 2])

        # Safe extraction of weather values
        temp_val = current_data.get("temperature")
        hum_val = current_data.get("humidity")
        wind_val = current_data.get("wind_speed")
        press_val = current_data.get("pressure")

        temp_val = temp_val if temp_val is not None else 0
        hum_val = hum_val if hum_val is not None else 0
        wind_val = wind_val if wind_val is not None else 0
        press_val = press_val if press_val is not None else 1013

        weather_data = [temp_val, hum_val, wind_val, press_val / 10]  # scale pressure
        weather_labels = ['Temp (°C)', 'Humidity (%)', 'Wind (m/s)', 'Pressure (×10 hPa)']
        weather_colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd']

        weather_bars = ax6.bar(weather_labels, weather_data, color=weather_colors)
        ax6.set_title("Current Weather")
        ax6.tick_params(axis="x", rotation=45)

        # Add values on bars
        for bar, val in zip(weather_bars, weather_data):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2., height, f"{val:.1f}",
                     ha="center", va="bottom", fontsize=8)

        # ROW 3

        # 7. weekly statistics - daily averages
        ax7 = fig.add_subplot(gs[2, :])
        if "timestamp" in historical_df.columns and len(historical_df) > 0:
            df_daily = historical_df.copy()
            df_daily["date"] = pd.to_datetime(df_daily["timestamp"]).dt.date
            daily_avg = df_daily.groupby("date")[["pm25", "pm10"]].mean().tail(7)

            if len(daily_avg) > 0:
                x = range(len(daily_avg))
                if "pm25" in daily_avg.columns and daily_avg["pm25"].notna().any():
                    ax7.plot(x, daily_avg["pm25"].values, "o-", color=COLORS["pm25"],
                             label="PM2.5", linewidth=2, markersize=6)
                if "pm10" in daily_avg.columns and daily_avg["pm10"].notna().any():
                    ax7.plot(x, daily_avg["pm10"].values, "s-", color=COLORS["pm10"],
                             label="PM10", linewidth=2, markersize=6)
                ax7.set_title("Daily Averages (Last 7 Days)")
                ax7.set_xlabel("Day")
                ax7.set_ylabel("Concentration (μg/m³)")
                ax7.set_xticks(x)
                ax7.set_xticklabels([d.strftime("%d.%m") for d in daily_avg.index])
                ax7.legend()
                ax7.grid(True, alpha=0.3)
            else:
                ax7.text(0.5, 0.5, "No daily data", ha="center", va="center", transform=ax7.transAxes)
        else:
            ax7.text(0.5, 0.5, "No daily data", ha="center", va="center", transform=ax7.transAxes)

        # Adjust layout - use try-except to handle tight_layout issues
        try:
            plt.tight_layout()
        except:
            # If tight_layout fails, use subplots_adjust
            plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, hspace=0.4, wspace=0.4)

        # saving
        filename = self.output_dir / f'dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        plt.show()
        return filename

# quick functions for easy access
def plot_current(data):
    vis = AirQualityVisualizer()
    return vis.plot_current_metrics(data)


def plot_trend(df):
    vis = AirQualityVisualizer()
    return vis.plot_24h_trend(df)


def plot_weekly(df):
    vis = AirQualityVisualizer()
    return vis.plot_weekly_stats(df)

if __name__ == "__main__":
    print("Testing visualizer")

    # creating test data
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

    # creating test dataframe
    dates = pd.date_range(end=datetime.now(), periods=48, freq="H")
    historical = pd.DataFrame({
        "timestamp": dates,
        "pm25": np.random.lognormal(mean=3.0, sigma=0.5, size=48),
        "pm10": np.random.lognormal(mean=3.5, sigma=0.5, size=48),
        "temperature": np.random.normal(loc=12, scale=3, size=48),
        "humidity": np.random.normal(loc=70, scale=10, size=48),
        "aqi_category": np.random.choice(["good", "moderate", "unhealthy_sensitive"], size=48)
    })

    # testing functions
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

