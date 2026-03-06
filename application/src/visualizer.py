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

            # temperature and humidity
            ax2 = axis[1]
            ax2_twin = ax2.twinx()
            if "temperature" in df.columns:
                line1 = ax2.plot(df["datetime"], df["temperature"], "o-", color = COLORS['temperature'],
                                 label="temperature", linewidth = 2, markersize = 4)
                ax2.set_ylabel("temperature (°C)", color= COLORS["temperature"])
                ax2.tick_params(axis='y', labelcolor=COLORS["temperature"])
            if "humidity" in df.columns:
                line2 = ax2_twin.plot(df["datetime"], df["humidity"], "s-", color = COLORS['humidity'],
                                      label="humidity", linewidth = 2, markersize = 4)
                ax2_twin.set_ylabel("humidity (%)", color = COLORS["humidity"])
                ax2_twin.tick_params(axis='y', labelcolor=COLORS["humidity"])
            ax2.set_title("weather conditions")
            ax2.grid(True, alpha = 0.3)

            # connecting the legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legends(lines, labels, loc = "upper right")
            ax3 = axis[2]
            if "aqi_category" in df.columns:
                # turning categories into numbers for graphs
                category_map = {"good": 1, "moderate": 2, "unhealthy_sensitive": 3, "unhealthy": 4, "hazardous": 5}
                df["aqi_level"] = df["aqi_category"].map(category_map)

                # colors for dots
                colors = [COLORS.get(cat, "grey") for cat in df["aqi_category"]]
                scatter = ax3.scatter(df["datetime"], df["aqi_level"], c = colors, s = 50, alpha = 0.7)
                ax3.plot(df["datetime"], df["aqi_level"], "grey", alpha = 0.3, linewidth = 1)
                ax3.set_yticks([1,2,3,4,5])
                ax3.set_ylabel("quality category")
                ax3.title("changes of air quality category")
                ax3.grid(True, alpha = 0.3)

                # forming time axis
                for ax in axis:
                    ax.xaxis.set_major_formatter(plt.DateFormatter('%H:%M'))
                    ax.set_xlabel("time")
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                plt.tight_layout()

                # saving
                filename = self.output_dir /  f'trend_24h_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                plt.savefig(filename, dpi = 100, bbox_inches = "tight")
                plt.show()
                return filename

    def plot_weekly_states(self, df, title = "weekly statics of air quality"):
        fig, axis = plt.subplots(2,2, figsize = (14,10))
        fig.suptitle(title, fontsize = 14, fontweight = 'bold')

        # adding day of week
        df["weekday"] = pd.to_datetime(df["timestamp"]).dt.day_name()
        week_day_order = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

        # average pm2.5 for week days
        ax1 = axis[0,0]
        week_day_pm25 = df.groupby("weekday")["pm25"].mean().reindex(week_day_order)
        bars = ax1.bar(range(7), week_day_pm25.values, color = "steelblue")
        ax1.set_xticks(range(7))
        ax1.set_xticklabels(["mon", "tues", "wed", "thur", "fr", "sat", "sun"])
        ax1.set_ylabel("average pm2.5 (μg/m³)")
        ax1.set_title("average pm2.5 by week days")

        # adding values
        for bar, value in zip(bars, week_day_pm25.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f"{value: .1f}", ha = "centre", va = "bottom")

            # category distribution
            ax2 = axis[0,1]
            category_counts = df["aqi_category"].value_counts()
            colors = [COLORS.get(cat, "grey") for cat in category_counts.index]
            wadges, texts, autotexts = ax2.pie(category_counts.values, labels = category_counts.index, colors = colors, autopct='%1.1f%%', startangle = 90)
            ax2.set_title("quality category distribution")
            ax3 = axis[1,0]
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
            hourly_data = [df[df["hour"] == h]["pm25"].values for h in range(24)]
            box = ax3.boxplot(hourly_data, positions = range(24), patch_artist = True)
            ax3.set_xlabel("hour of the day")
            ax3.set_ylabel("pm2.5 (μg/m³)")
            ax3.set_title("pm2.5 distribution by hours")
            ax3.grid(True, alpha = 0.3)

            # coloring the box
            for box_item in box["boxes"]:
                box_item.set_facecolor("lightblue")

                # relations between parameters
                ax4 = axis[1,1]
                numeric_cols = ["pm25", "pm10", "temperature", "humidity", "wind_speed", "pressure"]
                available_cols = [col for col in numeric_cols if col in df.columns]
                if len(available_cols) > 1:
                    corr_matrix = df[available_cols].corr()
                    im = ax4.imshow(corr_matrix, cmap="coolwarm", vmin = 0, vmax = 1, aspect = "auto")
                    ax4.set_xticks(range(len(available_cols)))
                    ax4.set_yticks(range(len(available_cols)))
                    ax4.set_xticklabels(available_cols, rotation = 45, ha = "right")
                    ax4.set_yticklabels(available_cols)
                    ax4.set_title("parameter relations")

                    # adding values into cells
                    for i in range(len(available_cols)):
                        for j in range(len(available_cols)):
                            text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha='center', va='center',
                                  color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
                    plt.colorbar(im, ax = ax4)
                plt.tight_layout()

                # saving
                filename = self.output_dir / f'weekly_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                plt.savefig(filename, dpi = 100, bbox_inches = "tight")
                plt.show()
                return filename

    # comparison of current values with the norm
    def plot_comparison_with_norm(self, data, norms = None):
        if norms is None:
            # WHO norms for Prague
            norms = {
                'PM2.5': {'who': 25, 'eu': 25, 'cz': 25},
                'PM10': {'who': 50, 'eu': 50, 'cz': 50},
                'NO2': {'who': 40, 'eu': 40, 'cz': 40}
            }
        fig, ax = plt.subplots(figsize = (10,6))
        polluters = []
        current_values = []
        who_norms = []
        colours = []
        for polluter, value in [('PM2.5', data.get('pm25', 0)),
                                 ('PM10', data.get('pm10', 0)),
                                 ('NO2', data.get('no2', 0))]:
            if value > 0:
                polluters.append(polluter)
                current_values.append(value)
                who_norms.append(norms[polluter]['who'])
                colours.append("red" if value > norms[polluter]['who'] else "green")
        x = np.arange(len(polluters))
        width = 0.35

        # current value
        bars1 = ax.bar(x - width / 2, current_values, width, label = "current values", color = colours, edgecolor = "black")
        bars2 = ax.bar(x + width / 2, who_norms, width, label="WHO norms", color="grey", alpha = 0.5, edgecolor="black")
        ax.set_xlabel("polluter")
        ax.set_ylabel("concentration (μg/m³)")
        ax.set_title("comparison of current values with WHO norms")
        ax.set_xticks(x)
        ax.set_xticklabels(polluters)
        ax.legend()

        # adding the value
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f"{height: .1f}", ha = "center", va = "bottom")
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f"{height: .1f}", ha="center", va="bottom")
        ax.grid(True, alpha = 0.3, axis = "y")
        plt.tight_layout()

        # saving
        filename = self.output_dir / f'compression_norms_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.show()
        return filename

    # creates a full dashboard with a few graphs
    def create_dashboard(self, current_data, historical_df):
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize = (16,12))
        fig.suptitle(f"air quality dashboard - Prague\n{datetime.now().strftime("%Y-%m-%d %H:%M")}", fontsize = 16, fontweight = "bold")

        # creating a grid for the graph
        gs = GridSpec(3,3, figure = fig, hspace= 0.3, wspace = 0.3)

        # current AQI (speedometer)
        ax1 = fig.add_subplot(gs[0,0])
        self._create_gauge_chart(ax1, current_data.get("aqi_score", 0), current_data.get("aqi_category", "unknown"))

        # current polluters
        ax2 = fig.add_subplot(gs[0,1])
        polluters = ["PM2.5", "PM10", "NO₂"]
        values = [current_data.get("pm25", 0), current_data.get("pm10", 0), current_data.get("no2", 0)]
        ax2.bar(polluters, values, color = ['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title("polluters")
        ax2.set_ylabel("(μg/m³)")

        # 24h trend
        ax3 = fig.add_subplot(gs[0,2])
        if len(historical_df) > 0:
            recent = historical_df.tail(24)
            ax3.plot(recent["pm25"].values, color = COLORS["pm25"], label = "PM2.5")
            ax3.plot(recent["pm10"].values, color=COLORS["pm10"], label="PM10")
            ax3.set_title("24h trend")
            ax3.legend(fontsize = 8)

            # category distribution
            ax4 = fig.add_subplot(gs[1,0])
            if "aqi_category" in historical_df.columns:
                cat_counts = historical_df["aqi_category"].value_counts()
                colours = [COLORS.get(cat,"grey") for cat in cat_counts.index]
                ax4.pie(cat_counts.values, labels = cat_counts.index, colors = colours, autopct='%1.1f%%')
                ax4.set_title("category distribution")

            # hourly statistics
            ax5 = fig.add_subplot(gs[1,1])
            historical_df["hour"] = pd.to_datetime(historical_df["timestamp"]).dt.hour
            hourly_avg = historical_df.groupby("hour")["pm25"].mean()
            ax5.plot(hourly_avg.index, hourly_avg.values, "o-", color = COLORS["pm25"])
            ax5.set_title("average PM2.5 by hour")
            ax5.set_xlabel("hour")
            ax5.set_ylabel("PM2.5")
            ax5.grid(True, alpha = 0.3)

            # weather
            ax6 = fig.add_subplot(gs[1,2])
            weather_data = [current_data.get("temperature", 0), current_data.get("humidity", 0), current_data.get("wind_speed", 0), current_data.get("pressure", 1013) / 10]
            weather_labels = ['temperature (°C)', 'humidity (%)', 'wind (m/s)', 'pressure (×10 hPa)']
            ax6.bar(weather_labels, weather_data, color = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd'])
            ax6.set_title("weather conditions")
            ax6.tick_params(axis = "x", rotation = 45)

            # weekly statistics
            ax7 = fig.add_subplot(gs[2,:])
            historical_df["data"] = pd.to_datetime(historical_df["timestamp"]).dt.date
            daily_avg = historical_df.groupby("data")["pm25", "pm10"].mean().tail(7)
            x = range(len(daily_avg))
            ax7.plot(x, daily_avg["pm25"].values, "o-", color = COLORS["pm25"], label = "PM2.5", linewidth = 2)
            ax7.plot(x, daily_avg["pm10"].values, "s-", color=COLORS["pm10"], label="PM10", linewidth=2)
            ax7.set_title("daily average for the last 7 days")
            ax7.set_xlabel("day")
            ax7.set_ylabel("concentration (μg/m³)")
            ax7.set_xticks(x)
            ax7.set_xticklabels([d.strftime("%d.%m") for d in daily_avg.index])
            ax7.legend()
            ax7.grid(True, alpha = 0.3)
            plt.tight_layout()

            # saving
            filename = self.output_dir / f'dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=120, bbox_inches="tight")
            plt.show()
            return filename

# quick graph for current data
def plot_current(data):
    vis = AirQualityVisualizer()
    return vis.plot_current_metrics(data)

# quick trend graph
def plot_trend(df):
    vis = AirQualityVisualizer()
    return vis.plot_24h_trend(df)

# quick weekly statistics
def plot_weekly(df):
    vis = AirQualityVisualizer()
    return vis.plot_weekly_states(df)

if __name__ == "__main__":
    print("testing visualizer")

    # creating test data
    current = {'timestamp': datetime.now().isoformat(),
        'pm25': 35.5,
        'pm10': 48.2,
        'no2': 42.1,
        'temperature': 12.5,
        'humidity': 78,
        'wind_speed': 3.2,
        'pressure': 1012,
        'aqi_score': 85,
        'aqi_category': 'moderate'}

    # creating test dataframe
    dates = pd.date_range(end = datetime.now(), periods = 48, freq="H")
    historical = pd.DataFrame({
        "timestamp": dates,
        "pm25": np.random.lognormal(mean = 3.0, sigma = 0.5, size = 48),
        "pm10": np.random.lognormal(mean = 3.5, sigma = 0.5, size = 48),
        "temperature": np.random.normal(loc=12, scale=3, size = 48),
        "humidity": np.random.normal(loc=70, scale=10, size = 48),
        "aqi_category": np.random.choice(["good", "moderate", "unhealthy_sensitive"], size = 48)
    })

    # testing functions
    vis = AirQualityVisualizer()
    print("Creating a graph of current metrics")
    vis.plot_current_metrics(current)
    print("Creating a graph of 24h trend")
    vis.plot_24h_trend(historical.tail(24))
    print("Creating weekly statistics")
    vis.plot_weekly_states(historical)
    print("Creating a dashboard")
    vis.create_dashboard(current,historical)
    print(f"All files saved to folder: {vis.output_dir}")

