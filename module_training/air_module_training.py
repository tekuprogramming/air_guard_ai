import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
import json
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# training model to predict air quality

# setting the looks
plt.style.use('seaborn-v0_8_darkgrid')
sns.set_palette("husl")

# setting the root folder of the project
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent

# data paths
DATA_PATH = CURRENT_DIR / 'historical_data.csv'
MODELS_PATH = CURRENT_DIR / 'trained_model.pkl'
REPORT_PATH = CURRENT_DIR / 'training_report.txt'
VIS_PATH = CURRENT_DIR / 'visualization.png'

VIS_PATH.mkdir(exist_ok=True)
print("=" * 80)
print("TRAINING MODEL FOR PREDICTING AIR QUALITY IN PRAGUE")
print("=" * 80)
print(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Data file: {DATA_PATH}")
print(f"Model will be saved in: {MODELS_PATH}")
print("=" * 80)
print("Loading data")
print("-" * 40)

# checking the existence of the file
if not DATA_PATH.exists():
    print("Error: Data file not found", DATA_PATH)
    print("First start the data collection: data_collection/collector.py")
    sys.exit(1)

# loading data
df = pd.read_csv(DATA_PATH)
print("Data loaded successfully")
print(f"  - Size: {df.shape}")
print(f"  - Columns: {len(df.columns)}")
print(f"  - Records: {len(df)}")
print(f"  - Period: {df['date'].min()} - {df['date'].max()}")

# basic information
print("\nFirst five records: ")
print(df.head().to_string())
print("\nData types: ")
print(df.dtypes.head(10))

# checking for missing columns
print("\nAnalysis and cleaning of the data")
print("-" * 40)
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print(f"Missing columns found")
    for col, count in missing_cols.items():
        percentage = (count / len(df)) * 100
        print(f"    {col}: {count} missing ({percentage:.1f}%)")
else:
    print("No missing columns found")

# deleting rows without pm25 and pm10
initial_rows = len(df)
df_clean = df.dropna(subset=['pm25', 'pm10'])
print(f"\n Rows without PM2.5/PM10 were deleted:")
print(f"    Before: {initial_rows} records")
print(f"    After: {len(df_clean)} records")
print(f"    Deleted: {initial_rows - len(df_clean)} records")

# filtrating the dumps
if "temperature" in df_clean.columns:
    df_clean = df_clean[(df_clean["temperature"] >= -25) & (df_clean["temperature"] <= 38)]
    print("Filtrating temperature from -25 to 38")
if "pm25" in df_clean.columns:
    df_clean = df_clean[(df_clean["pm25"] >= 0) & (df_clean["pm25"] <= 500)]
    print(f" Filtrating PM2.5: 0-500 μg/m³")
if "pm10" in df_clean.columns:
    df_clean = df_clean[(df_clean["pm10"] >= 0) & (df_clean["pm10"] <= 600)]
    print(f" Filtrating PM10: 0-600 μg/m³")

# interpolation for other missing places
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method="linear", limit_direction="both")
print(f"\nFinal size after cleaning: {len(df_clean)} records")

df_clean.to_csv(CURRENT_DIR / 'historical_data_clean.csv', index=False)
print("Clean data saved: historical_data_clean.csv")

# statistics of clean data
print("\nStatistics of number signs")
print(df_clean[numeric_cols].describe().round(2).to_string())

# distribution visualization
fig, axis = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Distribution of key signs after cleaning", fontsize = 16)

# PM2.5
axis[0,0].hist(df_clean["pm25"].dropna(), bins = 50, color = "skyblue", edgecolor = "black")
axis[0,0].set_title("PM2.5 distribution")
axis[0,0].set_xlabel("PM2.5 (μg/m³)")
axis[0,0].set_ylabel("Frequency")

# PM10
axis[0,1].hist(df_clean["pm10"].dropna(), bins = 50, color = "lightgreen", edgecolor = "black")
axis[0,1].set_title("PM10 distribution")
axis[0,1].set_xlabel("PM10 (μg/m³)")

# temperature
axis[0,2].hist(df_clean["temperature"].dropna(), bins = 50, color = "salmon", edgecolor = "black")
axis[0,2].set_title("Temperature distribution")
axis[0,2].set_xlabel("Temperature (°C)")

# humidity
axis[1,0].hist(df_clean["humidity"].dropna(), bins = 50, color = "lightblue", edgecolor = "black")
axis[1,0].set_title("Humidity distribution")
axis[1,0].set_xlabel("Humidity (%")

# wind speed
axis[1,1].hist(df_clean["wind_speed"].dropna(), bins = 50, color = "orange", edgecolor = "black")
axis[1,1].set_title("Wind speed distribution")
axis[1,1].set_xlabel("Wind speed (m/s)")

# AQI categories
if "aqi_categories" in df_clean.columns:
    category_counts = df_clean["aqi_categories"].value_counts()
    axis[1,2].bar(category_counts.index, category_counts.values, color = "purple")
    axis[1,2].set_title("AQI categories")
    axis[1,2].set_xlabel("Category")
    axis[1,2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(VIS_PATH/"data_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Visualization saved: {VIS_PATH/'data_distribution.png'}")

print("Engineering signs")
print("-" * 40)

# time sorting
df_clean = df_clean.sort_values("timestamp")

# target variable (quality category for the next hour)
df_clean["aqi_category_next"] = df_clean["aqi_category"].shift(-1)
print("Target variable created: aqi_category_next")

# deleting the last row
df_clean = df_clean.iloc[:-1]

# coding category signs
label_encoder = LabelEncoder()
df_clean ["weather_main_encoded"] = label_encoder.fit_transform(df_clean["weather_main"])

# creating time signs
df_clean ["hour_sin"] = np.sin(2 * np.pi * df_clean["hour"]/24)
df_clean ["hour_cos"] = np.cos(2 * np.pi * df_clean["hour"]/24)
df_clean ["day_sin"] = np.sin(2 * np.pi * df_clean["day_of_week"]/7)
df_clean ["day_cos"] = np.cos(2 * np.pi * df_clean["day_of_week"]/7)

# choosing features for models
features = ["temperature", "humidity", "pressure", "clouds", "hour_sin", "hour_cos", "day_sin", "day_cos", "weather_main_encoded", "is_weekend", "pm25", "pm10"]
x = df_clean[features]
y = df_clean["aqi_category_next"]

# separating train / test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, shuffle = False)

# scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(f"Training data shape: {x_train_scaled.shape}")
print(f"Testing data shape: {x_test_scaled.shape}")

# teaching model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# teaching random forest
model = RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state=42, n_jobs = -1)
model.fit(x_train_scaled, y_train)

# prediction
y_pred = model.predict(x_test_scaled)

# model grade
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# error matrix
cn = confusion_matrix(y_test, y_pred, labels = model.classes_)
plt.figure(figsize = (8,6))
sns.heatmap(cn, annot=True, fmt="d", cmap="Blues", xticklabels = model.classes_, yticklabels = model.classes_)
plt.title("Confusion matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()

# sign importance
feature_importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
plt.figure(figsize = (10,6))
plt.barh(feature_importance["feature"], feature_importance["importance"])
plt.xlabel("Importance")
plt.title("Feature importance")
plt.gca().invert_yaxis()
plt.show()

# saving model
import joblib
import pickle

# saving model and scaler
model_data = {"model": model, "scaler": scaler, "label_encoder": label_encoder, "features": features, "classes": model.classes_.tolist()}

# saving to file
with open("air_quality_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
