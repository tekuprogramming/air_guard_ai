import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score



plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")



CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_PATH = PROJECT_ROOT / "data_collection" / 'historical_data.csv'
MODEL_PATH = CURRENT_DIR / 'trained_model.pkl'
COMPACT_MODEL_PATH = CURRENT_DIR / 'trained_model_compact.pkl'
REPORT_PATH = CURRENT_DIR / 'training_report.txt'
VIS_DIR = CURRENT_DIR / 'visualizations'

VIS_DIR.mkdir(exist_ok=True)



print("=" * 80)
print("TRAINING MODEL FOR AIR QUALITY PREDICTION (PRAGUE)")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset: {DATA_PATH}")
print(f"Model output: {MODEL_PATH}")
print("=" * 80)

print("Loading dataset...")
print("-" * 40)


if not DATA_PATH.exists():
    print("ERROR: Data file not found:", DATA_PATH)
    print("Run data collection first.")
    sys.exit(1)



df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"Records: {len(df)}")
print(f"Time range: {df['date'].min()} - {df['date'].max()}")



if 'aqi_category' in df.columns:
    unique_vals = df['aqi_category'].unique()
    print(f"AQI categories found: {unique_vals}")

    if all(val in ['Unknown', 'unknown', None] for val in unique_vals) or df['aqi_category'].isna().all():
        print("AQI missing → generating from PM2.5 values")

        # simple rule-based AQI mapping
        def calc_cat(pm25):
            if pd.isna(pm25):
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

        df['aqi_category'] = df['pm25'].apply(calc_cat)

        print("AQI distribution after filling:")
        print(df['aqi_category'].value_counts())



print("\nFirst rows:")
print(df.head())

print("\nData types:")
print(df.dtypes.head(10))



print("\nMissing values analysis")
print("-" * 40)

missing = df.isnull().sum()
missing_cols = missing[missing > 0]

if len(missing_cols) > 0:
    print("Missing values detected:")
    for col, count in missing_cols.items():
        pct = (count / len(df)) * 100
        print(f"{col}: {count} ({pct:.1f}%)")
else:
    print("No missing values found")


initial_rows = len(df)

df_clean = df.dropna(subset=['pm25', 'pm10'])

print("\nRemoving invalid rows:")
print(f"Before: {initial_rows}")
print(f"After: {len(df_clean)}")
print(f"Removed: {initial_rows - len(df_clean)}")


if "temperature" in df_clean.columns:
    df_clean = df_clean[(df_clean["temperature"] >= -25) & (df_clean["temperature"] <= 38)]

if "pm25" in df_clean.columns:
    df_clean = df_clean[(df_clean["pm25"] >= 0) & (df_clean["pm25"] <= 500)]

if "pm10" in df_clean.columns:
    df_clean = df_clean[(df_clean["pm10"] >= 0) & (df_clean["pm10"] <= 600)]


numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method="linear", limit_direction="both")

print(f"Final dataset size: {len(df_clean)}")


df_clean.to_csv(CURRENT_DIR / 'historical_data_clean.csv', index=False)
print("Clean dataset saved")



print("\nDataset statistics:")
print(df_clean[numeric_cols].describe().round(2))



print("\nFeature engineering")

df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
df_clean = df_clean.sort_values("timestamp")

df_clean["aqi_category_next"] = df_clean["aqi_category"].shift(-1)

df_clean = df_clean.iloc[:-1]

label_encoder = LabelEncoder()
df_clean["weather_main_encoded"] = label_encoder.fit_transform(
    df_clean["weather_main"].fillna("Clear")
)

df_clean["hour_sin"] = np.sin(2 * np.pi * df_clean["hour"] / 24)
df_clean["hour_cos"] = np.cos(2 * np.pi * df_clean["hour"] / 24)

df_clean["day_sin"] = np.sin(2 * np.pi * df_clean["day_of_week"] / 7)
df_clean["day_cos"] = np.cos(2 * np.pi * df_clean["day_of_week"] / 7)



features = [
    "temperature", "humidity", "pressure", "wind_speed", "clouds",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "weather_main_encoded", "is_weekend"
]

X = df_clean[features]
y = df_clean["aqi_category_next"]


if len(X) == 0:
    print("ERROR: No training data available")
    sys.exit(1)


print("Target distribution:")
print(y.value_counts())



X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)



y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig(VIS_DIR / "confusion_matrix.png")
plt.close()


feature_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance["feature"], feature_importance["importance"])
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.savefig(VIS_DIR / "feature_importance.png")
plt.close()



model_data = {
    "model": model,
    "scaler": scaler,
    "label_encoder": label_encoder,
    "features": features,
    "classes": model.classes_.tolist(),
    "feature_importance": feature_importance.set_index("feature")["importance"].to_dict(),
    "metadata": {
        "training_date": datetime.now().isoformat(),
        "accuracy": accuracy_score(y_test, y_pred),
        "n_samples": len(X_train)
    }
}


with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)


compact_data = {k: model_data[k] for k in ["model", "scaler", "label_encoder", "features", "classes"]}

with open(COMPACT_MODEL_PATH, "wb") as f:
    pickle.dump(compact_data, f)

print("Model saved successfully")
