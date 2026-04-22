import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent

DATA_PATH = PROJECT_ROOT / "data_collection" / "historical_data.csv"
MODEL_PATH = CURRENT_DIR / "trained_model.pkl"
COMPACT_MODEL_PATH = CURRENT_DIR / "trained_model_compact.pkl"
VIS_DIR = CURRENT_DIR / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("TRAINING MODEL FOR AIR QUALITY PREDICTION")
print("=" * 80)
print(f"Time: {datetime.now()}")
print(f"Data: {DATA_PATH}")

if not DATA_PATH.exists():
    print("ERROR: dataset not found")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)

print(f"Shape: {df.shape}")

if "aqi_category" not in df.columns:
    print("ERROR: missing aqi_category column")
    sys.exit(1)

df["aqi_category"] = df["aqi_category"].fillna("unknown")

if df["aqi_category"].nunique() < 2:
    print("WARNING: Only one class in dataset → model cannot learn properly")

df_clean = df.copy()

required_cols = ["temperature", "humidity", "pressure", "wind_speed", "hour", "day_of_week"]
df_clean = df_clean.dropna(subset=[c for c in required_cols if c in df_clean.columns])

print(f"After cleaning: {len(df_clean)} rows")

if len(df_clean) < 10:
    print("ERROR: too little data after cleaning")
    sys.exit(1)

df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
df_clean = df_clean.sort_values("timestamp")

df_clean["aqi_category_next"] = df_clean["aqi_category"].shift(-1)
df_clean = df_clean.dropna(subset=["aqi_category_next"])

le = LabelEncoder()
df_clean["weather_main_encoded"] = le.fit_transform(
    df_clean["weather_main"].fillna("Clear")
)

df_clean["hour_sin"] = np.sin(2 * np.pi * df_clean["hour"] / 24)
df_clean["hour_cos"] = np.cos(2 * np.pi * df_clean["hour"] / 24)

df_clean["day_sin"] = np.sin(2 * np.pi * df_clean["day_of_week"] / 7)
df_clean["day_cos"] = np.cos(2 * np.pi * df_clean["day_of_week"] / 7)

features = [
    "temperature", "humidity", "pressure", "wind_speed",
    "clouds",
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "weather_main_encoded",
    "is_weekend"
]

features = [f for f in features if f in df_clean.columns]

X = df_clean[features]
y = df_clean["aqi_category_next"]

print("\nTarget distribution:")
print(y.value_counts())

if len(X) == 0:
    print("ERROR: no usable data")
    sys.exit(1)

if y.nunique() < 2:
    print("ERROR: only one class → cannot train model")
    sys.exit(1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig(VIS_DIR / "confusion_matrix.png")
plt.close()

fi = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(fi["feature"], fi["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.savefig(VIS_DIR / "feature_importance.png")
plt.close()

model_data = {
    "model": model,
    "scaler": scaler,
    "label_encoder": le,
    "features": features,
    "classes": model.classes_.tolist(),
    "metadata": {
        "accuracy": accuracy_score(y_test, y_pred),
        "samples": len(X_train)
    }
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)

print("\nMODEL SAVED SUCCESSFULLY")
