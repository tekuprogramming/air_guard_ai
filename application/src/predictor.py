# predictor.py - module for loading models and predicting the weather

import pickle
import numpy as np
from pathlib import Path

class AirQualityPredictor:
    def __init__(self, model_path = ""):
        model_path = Path(__file__).parent / model_path
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.label_encoder = model_data["label_encoder"]
            self.features = model_data["features"]
            self.classes = model_data["classes"]
            print("Model loaded", model_path)
            print(f"  Classes: {', '.join(self.classes)}")
            print(f"  Features: {len(self.features)}")
        except FileNotFoundError:
            print("Model was not found. Please check the path and try again.", model_path)
            print("Complete model training first")
            self.model = None

    # predicts the category of air quality
    def predict(self, features_dict): # features_dict = features dictionary
        if self.model is None:
            return {"category": "Unknown", "confidence": None}
        # vector for features in the right order
        feature_vector = []
        for feature in self.features:
            value = features_dict.get(feature,0)
            feature_vector.append(value)
        # changing into numpy
        x = np.array(feature_vector).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        # prediction
        prediction_idx = self.model.predict(x_scaled)[0]
        probabilities = self.model.predict_proba(x_scaled)[0]

        # changing index into category
        category = self.label_encoder.inverse_transform([prediction_idx])[0]
        confidence = np.max(probabilities)
        return {"category": category, "confidence": confidence, "probabilities": dict(zip(self.classes, probabilities))}
