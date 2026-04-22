import os
import pickle
import numpy as np
import pandas as pd

from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabular import TabularModel

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "irrigation_model_files")

# ---------------- FEATURE CONFIG ----------------
TABNET_FEATURE_COLS = [
    'min_temperature', 'max_temperature',
    'crop_type_encoded', 'soil_type_encoded', 'region_encoded', 'weather_encoded',
    'SOIL TYPE_DRY', 'SOIL TYPE_HUMID', 'SOIL TYPE_WET',
    'REGION_DESERT', 'REGION_HUMID', 'REGION_SEMI ARID', 'REGION_SEMI HUMID',
    'WEATHER CONDITION_NORMAL', 'WEATHER CONDITION_RAINY',
    'WEATHER CONDITION_SUNNY', 'WEATHER CONDITION_WINDY',
    'CROP TYPE_BANANA', 'CROP TYPE_BEAN', 'CROP TYPE_CABBAGE', 'CROP TYPE_CITRUS',
    'CROP TYPE_COTTON', 'CROP TYPE_MAIZE', 'CROP TYPE_MELON', 'CROP TYPE_MUSTARD',
    'CROP TYPE_ONION', 'CROP TYPE_POTATO', 'CROP TYPE_RICE', 'CROP TYPE_SOYABEAN',
    'CROP TYPE_SUGARCANE', 'CROP TYPE_TOMATO', 'CROP TYPE_WHEAT'
]

FTTRANSFORMER_NUMERICAL_COLS = ['min_temperature', 'max_temperature']
FTTRANSFORMER_CATEGORICAL_COLS = [
    'TEMPERATURE', 'crop_type_encoded', 'soil_type_encoded', 'region_encoded', 'weather_encoded',
    'SOIL TYPE_DRY', 'SOIL TYPE_HUMID', 'SOIL TYPE_WET',
    'REGION_DESERT', 'REGION_HUMID', 'REGION_SEMI ARID', 'REGION_SEMI HUMID',
    'WEATHER CONDITION_NORMAL', 'WEATHER CONDITION_RAINY',
    'WEATHER CONDITION_SUNNY', 'WEATHER CONDITION_WINDY',
    'CROP TYPE_BANANA', 'CROP TYPE_BEAN', 'CROP TYPE_CABBAGE', 'CROP TYPE_CITRUS',
    'CROP TYPE_COTTON', 'CROP TYPE_MAIZE', 'CROP TYPE_MELON', 'CROP TYPE_MUSTARD',
    'CROP TYPE_ONION', 'CROP TYPE_POTATO', 'CROP TYPE_RICE', 'CROP TYPE_SOYABEAN',
    'CROP TYPE_SUGARCANE', 'CROP TYPE_TOMATO', 'CROP TYPE_WHEAT'
]

# ---------mapping----------
CROP_MAP = {
    "banana": 0, "bean": 1, "cabbage": 2, "citrus": 3,
    "cotton": 4, "maize": 5, "melon": 6, "mustard": 7,
    "onion": 8, "potato": 9, "rice": 10, "soyabean": 11,
    "sugarcane": 12, "tomato": 13, "wheat": 14
}

SOIL_MAP = {"dry": 0, "humid": 1, "wet": 2}

REGION_MAP = {"desert": 0, "humid": 1, "semi arid": 2, "semi humid": 3}

WEATHER_MAP = {"normal": 0, "rainy": 1, "sunny": 2, "windy": 3}

#------------prepossessing---------------
def preprocess_input(data):
    processed = {}

    # numerical
    processed["min_temperature"] = float(data["min_temperature"])
    processed["max_temperature"] = float(data["max_temperature"])

    # encoded values
    processed["crop_type_encoded"] = CROP_MAP[data["crop_type"].lower()]
    processed["soil_type_encoded"] = SOIL_MAP[data["soil_type"].lower()]
    processed["region_encoded"] = REGION_MAP[data["region"].lower()]
    processed["weather_encoded"] = WEATHER_MAP[data["weather"].lower()]

    # TEMPERATURE category (simple logic)
    avg_temp = (processed["min_temperature"] + processed["max_temperature"]) / 2
    if avg_temp < 20:
        processed["TEMPERATURE"] = "LOW"
    elif avg_temp < 30:
        processed["TEMPERATURE"] = "NORMAL"
    else:
        processed["TEMPERATURE"] = "HIGH"

    # one-hot encoding (initialize all 0)
    for col in TABNET_FEATURE_COLS:
        if col not in processed:
            processed[col] = 0

    # set correct one-hot values
    processed[f"SOIL TYPE_{data['soil_type'].upper()}"] = 1
    processed[f"REGION_{data['region'].upper()}"] = 1
    processed[f"WEATHER CONDITION_{data['weather'].upper()}"] = 1
    processed[f"CROP TYPE_{data['crop_type'].upper()}"] = 1

    return processed

# ---------------- LOAD FILES ----------------
with open(os.path.join(MODEL_DIR, "tabnet_scaler.pkl"), "rb") as f:
    tabnet_scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "irrigation_model_results.pkl"), "rb") as f:
    model_results = pickle.load(f)

# ---------------- LOAD MODELS ----------------
def load_models():
    tabnet_model = TabNetRegressor()
    tabnet_model.load_model(os.path.join(MODEL_DIR, "tabnet_irrigation_model.zip"))

    ft_model = TabularModel.load_model(
        os.path.join(MODEL_DIR, "ft_transformer_water_requirement_model")
        )

    return tabnet_model, ft_model

tabnet_model, ft_model = load_models()

# ---------------- PREDICTION FUNCTIONS ----------------
def predict_tabnet(data_dict):
    input_values = [float(data_dict[f]) for f in TABNET_FEATURE_COLS]
    X = np.array([input_values], dtype=np.float32)

    X[:, [0, 1]] = tabnet_scaler.transform(X[:, [0, 1]])
    pred = tabnet_model.predict(X)

    return float(pred[0][0])


def predict_fttransformer(data_dict):
    row = {
        col: data_dict.get(col)
        for col in FTTRANSFORMER_NUMERICAL_COLS + FTTRANSFORMER_CATEGORICAL_COLS
    }

    row['TEMPERATURE'] = str(data_dict.get('TEMPERATURE', ''))

    for col in FTTRANSFORMER_CATEGORICAL_COLS:
        if col != 'TEMPERATURE':
            row[col] = int(row[col]) if row[col] not in [None, ""] else 0

    for col in FTTRANSFORMER_NUMERICAL_COLS:
        row[col] = float(row[col]) if row[col] not in [None, ""] else 0.0

    input_df = pd.DataFrame([row])
    pred_df = ft_model.predict(input_df)

    return float(pred_df.iloc[0][0])

# ---------------- MAIN FUNCTION (FOR app.py) ----------------
def irrigation_predict_logic(data):
    try:
        # -------- Validate SIMPLE input --------
        required_fields = [
            "crop_type", "soil_type", "region", "weather",
            "min_temperature", "max_temperature"
        ]

        missing = [f for f in required_fields if f not in data]
        if missing:
            return {
                "error": f"Missing required fields: {missing}",
                "required_fields": required_fields
            }, 400

        # -------- PREPROCESS --------
        processed_data = preprocess_input(data)

        predictions = {}
        errors = {}

        # -------- TabNet --------
        try:
            predictions['TabNet'] = predict_tabnet(processed_data)
        except Exception as e:
            errors['TabNet'] = str(e)

        # -------- FTTransformer --------
        try:
            predictions['FTTransformer'] = predict_fttransformer(processed_data)
        except Exception as e:
            errors['FTTransformer'] = str(e)

        if not predictions:
            return {
                "error": "All models failed",
                "details": errors
            }, 500

        # -------- Best model --------
        best_model_name = max(
            predictions,
            key=lambda m: model_results.get(m, {}).get('r2', -999)
        )

        best_prediction = predictions[best_model_name]
        best_metrics = model_results.get(best_model_name, {})

        # -------- All models (for frontend chart) --------
        all_models_data = []
        for model_name in ['TabNet', 'FTTransformer']:
            metrics = model_results.get(model_name, {})
            all_models_data.append({
                "name": model_name,
                "predicted_water_requirement": round(predictions.get(model_name, 0), 4),
                "mae": round(metrics.get('mae', 0), 4),
                "mse": round(metrics.get('mse', 0), 4),
                "r2": round(metrics.get('r2', 0), 4),
            })

        return {
            "best_model": best_model_name,
            "predicted_water_requirement": round(best_prediction, 4),
            "unit": "mm/day",
            "metrics": {
                "mae": round(best_metrics.get('mae', 0), 4),
                "mse": round(best_metrics.get('mse', 0), 4),
                "r2": round(best_metrics.get('r2', 0), 4),
            },
            "all_models": all_models_data,
            "model_errors": errors if errors else None
        }, 200

    except Exception as e:
        return {"error": str(e)}, 500