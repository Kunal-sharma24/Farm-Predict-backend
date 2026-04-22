import torch
import pandas as pd
import pickle
import os
import sys
import models

from models import DNFNet, AutoInt, GrowNet, SAINT, NAM

# 🔥 critical fix
sys.modules['__main__'] = models

# -------------------------------
# Paths
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "crop_model_files")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Preprocessing
# -------------------------------

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

with open(os.path.join(MODEL_DIR, "model_results.pkl"), "rb") as f:
    model_results_list = pickle.load(f)

model_results = {name: {"accuracy": acc, "f1_score": f1} for name, acc, f1 in model_results_list}

# -------------------------------
# Load Models
# -------------------------------

model_names = ['DNF-Net', 'AutoInt', 'GrowNet', 'SAINT', 'NAM']
loaded_models = {}

for name in model_names:
    with open(os.path.join(MODEL_DIR, f"{name}_model.pkl"), "rb") as f:
        model = pickle.load(f)
    model.to(device)
    model.eval()
    loaded_models[name] = model

print("✅ Models loaded successfully")


# -------------------------------
# MAIN FUNCTION (used by app.py)
# -------------------------------

def crop_recommendation_logic(data):
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    if not isinstance(data, dict):
        raise ValueError("Input must be JSON")

    if not all(f in data for f in features):
        raise ValueError(f"Missing fields: {features}")

    df = pd.DataFrame([[data[f] for f in features]], columns=features)
    scaled = scaler.transform(df)
    tensor = torch.tensor(scaled, dtype=torch.float32).to(device)

    best_model_name = max(model_results, key=lambda x: model_results[x]["accuracy"])
    model = loaded_models[best_model_name]

    with torch.no_grad():
        out = model(tensor)
        pred = torch.argmax(out, dim=1).item()

    crop = label_encoder.inverse_transform([pred])[0]

    return {
        "model_name": best_model_name,
        "predicted_crop": crop,
        "accuracy": model_results[best_model_name]["accuracy"],
        "f1_score": model_results[best_model_name]["f1_score"]
        ,
        "all_models": [
            {
                "name": name,
                "accuracy": round(model_results[name]["accuracy"]*100,2),
                "f1_score": round(model_results[name]["f1_score"]*100,2)
            }
            for name in model_names
        ]
    }



# from pyngrok import ngrok

# ngrok.set_auth_token("3CgHetjrfcyXc8OKl4Fb9Z2y0aN_6GTYfbgLFfF9cJrfYRMmp")
# public_url = ngrok.connect(4000)
# print(public_url)