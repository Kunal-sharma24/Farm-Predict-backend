import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pickle
import os

# -------------------------------
# Model Definitions (same as yours)
# -------------------------------

class DNFNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_trees, n_classes):
        super().__init__()
        self.trees = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(n_trees)])
        self.output = nn.Linear(hidden_dim * n_trees, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        tree_outputs = [F.relu(tree(x)) for tree in self.trees]
        h = torch.cat(tree_outputs, dim=-1)
        return self.output(self.dropout(h))


class AutoInt(nn.Module):
    def __init__(self, in_dim, n_heads, n_layers, embed_dim, n_classes):
        super().__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)
        for attn in self.attn_layers:
            x, _ = attn(x, x, x)
        return self.fc(x.squeeze(0))


class GrowNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_stages, n_classes):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim + (n_classes if i > 0 else 0), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_classes)
            ) for i in range(n_stages)
        ])

    def forward(self, x):
        prev = None
        outputs = []
        for i, stage in enumerate(self.stages):
            out = stage(x if i == 0 else torch.cat([x, prev], dim=-1))
            outputs.append(out)
            prev = out
        return sum(outputs) / len(outputs)


class SAINT(nn.Module):
    def __init__(self, in_dim, embed_dim, n_heads, n_layers, n_classes):
        super().__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)
        for attn in self.attn:
            x, _ = attn(x, x, x)
        return self.fc(x.squeeze(0))


class NAM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.feature_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(in_dim)
        ])
        self.output = nn.Linear(hidden_dim * in_dim, n_classes)

    def forward(self, x):
        feats = [net(x[:, i:i+1]) for i, net in enumerate(self.feature_nets)]
        return self.output(torch.cat(feats, dim=-1))


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
    }