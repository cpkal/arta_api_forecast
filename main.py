"""
UMKM Cashflow Forecast — Inference Service
Jalankan: uvicorn main:app --reload

Struktur folder model (taruh semua di ./models/):
    models/
    ├── retail_model.keras
    ├── retail_scaler.pkl
    ├── retail_metadata.json
    ├── makanan_model.keras
    ├── makanan_scaler.pkl
    └── makanan_metadata.json
"""

import json
import os
from datetime import timedelta
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable

# ---------------------------------------------------------------------------
# Custom Layer — harus didefinisikan SEBELUM load_model dipanggil
# Salin persis dari notebook supaya Keras bisa deserialize model
# ---------------------------------------------------------------------------

@register_keras_serializable()
class FinancialNormalizationLayer(layers.Layer):
    """
    Custom Layer dari notebook — wajib ada di sini agar model bisa di-load.
    Melakukan Layer Normalization + scaling adaptif per fitur.
    """

    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        n_features = input_shape[-1]
        self.gamma = self.add_weight(
            name="gamma", shape=(n_features,), initializer="ones", trainable=True
        )
        self.beta = self.add_weight(
            name="beta", shape=(n_features,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
        x_norm = (x - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * x_norm + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")

# ---------------------------------------------------------------------------
# Load semua sector model saat startup
# ---------------------------------------------------------------------------

# models dict: { "retail": {"model": ..., "scaler": ..., "meta": ...}, ... }
MODELS: Dict[str, dict] = {}


def load_all_models():
    if not os.path.isdir(MODEL_DIR):
        print(f"[WARN] MODEL_DIR '{MODEL_DIR}' tidak ditemukan, tidak ada model yang di-load.")
        return

    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith("_metadata.json"):
            continue

        sector = fname.replace("_metadata.json", "")
        model_path  = os.path.join(MODEL_DIR, f"{sector}_model.keras")
        scaler_path = os.path.join(MODEL_DIR, f"{sector}_scaler.pkl")
        meta_path   = os.path.join(MODEL_DIR, fname)

        missing = [p for p in [model_path, scaler_path, meta_path] if not os.path.exists(p)]
        if missing:
            print(f"[SKIP] Sektor '{sector}': file tidak lengkap → {missing}")
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)

            # custom_objects memastikan FinancialNormalizationLayer dikenali
            MODELS[sector] = {
                "model":  keras.models.load_model(
                    model_path,
                    custom_objects={"FinancialNormalizationLayer": FinancialNormalizationLayer}
                ),
                "scaler": joblib.load(scaler_path),
                "meta":   meta,
            }
            print(f"[OK] Sektor '{sector}' berhasil di-load | "
                  f"window={meta['window_size']} | steps={meta['forecast_steps']}")
        except Exception as e:
            print(f"[ERROR] Gagal load sektor '{sector}': {e}")


load_all_models()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="UMKM Cashflow Forecast API",
    description="Multi-sector LSTM inference service untuk forecasting arus kas harian UMKM.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CashflowRow(BaseModel):
    date: str           # "YYYY-MM-DD"
    income: float
    expense: float
    net: float


class ForecastRequest(BaseModel):
    company_id: str
    history: List[CashflowRow]  # minimal window_size baris (default 90)


class DayPrediction(BaseModel):
    date: str
    predicted_net: float


class ForecastResponse(BaseModel):
    company_id: str
    sector: str
    last_known_date: str
    predictions: List[DayPrediction]


# ---------------------------------------------------------------------------
# Helper: jalankan prediksi (sama persis dengan notebook)
# ---------------------------------------------------------------------------

def predict(sector: str, request: ForecastRequest) -> ForecastResponse:
    entry          = MODELS[sector]
    model          = entry["model"]
    scaler         = entry["scaler"]
    meta           = entry["meta"]
    feature_cols   = meta["feature_cols"]
    target_col     = meta["target_col"]
    window_size    = meta["window_size"]
    forecast_steps = meta["forecast_steps"]
    target_idx     = feature_cols.index(target_col)

    # Bangun DataFrame & sort by date
    df = pd.DataFrame([r.model_dump() for r in request.history])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < window_size:
        raise HTTPException(
            status_code=422,
            detail=f"Butuh minimal {window_size} baris history, hanya ada {len(df)}."
        )

    # Scale → predict → inverse transform (mirror notebook Cell 14)
    last_window        = df[feature_cols].values[-window_size:].astype(np.float32)
    last_window_scaled = scaler.transform(last_window)

    X           = last_window_scaled[np.newaxis, ...]    # (1, window, features)
    pred_scaled = model.predict(X, verbose=0)[0]         # (forecast_steps,)

    dummy = np.zeros((forecast_steps, len(feature_cols)), dtype=np.float32)
    dummy[:, target_idx] = pred_scaled
    pred_rupiah = scaler.inverse_transform(dummy)[:, target_idx]

    last_date = df["date"].max()
    predictions = [
        DayPrediction(
            date=(last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
            predicted_net=float(pred_rupiah[i]),
        )
        for i in range(forecast_steps)
    ]

    return ForecastResponse(
        company_id=request.company_id,
        sector=sector,
        last_known_date=last_date.strftime("%Y-%m-%d"),
        predictions=predictions,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "loaded_sectors": list(MODELS.keys())}


@app.get("/models")
def list_models():
    return {
        sector: {
            "window_size":    m["meta"]["window_size"],
            "forecast_steps": m["meta"]["forecast_steps"],
            "feature_cols":   m["meta"]["feature_cols"],
            "target_col":     m["meta"]["target_col"],
            "companies":      m["meta"].get("companies", []),
        }
        for sector, m in MODELS.items()
    }


@app.post("/forecast/{sector}", response_model=ForecastResponse)
def forecast(sector: str, body: ForecastRequest):
    if sector not in MODELS:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Sektor '{sector}' tidak ditemukan.",
                    "tersedia": list(MODELS.keys())}
        )
    return predict(sector, body)


@app.post("/models/{sector}/reload")
def reload_model(sector: str):
    """Hot-reload satu sektor setelah re-training tanpa restart server."""
    model_path  = os.path.join(MODEL_DIR, f"{sector}_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{sector}_scaler.pkl")
    meta_path   = os.path.join(MODEL_DIR, f"{sector}_metadata.json")

    missing = [p for p in [model_path, scaler_path, meta_path] if not os.path.exists(p)]
    if missing:
        raise HTTPException(status_code=404, detail=f"File tidak ditemukan: {missing}")

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        MODELS[sector] = {
            "model":  keras.models.load_model(
                model_path,
                custom_objects={"FinancialNormalizationLayer": FinancialNormalizationLayer}
            ),
            "scaler": joblib.load(scaler_path),
            "meta":   meta,
        }
        return {"message": f"Sektor '{sector}' berhasil di-reload.",
                "window_size": meta["window_size"],
                "forecast_steps": meta["forecast_steps"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
