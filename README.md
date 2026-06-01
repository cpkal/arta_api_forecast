# UMKM Cashflow Forecast — Inference Service

API untuk prediksi arus kas harian UMKM menggunakan model LSTM yang sudah ditraining per sektor.

---

## Struktur Folder

```
project/
├── main.py
├── requirements.txt
└── models/
    ├── retail_model.keras
    ├── retail_scaler.pkl
    ├── retail_metadata.json
    ├── makanan_model.keras
    ├── makanan_scaler.pkl
    └── makanan_metadata.json
```

> Naming convention: `{nama_sektor}_model.keras`, `{nama_sektor}_scaler.pkl`, `{nama_sektor}_metadata.json`  
> Tambah sektor baru tinggal taruh 3 file itu di folder `models/`, tidak perlu ubah kode.

---

## Menyiapkan File Model dari Notebook

Jalankan Cell 12 di notebook untuk export artifacts. Pastikan hasil export ada 3 file per sektor, lalu rename sesuai konvensi di atas:

| Output Notebook | Rename jadi |
|---|---|
| `best_model.keras` | `retail_model.keras` |
| `scaler.pkl` | `retail_scaler.pkl` |
| `{sector}_metadata.json` | `retail_metadata.json` |

---

## Menjalankan Lokal

**1. Buat virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Jalankan server**
```bash
uvicorn main:app --reload
```

Server berjalan di `http://localhost:8000`  
Dokumentasi interaktif: `http://localhost:8000/docs`

---

## Deploy ke Railway

**1. Buat `Dockerfile`**
```dockerfile
FROM python:3.11-slim

WORKDIR /service

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY models/ ./models/

ENV MODEL_DIR=/service/models
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
```

**2. Buat `railway.json`**
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 120,
    "restartPolicyType": "ON_FAILURE"
  }
}
```

**3. Push ke GitHub dan connect di Railway**
```bash
git init
git add main.py requirements.txt Dockerfile railway.json models/
git commit -m "init inference service"
git push
```
Lalu di Railway: **New Project → Deploy from GitHub Repo → pilih repo.**

---

## Endpoint API

### `GET /health`
Cek apakah service berjalan dan sektor apa saja yang sudah ter-load.

```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "loaded_sectors": ["retail", "makanan"]
}
```

---

### `GET /models`
Lihat konfigurasi semua sektor yang tersedia.

```bash
curl http://localhost:8000/models
```
```json
{
  "retail": {
    "window_size": 90,
    "forecast_steps": 7,
    "feature_cols": ["expense", "income", "net"],
    "target_col": "net",
    "companies": ["retail_001", "retail_002"]
  }
}
```

---

### `POST /forecast/{sector}`
Prediksi arus kas untuk satu perusahaan.

- **`{sector}`** — nama sektor, contoh: `retail`, `makanan`
- **Body** — kirim `company_id` dan `history` minimal 90 baris (sesuai `window_size` saat training)

```bash
curl -X POST http://localhost:8000/forecast/retail \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": "retail_001",
    "history": [
      {"date": "2024-01-01", "income": 5000000, "expense": 3500000, "net": 1500000},
      {"date": "2024-01-02", "income": 4800000, "expense": 3200000, "net": 1600000}
    ]
  }'
```

> ⚠️ `history` harus berisi minimal **90 baris** (nilai `window_size` di metadata).

**Response:**
```json
{
  "company_id": "retail_001",
  "sector": "retail",
  "last_known_date": "2024-04-10",
  "predictions": [
    {"date": "2024-04-11", "predicted_net": 1734500.25},
    {"date": "2024-04-12", "predicted_net": 1812000.50},
    {"date": "2024-04-13", "predicted_net": 1690000.00},
    {"date": "2024-04-14", "predicted_net": 1755000.75},
    {"date": "2024-04-15", "predicted_net": 1823000.00},
    {"date": "2024-04-16", "predicted_net": 1600000.25},
    {"date": "2024-04-17", "predicted_net": 1710000.50}
  ]
}
```

---

### `POST /models/{sector}/reload`
Hot-reload model setelah re-training tanpa restart server.

```bash
curl -X POST http://localhost:8000/models/retail/reload
```
```json
{
  "message": "Sektor 'retail' berhasil di-reload.",
  "window_size": 90,
  "forecast_steps": 7
}
```

---

## Menambah Sektor Baru

1. Training model sektor baru di notebook (ganti `SECTOR = 'jasa'`)
2. Export 3 file artifacts dari Cell 12
3. Rename dan taruh di folder `models/`:
   - `jasa_model.keras`
   - `jasa_scaler.pkl`
   - `jasa_metadata.json`
4. Restart server (atau panggil `/models/jasa/reload`)

Tidak perlu ubah `main.py` sama sekali.

---

## Variabel Environment

| Variabel | Default | Keterangan |
|---|---|---|
| `MODEL_DIR` | `./models` | Path folder berisi file model |
| `PORT` | `8000` | Port server (Railway inject otomatis) |
