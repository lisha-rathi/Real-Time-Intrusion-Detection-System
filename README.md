# Real Time Intrusion Detection System

A Flask-based Network Intrusion Detection System (NIDS) with a professional SOC-style dashboard, real-time traffic classification, model explainability, batch CSV scanning, and persistent alert logging.

## Overview

This project simulates a real-world Security Operations Center tool. It takes network flow data, classifies it as benign or a specific attack type, maps it to the MITRE ATT&CK framework, and presents the results on a live dashboard with charts, logs, and model performance metrics.

Built using the CIC-IDS2017 dataset from the Canadian Institute for Cybersecurity.

## Features

- Real-time traffic classification using two models:
  - Random Forest — 99.97% accuracy
  - Deep Learning (Keras/TensorFlow) — 99.86% accuracy
- Multi-class attack type detection — not just attack vs benign
- MITRE ATT&CK technique mapping for every prediction
- SHAP explainability showing top contributing features per prediction
- Batch CSV upload and scan with attack type breakdown
- Persistent prediction logging in SQLite with a live alert table
- Live traffic timeline chart and attack type doughnut chart
- Model performance gauges showing accuracy, precision, recall, and F1 score for both models

## Dashboard Sections

- **Dashboard** — live stat cards, traffic classification timeline, attack type breakdown chart
- **Analyze** — real-time single-row prediction with MITRE badge and SHAP bars
- **Batch Scan** — CSV upload with visual breakdown and attack type pills
- **Alerts** — persistent log of all predictions with color-coded rows
- **Metrics** — animated gauges comparing RF and DL model performance

## Repository Structure

```
├── app/
│   ├── app.py                  Flask server and API endpoints
│   └── templates/
│       └── index.html          SOC dashboard frontend
├── data/
│   └── friday.csv              Sample dataset for the row loader
├── model/
│   ├── rf_model.pkl            Trained Random Forest model
│   ├── dl_model.keras          Trained Deep Learning model (Keras format)
│   ├── scaler.pkl              StandardScaler preprocessing artifact
│   ├── feature_names.pkl       Selected feature list
│   ├── label_encoder.pkl       Multi-class label encoder
│   ├── metrics.json            Saved evaluation metrics
│   └── predictions.db          SQLite alert history database
├── train.py                    Model training and artifact generation
├── requirements.txt            Python dependencies
├── Dockerfile                  Container deployment support
└── README.md
```

## Dataset

This project uses the CIC-IDS2017 dataset. The full dataset is not included in this repository due to file size. Download it from the Canadian Institute for Cybersecurity:

https://www.unb.ca/cic/datasets/ids-2017.html

Place the CSV files inside the `data/` folder. The `friday.csv` file included in this repo is a sample for the row loader only and is not sufficient for full training.

## Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/your-username/your-repo-name
cd your-repo-name
python -m venv venv
```

Activate the environment:

```bash
# Windows
.\venv\Scripts\Activate.ps1

# Mac / Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run the training script to generate all model artifacts before starting the app:

```bash
python train.py
```

This will produce the following files in the `model/` directory:

- `rf_model.pkl`
- `dl_model.keras` (Keras 3.x compatible format)
- `scaler.pkl`
- `feature_names.pkl`
- `label_encoder.pkl`
- `metrics.json`

## Model Format

The Deep Learning model is saved in the modern `.keras` format (native Keras format) which is compatible with Keras 3.x and newer versions of TensorFlow. This ensures compatibility across different environments and prevents deserialization errors that can occur with legacy `.h5` format files.

## Running the App

```bash
python app/app.py
```

Open your browser at:

```
http://localhost:5000
```

## Docker

Build the image:

```bash
docker build -t rtids .
```

Run the container:

```bash
docker run -p 5000:5000 rtids
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/features` | Feature names used by the models |
| GET | `/api/sample` | Random sample row from the dataset |
| GET | `/api/metrics` | Model evaluation metrics |
| POST | `/api/predict` | Real-time single-row prediction |
| GET | `/api/alerts` | Recent stored predictions |
| POST | `/api/analyze-batch` | Bulk CSV prediction summary |
| GET | `/api/health` | Model load status and health check |

### Prediction Request Example

```json
{
  "model": "rf",
  "Flow Duration": 12345,
  "Total Fwd Packets": 10,
  "Total Backward Packets": 5,
  "Flow Bytes/s": 1024.5
}
```

### Prediction Response Example

```json
{
  "prediction": "DoS Hulk",
  "confidence": "97.43%",
  "attack_probability": "97.43%",
  "benign_probability": "2.57%",
  "model": "Random Forest",
  "mitre_technique": "T1499",
  "mitre_name": "Endpoint Denial of Service",
  "shap_features": [
    { "feature": "Flow Bytes/s", "value": 0.8421 },
    { "feature": "Flow Duration", "value": -0.3102 }
  ],
  "timestamp": "2026-05-23T14:32:10"
}
```

## Tech Stack

- **Backend** — Python, Flask, Flask-CORS
- **Models** — scikit-learn, TensorFlow, Keras
- **Explainability** — SHAP
- **Data** — pandas, numpy, imbalanced-learn
- **Storage** — SQLite
- **Frontend** — HTML, CSS, JavaScript, Chart.js (CDN)
- **Deployment** — Docker

## Notes

- Chart.js is loaded via CDN in the frontend. No additional frontend installation is needed.
- The `predictions.db` file is created automatically on first run inside the `model/` folder.
- Do not commit `*.pkl`, `*.h5`, `*.db`, or `*.csv` files to GitHub. The `.gitignore` handles this.

## License

This project is provided for educational and demonstration purposes.