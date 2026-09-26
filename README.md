# Predictive Maintenance AI Platform

> A two-module predictive maintenance system that combines machine learning, 
> explainable AI (SHAP), and large language models (Gemini) to deliver 
> maintenance-grade predictions with plain-English explanations.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-189FDD?logo=xgboost&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## Overview

Industrial equipment fails unpredictably, and traditional maintenance is 
either **reactive** (fix after it breaks) or **preventive** (fix on a fixed 
schedule regardless of condition). This project demonstrates a **predictive** 
approach: sensor data in, remaining life or fault category out, with an 
explanation of *why*.

The platform handles two distinct problem types under one shared pipeline:

| Module | Dataset | Task | Output |
|---|---|---|---|
| **Turbofan** | NASA C-MAPSS FD001 | Regression | Remaining Useful Life (cycles) |
| **Bearing** | CWRU Bearing Dataset | Classification | Fault category + confidence |

Both modules share an **explainability layer** (SHAP) and a **reporting layer** 
(Gemini LLM) that turns model output into a maintenance note a technician can act on.

---

## Screenshots

<img width="1792" height="1050" alt="Screenshot 2026-09-26 at 10 41 23 AM" src="https://github.com/user-attachments/assets/3d0ac5d7-8140-401e-8148-c097e9975c5e" />

<img width="1792" height="1050" alt="Screenshot 2026-09-26 at 10 41 36 AM" src="https://github.com/user-attachments/assets/d2c4bd2e-ee6c-4a81-998e-7baabd426356" />

<img width="1792" height="1050" alt="Screenshot 2026-09-26 at 10 41 58 AM" src="https://github.com/user-attachments/assets/1116184b-c48d-494f-8c4a-36497d390ed3" />

<img width="1792" height="1050" alt="Screenshot 2026-09-26 at 10 42 00 AM" src="https://github.com/user-attachments/assets/894befdb-b694-4143-b0e6-558af742e92e" />

<img width="1792" height="1050" alt="Screenshot 2026-09-26 at 10 42 07 AM" src="https://github.com/user-attachments/assets/b24aef13-7ac1-41f2-8c63-395034054ddc" />


### Turbofan module — RUL prediction
![Turbofan dashboard](plots/screenshots/turbofan.png)

### Bearing module — fault classification
![Bearing dashboard](plots/screenshots/bearing.png)

### AI-generated maintenance report
![AI report](plots/screenshots/report.png)

---

## Architecture
predictive/
├── code/
│   ├── app.py            # Streamlit dashboard

│   ├── cwru.py           # Bearing module — data, features, model
│   ├── train_cwru.py     # bearing training script
│   ├── main.py           # Turbofan XGBoost training
│   ├── main2.py          # Turbofan LSTM training
│   ├── explain.py        # SHAP global + local plots
│   └── llm_report.py     # Gemini integration
├── data/
│   ├── train_FD001.txt   # C-MAPSS training
│   ├── test_FD001.txt    # C-MAPSS test
│   ├── RUL_FD001.txt     # C-MAPSS answer key
│   └── cwru/             # CWRU .npz files
├── models/               # Trained model artifacts
├── plots/                # SHAP and evaluation plots
├── requirements.txt
├── .env                  
└── README.md




---

## Results

### Turbofan — RUL regression (C-MAPSS FD001 test set)

| Model | RMSE (cycles) | MAE (cycles) | Notes |
|-------|---------------|--------------|-------|
| XGBoost (rolling stats) | 48.4 | 40.2 | Baseline, tabular approach |
| LSTM (30-cycle windows) | 16.9 | 12.9 | Sequence model, ~3× improvement |

The LSTM was implemented as a proof-of-concept for the synopsis's stated 
future scope (deep learning for improved accuracy). XGBoost remains the 
primary model because it feeds the SHAP explainability layer directly.

### Bearing — fault classification (CWRU 12k drive-end)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Normal | 1.00 | 1.00 | 1.00 | 95 |
| Inner Race | 1.00 | 1.00 | 1.00 | 47 |
| Ball | 1.00 | 1.00 | 1.00 | 48 |
| Outer Race | 1.00 | 1.00 | 1.00 | 47 |
| **Overall accuracy** | | | **1.00** | 237 |

The bearing classification task is inherently more separable than RUL 
regression — the vibration signatures of the four fault classes are 
well-distinguished by time-domain statistics (RMS, kurtosis, crest factor) 
plus frequency band energies.

---

## Setup

### Requirements
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install

```bash
git clone <your-repo-url>
cd predictive
uv venv
uv pip install -r requirements.txt
