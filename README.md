# AI-based Mental Health Monitoring using Social Signals (Multi-Modal)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-informational)](https://www.docker.com/)

Research prototype for early risk detection using multi-modal social signals: text (posts), activity metadata, and optional speech features. Ethically designed; uses only synthetic/anonymized data.

## Overview
- Text: transformer-based affect and stress detection; emotion distributions over time
- Activity: posting cadence, circadian rhythm features, social interaction patterns
- Speech (optional): prosody, MFCCs, voice energy variability
- Fusion: late-fusion with calibrated probabilities and temporal smoothing (EWMA/HMM)

## Responsible AI & Ethics
- Differential privacy for aggregation; opt-in consent only
- Bias checks across demographics; model cards and datasheets
- Not a diagnostic tool; for research and wellness insights only

## Tech Stack
- NLP: Transformers, emotion classification, LIWC-like features
- Audio: librosa, TorchAudio (if speech used)
- Time-series: tsfresh, statsmodels
- Serving: FastAPI, Uvicorn, Docker
- MLOps: MLflow, DVC

## Repository Structure
```
mental-health-mm/
  data/
  notebooks/
  src/
    text/
      preprocess.py
      affect_model.py
    activity/
      cadence.py
      circadian.py
    audio/
      extract.py
      features.py
    fusion/
      fuse.py
      smooth.py
    evaluate.py
    api/
      main.py
  configs/
  tests/
  docker/
```

## Sample Results (synthetic)
- Text affect F1: 0.86
- Activity anomaly AUPRC: 0.71
- Fusion ROC-AUC: 0.91; EER: 0.18

## Installation
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Run API:
```
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```
Batch score:
```
python -m src.evaluate --input data/sample_events.jsonl
```

## Roadmap
- Temporal transformers for longitudinal signals
- On-device privacy-preserving inference
- Human-in-the-loop review tooling
