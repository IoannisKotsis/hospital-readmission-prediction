## Hospital Readmission Prediction

This ML system predicts the risk of hospital readmission for diabetic patients within 30 dyas of discharge.

Reducing early readmissions is critical for hospitals as it reflects:
  - Improved Healthcare Quality
  - Patient Safety
  - Cost Reduction

Built with:
  - **XGBoost**
  - **SHAP explainability**
  - **FastAPI REST API**
  - **PostgreSQL**
  - **Gemini LLM-powered clinical summaries**
  - **Containerized with Docker**
  - **Deployed on Railway**


## Live Demo

The API is deployed and publicly accessible at:
[Live API](hospital-readmission-prediction-production.up.railway.app/docs)

Try the `/predict` endpoint with the sample payloads 
in the `examples/` folder.


## Features

- The ML model outputs the predicted probability of readmission and the predicted class (0: No readmission, 1: Readmission in <30 days) using XGBoost.
- SHAP explainability identifies the key clinical factors driving each prediction, enabling transparent decision-making.
- The Gemini LLM translates model predictions into human-readable explanations.
- PostgreSQL logging of all predictions with raw input data for model performance monitoring.
- RESTful API built with FastAPI, providing a /predict endpoint for real-time inference.
- Dockerized application with docker-compose for reproducible deployment across any computer environment.
- Deployed on Railway (PaaS) with a public endpoint accessible to any user.

## Tech Stack

|
ML Model
|
XGBoost, RandomizedSearchCV, Scikit-learn
|
|
Explainability
|
SHAP values
|
|
API
|
FastAPI, Pydantic
|
|
Database
|
PostgreSQL
|
|
LLM
|
Google Gemini
|
|
Containerization
|
Docker, docker-compose
|
|
Deployment
|
Railway (PaaS)
|
|
Language
|
Python
|

## Model Performance

![Alt-text](src/roc_curve.png)

