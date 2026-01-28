# Predicting Market Breakouts

This repository contains a **machine learning–based market breakout prediction system** built using **XGBoost, scikit-learn, and Flask**.  
The project focuses on identifying potential breakout opportunities using technical indicators and historical market data.

---

## Project Overview

The system:
- Fetches and processes market data
- Applies technical analysis indicators
- Uses an XGBoost model to predict breakout signals
- Exposes predictions through a simple Flask web application

This project is built for **learning, experimentation, and practical application of ML in financial markets**.

---

## Tech Stack

- Python  
- XGBoost  
- scikit-learn  
- Flask  
- SQLite  
- HTML (templates)

---

## Key Components

- **Modeling:** XGBoost-based prediction models  
- **Data Processing:** Feature scaling, technical indicators  
- **Backend:** Flask application for serving predictions  
- **Storage:** SQLite database and serialized models  

---

## Project Structure

- `app.py` – Flask application entry point  
- `model.py / enhanced_model.py` – ML model logic  
- `technical_analysis.py` – Indicator calculations  
- `risk_manager.py` – Risk-related logic  
- `templates/` – HTML templates  
- `models/` – Saved ML models  

---

## Purpose

This project was created to:
- Apply machine learning to real-world market data
- Understand breakout detection using technical analysis
- Practice end-to-end ML project development and deployment

---

## Disclaimer

This project is for **educational purposes only** and should not be considered financial or investment advice.

---
