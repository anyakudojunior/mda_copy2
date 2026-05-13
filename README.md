# Predicting Daily Cycling Traffic in Flanders

## Research Question
To what extent can cycling traffic volume across Flanders be predicted using temporal, meteorological, spatial, and mobility-related features?

---

## Project Overview

This project builds a machine learning model to predict cycling traffic across counting stations in Flanders. It combines:

- Cycling count data from AWV
- Meteorological data
- Public transport accessibility data from De Lijn
- Spatial station information

A LightGBM model is trained on 2024 data and validated on 2025 data.

The results are visualised in an interactive Streamlit dashboard.