# Athens Climate Feature Engineering and Climate Instability Index (CII)

This repository provides a Python script to process historical daily weather observations from Athens and generate a **feature-rich climate dataset**. It introduces a novel **Climate Instability Index (CII)** that quantifies atmospheric variability and instability on a continuous scale.

---

## Features

- **Date Parsing & Cleaning**
  - Handles ambiguous 2-digit years (1900s vs 2000s).
  - Repairs missing or inconsistent weather measurements.

- **Feature Engineering**
  - 24 continuous features, including:
    - Basic weather variables (`T`, `Tmax`, `Tmin`, `RH`, `Precipitation`)
    - Derived metrics (temperature range, midpoint, asymmetry)
    - Temporal lag features (1, 3, 7 days)
    - Rolling averages & sums (3, 7, 30 days)
    - Seasonal encodings (sin/cos transforms of month & day)

- **Climate Instability Index (CII)**
  - Combines four components:
    1. Thermal instability
    2. Moisture instability
    3. Weather regime instability
    4. Seasonal pattern disruption
  - Scaled to **0–100** for interpretability.

- **Outputs**
  - Cleaned dataset saved as CSV
  - Summary statistics and feature correlations

---

## Requirements

- Python 3.8+
- pandas
- numpy

Install dependencies:
```bash
pip install pandas numpy
