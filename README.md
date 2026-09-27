# Machine Learning-Based Inventory Anomaly Detection System

A complete Flask + scikit-learn application that helps inventory managers find unusual stock, sales, order, and demand patterns for investigation.

> **Important:** This is unsupervised anomaly detection, not ordinary classification or regression. An anomaly is a signal that a record is unusual compared with the learned pattern; it is not proof of fraud, theft, or an incorrect record.

## Dataset

The application uses the original **Retail Store Inventory Forecasting Dataset** from Kaggle: [source](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset). It contains 73,100 daily store/product observations and is licensed CC0 Public Domain. Kaggle describes it as synthetic but realistic; therefore it is not real company data.

Raw file: `data/retail_store_inventory.csv`.

### Data dictionary

| Original column | Meaning |
|---|---|
| Date | Daily observation date |
| Store ID | Store identifier |
| Product ID | Product/SKU identifier |
| Category | Product category |
| Region | Store region |
| Inventory Level | Units available at the beginning of the observation |
| Units Sold | Units sold during the day |
| Units Ordered | Units ordered for replenishment |
| Demand Forecast | Forecast demand value |
| Price | Selling price |
| Discount | Discount value/indicator in the source |
| Weather Condition | Recorded weather category |
| Holiday/Promotion | Promotion or holiday indicator |
| Competitor Pricing | Competitor price |
| Seasonality | Seasonal category |

Derived features: `Stock_to_Sales_Ratio`, `Days_of_Inventory`, `Order_to_Sales_Ratio`, `Demand_Change`, `Rolling_Sales_Average`, `Forecast_Gap`, `Price_Gap`, `Inventory_Value`, and `Inventory_Turnover`. These are calculated transparently from the original fields.

## Architecture

`CSV → validation → cleaning → feature engineering → saved preprocessing → KDE/GMM/K-Means/IQR scores → 0–1 score → 95th percentile threshold → NORMAL/ANOMALY → explanation → recommended action → Flask dashboard`

PCA is used for dimensionality reduction and 2D analysis. It is not treated as a standalone anomaly classifier.

## Models and evaluation

- **IQR baseline:** simple univariate statistical reference.
- **KDE:** low estimated density indicates unusual observations.
- **GMM:** low log-likelihood indicates unusual observations.
- **K-Means:** distance from a cluster center is comparative evidence.
- **PCA:** two-component visualization of multivariate structure.
- **Combined score:** 30% KDE + 30% GMM + 20% K-Means distance + 20% IQR baseline.
- **Threshold:** default 95th percentile of training scores; higher threshold means fewer alerts and lower investigation workload, while lower threshold increases coverage and false alerts.

Current training output is written to `outputs/evaluation.json` and `outputs/anomaly_results.csv`.

## Installation and running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000`.

The app includes `/`, `/dashboard`, `/upload`, `/analyze`, `/anomalies`, `/details/<id>`, `/results`, `/blog`, and `/download`.

## Project structure

```text
app.py                    Flask server and inference routes
train_model.py            training and artifact persistence
feature_engineering.py    cleaning and derived inventory features
preprocessing.py           Robust imputation/scaling
anomaly_detection.py      KDE, GMM, PCA, K-Means, explanations
evaluation.py             EDA plots and unsupervised evaluation
generate_data.py           optional clearly-labelled demo subset helper
requirements.txt
BLOG.md
PRESENTATION.md
VIVA.md
TERMINOLOGY.md
data/retail_store_inventory.csv
models/                    joblib models and metadata
outputs/                   results, evaluation, plots
templates/                 Jinja2 pages
static/                    CSS, JavaScript, charts
notebooks/EDA.ipynb
```

## CSV upload

Uploaded CSVs must include the 15 original Kaggle columns. The application validates the file type and columns, cleans types, preserves unusual valid values, applies the saved preprocessing and models, and writes a downloadable result. It never retrains on upload.

## Business use and limitations

Use the result to prioritize review: high stock coverage may suggest overstock; low coverage may suggest stockout risk; large orders relative to sales may suggest a purchase-plan review; sharp demand changes may need validation. All recommendations are decision-support suggestions.

Limitations include synthetic source behavior, false positives from unsupervised models, threshold sensitivity, and product-specific normality. Real deployment should use validated company data, time-aware validation, monitoring, access control, alerts, and domain review.
