# Machine Learning-Based Inventory Anomaly Detection System

A complete **Flask + scikit-learn** application that uses unsupervised machine learning to identify unusual inventory, sales, order, and demand patterns for investigation.

The system analyzes historical retail inventory data, creates additional operational features, combines multiple anomaly-detection techniques into a single normalized anomaly score, and presents the results through an interactive Flask web application.

> **Important:** This is an **unsupervised anomaly detection system**, not a traditional classification or regression model. An anomaly means that an observation is unusual compared with the learned pattern. It is **not proof of fraud, theft, data-entry error, or any other specific cause**.

---

## 1. Project Overview

Inventory managers may need to review thousands of daily store-product records. Unusual patterns can be difficult to identify manually.

Examples include:

* Unusually high inventory
* Unusually low inventory
* Unexpected sales patterns
* Large orders compared with sales
* Sudden demand changes
* Unusual forecast gaps
* Unusual pricing relationships
* Potential slow-moving or overstock situations

This project provides an automated way to **prioritize records for investigation**.

The system does not automatically declare that an inventory record is wrong. Instead, it identifies records that deserve additional human review.

### Core Business Question

> **Which inventory records look unusual compared with the normal patterns learned from the available retail data?**

---

# 2. Business Objective

The main objective is to develop a machine-learning system that can:

1. Process historical inventory data.
2. Create meaningful inventory and demand features.
3. Learn patterns without requiring a labeled anomaly dataset.
4. Assign an anomaly score to each observation.
5. Flag the most unusual observations.
6. Explain possible indicator signals behind an alert.
7. Provide recommended investigation actions.
8. Display the results through a Flask dashboard.

The output is designed for **decision support**, not automatic operational action.

---

# 3. Why Anomaly Detection?

Traditional supervised classification requires labeled examples such as:

```text
Normal
Anomaly
```

However, the source dataset does not provide a trusted ground-truth anomaly label.

Therefore, this project uses **unsupervised machine learning**.

Instead of teaching the model exactly what an anomaly looks like, the system learns the structure and distribution of the available observations and identifies records that appear unusual.

This makes the project suitable for exploratory inventory monitoring where verified anomaly labels are unavailable.

---

# 4. Dataset

The project uses the original:

**Retail Store Inventory Forecasting Dataset**

from Kaggle.

The dataset contains:

* **73,100 daily store/product observations**
* **15 original columns**
* Synthetic-but-realistic retail inventory data
* **CC0 Public Domain** license

The source dataset is described as synthetic but realistic, so it should **not be represented as real company transaction data**.

### Raw Dataset

```text
data/retail_store_inventory.csv
```

---

# 5. Data Dictionary

| Original Column    | Meaning                                             |
| ------------------ | --------------------------------------------------- |
| Date               | Daily observation date                              |
| Store ID           | Store identifier                                    |
| Product ID         | Product/SKU identifier                              |
| Category           | Product category                                    |
| Region             | Store region                                        |
| Inventory Level    | Units available at the beginning of the observation |
| Units Sold         | Units sold during the day                           |
| Units Ordered      | Units ordered for replenishment                     |
| Demand Forecast    | Forecast demand value                               |
| Price              | Selling price                                       |
| Discount           | Discount value/indicator                            |
| Weather Condition  | Recorded weather category                           |
| Holiday/Promotion  | Promotion or holiday indicator                      |
| Competitor Pricing | Competitor price                                    |
| Seasonality        | Seasonal category                                   |

---

# 6. Feature Engineering

The system does not rely only on the original columns.

Additional inventory-related features are derived transparently from the source data.

### Stock-to-Sales Ratio

Measures inventory relative to sales.

```text
Stock_to_Sales_Ratio
```

A very large or very small relationship may indicate an unusual inventory situation.

### Days of Inventory

Estimates inventory coverage based on sales behavior.

```text
Days_of_Inventory
```

### Order-to-Sales Ratio

Compares replenishment orders with units sold.

```text
Order_to_Sales_Ratio
```

### Demand Change

Captures changes in sales/demand behavior.

```text
Demand_Change
```

### Rolling Sales Average

Represents recent sales behavior.

```text
Rolling_Sales_Average
```

### Forecast Gap

Measures the difference between forecast demand and observed demand.

```text
Forecast_Gap
```

### Price Gap

Represents the difference between the product price and competitor pricing.

```text
Price_Gap
```

### Inventory Value

Estimates inventory value using inventory and price information.

```text
Inventory_Value
```

These features help the anomaly-detection system capture relationships between inventory, sales, orders, demand, and pricing.

---

# 7. Data Preprocessing

Before anomaly detection, the system performs data preparation.

The workflow includes:

```text
Raw CSV
   ↓
Validation
   ↓
Type Cleaning
   ↓
Duplicate Handling
   ↓
Feature Engineering
   ↓
Missing-Value Handling
   ↓
Robust Scaling
   ↓
Anomaly Detection
```

Potential anomalies are **not simply removed before modeling**, because unusual observations are the actual subject of this project.

Robust preprocessing is used to reduce the influence of extreme values while preserving unusual observations for analysis.

---

# 8. Machine Learning Architecture

The complete system follows this architecture:

```text
CSV Dataset
      ↓
Validation
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Saved Preprocessing
      ↓
 ┌───────────────┐
 │ KDE           │
 │ GMM           │
 │ K-Means       │
 │ IQR Baseline  │
 └───────────────┘
      ↓
Individual Scores
      ↓
Normalized 0–1 Scores
      ↓
Weighted Combined Score
      ↓
95th Percentile Threshold
      ↓
NORMAL / ANOMALY
      ↓
Explanation
      ↓
Recommended Action
      ↓
Flask Dashboard
```

---

# 9. Anomaly Detection Methods

The project combines several different approaches.

## 9.1 IQR Baseline

The **Interquartile Range (IQR)** method provides a simple statistical baseline.

It identifies observations that fall unusually far outside the central distribution of individual variables.

It is useful as a reference because it provides a simple statistical comparison against the more complex machine-learning approaches.

---

## 9.2 Kernel Density Estimation

**Kernel Density Estimation (KDE)** estimates how densely observations are distributed.

The basic idea is:

> Observations in very low-density regions are more unusual.

Therefore, lower estimated density contributes more strongly toward the anomaly score.

---

## 9.3 Gaussian Mixture Model

The **Gaussian Mixture Model (GMM)** models the data as a mixture of underlying probability distributions.

The model estimates how likely an observation is under the learned distribution.

A low likelihood indicates that the observation is less consistent with the learned pattern.

---

## 9.4 K-Means

K-Means groups observations into clusters.

The distance between an observation and its nearest cluster center provides comparative evidence about how closely the observation fits an existing group.

A larger distance can indicate a potentially unusual observation.

However, cluster distance alone is not treated as proof that a record is anomalous.

---

## 9.5 PCA

**Principal Component Analysis (PCA)** is used for dimensionality reduction and two-dimensional analysis.

It helps visualize the structure of the high-dimensional feature space.

PCA is **not used as a standalone anomaly classifier** in this project.

---

# 10. Combined Anomaly Score

Rather than relying on one algorithm, the project combines evidence from multiple methods.

The final score uses:

```text
30% KDE
+
30% GMM
+
20% K-Means Distance
+
20% IQR Baseline
```

Therefore:

```text
Combined Score =
0.30 × KDE
+ 0.30 × GMM
+ 0.20 × K-Means
+ 0.20 × IQR
```

The resulting score is normalized to a **0–1 scale**.

A higher score indicates stronger evidence that the observation is unusual according to the combined system.

---

# 11. Anomaly Threshold

The default threshold is the **95th percentile of the training anomaly scores**.

The actual trained threshold is approximately:

```text
0.39545
```

Records at or above the threshold are classified as:

```text
ANOMALY
```

Records below the threshold are classified as:

```text
NORMAL
```

The threshold is a configurable decision rule rather than a universal definition of an anomaly.

### Threshold Trade-Off

A higher threshold:

* Produces fewer alerts
* Reduces investigation workload
* May miss some unusual observations

A lower threshold:

* Produces more alerts
* Increases investigation workload
* May generate more false alerts

---

# 12. Current Model Results

The current training output contains:

```text
Records:             73,100
Anomalies:            3,655
Anomaly Percentage:      5.0%
Mean Score:          0.14693
Threshold:           0.39545
```

The 95th-percentile threshold results in approximately **5% of observations being flagged as anomalies**.

This is consistent with the percentile-based alerting rule and should not be interpreted as evidence that exactly 5% of the real-world records are truly anomalous.

---

# 13. Unsupervised Evaluation

Because the dataset does not provide verified anomaly labels, conventional supervised metrics such as:

```text
Accuracy
Precision
Recall
F1-score
```

are not the primary evaluation framework.

Instead, the project reports model-specific comparative information and clustering structure.

The current K-Means clustering evaluation reports a:

```text
Silhouette Score: 0.81694
```

The silhouette score provides information about how well the observations fit the resulting cluster structure.

It should not be interpreted as an anomaly-detection accuracy score.

---

# 14. Flask Web Application

The machine-learning system is integrated into a **Flask web application**.

The application provides an interactive interface for exploring anomaly results.

### Main Routes

```text
/
 /dashboard
 /upload
 /analyze
 /anomalies
 /details/<id>
 /results
 /blog
 /download
```

---

# 15. Dashboard

The dashboard provides an overview of the inventory data and anomaly results.

It can present information such as:

* Total records
* Number of anomalies
* Anomaly percentage
* Anomaly scores
* Inventory patterns
* Sales patterns
* Charts
* Filters
* Detailed records

This allows inventory managers to move from a high-level overview to individual observations requiring investigation.

---

# 16. Anomaly Explanation

An anomaly alert does not automatically identify the exact cause.

Instead, the application provides **indicator-based explanations**.

For example, a record may be flagged because it shows an unusual combination of:

* Inventory coverage
* Sales behavior
* Order quantity
* Demand change
* Forecast gap
* Price relationship

These are signals for investigation rather than causal conclusions.

---

# 17. Business Interpretation

The system can support several types of inventory review.

### High Stock Coverage

May indicate potential:

* Overstock
* Slow-moving inventory
* Lower-than-expected sales

### Low Stock Coverage

May indicate possible:

* Stockout risk
* Replenishment concern
* Increased demand

### Large Orders Compared with Sales

May indicate that the purchase plan should be reviewed.

### Sudden Demand Change

May require validation of:

* Promotion effects
* Seasonality
* Market changes
* Data quality
* Unexpected customer behavior

The system provides these as **decision-support suggestions**, not automatic business conclusions.

---

# 18. CSV Upload

The application supports uploading inventory CSV files for analysis.

Uploaded files must contain the required **15 original Kaggle columns**.

The application:

1. Validates the file.
2. Checks required columns.
3. Cleans data types.
4. Preserves unusual valid values.
5. Creates the derived features.
6. Applies the saved preprocessing.
7. Applies the saved anomaly-detection models.
8. Generates anomaly scores.
9. Produces downloadable results.

### Important

The application **does not retrain the models when a CSV is uploaded**.

This separates:

```text
Training
```

from:

```text
Inference / Analysis
```

and helps maintain a consistent deployed model.

---

# 19. Project Structure

```text
inventory_anomaly_system/
│
├── app.py
├── train_model.py
├── feature_engineering.py
├── preprocessing.py
├── anomaly_detection.py
├── evaluation.py
├── generate_data.py
├── requirements.txt
│
├── BLOG.md
├── PRESENTATION.md
├── VIVA.md
├── TERMINOLOGY.md
├── README.md
├── test_project.py
│
├── data/
│   └── retail_store_inventory.csv
│
├── models/
│   ├── preprocessing.pkl
│   ├── models.pkl
│   └── metadata.json
│
├── outputs/
│   ├── anomaly_results.csv
│   ├── current_results.csv
│   └── evaluation.json
│
├── notebooks/
│   └── EDA.ipynb
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── anomalies.html
│   ├── details.html
│   ├── upload.html
│   ├── results.html
│   └── blog.html
│
└── static/
    ├── css/
    ├── js/
    └── plots/
```

---

# 20. Installation

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 21. Train the Models

Run:

```bash
python train_model.py
```

The training process generates and saves the preprocessing and anomaly-detection artifacts.

Results are written to:

```text
outputs/evaluation.json
outputs/anomaly_results.csv
```

Model artifacts are stored in:

```text
models/
```

---

# 22. Run the Flask Application

Start the application:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The application can then be used to explore inventory patterns and investigate unusual records.

---

# 23. Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* KDE
* Gaussian Mixture Model
* K-Means
* PCA
* IQR statistical baseline

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* JavaScript-based dashboard charts

### Web Development

* Flask
* HTML
* CSS
* JavaScript
* Jinja2

### Model Persistence

* Joblib

---

# 24. Important Limitations

This project has several limitations that should be clearly communicated.

### Synthetic Dataset

The source dataset is synthetic rather than actual company inventory data.

### No Ground-Truth Anomaly Label

The dataset does not provide verified labels identifying which records are truly anomalous.

Therefore, the system cannot claim that every flagged record is genuinely abnormal in a real-world business sense.

### False Positives

Unsupervised anomaly detection can produce false alerts.

An observation may be unusual mathematically but completely valid from a business perspective.

### Product-Specific Normality

Different products and stores may naturally have different patterns.

A value that is unusual for one product may be normal for another.

### Threshold Sensitivity

Changing the threshold changes the number of alerts and the investigation workload.

### Human Validation

Every important anomaly should be reviewed by a domain expert before taking operational action.

---

# 25. Future Improvements

A production-oriented version could be improved by adding:

* Real company inventory data
* Time-aware retraining
* Automated anomaly alerts
* Supplier-risk features
* Demand forecasting
* Stockout prediction
* Database integration
* Role-based access control
* Authentication
* Production monitoring
* Model drift detection
* Product-specific thresholds
* Store-specific thresholds
* Human feedback loops
* Alert history and investigation tracking

---

# 26. Key Machine Learning Insight

One of the most important lessons from this project is:

> **An anomaly score is not the same thing as a confirmed business problem.**

Machine learning can identify observations that differ from learned patterns.

However, the model cannot automatically know whether the difference represents:

* Fraud
* Theft
* A data-entry problem
* A promotion
* Seasonal demand
* A legitimate business event
* A supplier issue
* A sudden market change

Therefore, the correct workflow is:

```text
Machine Learning Alert
        ↓
Human Investigation
        ↓
Business Validation
        ↓
Decision
```

---

# 27. Conclusion

The **Machine Learning-Based Inventory Anomaly Detection System** demonstrates how unsupervised machine learning can be applied to inventory monitoring when reliable anomaly labels are unavailable.

The system combines:

* Data validation
* Feature engineering
* Robust preprocessing
* IQR
* KDE
* Gaussian Mixture Models
* K-Means
* PCA visualization
* Combined anomaly scoring
* Percentile-based thresholding
* Explanation generation
* Flask web deployment

The current implementation analyzes **73,100 records** and flags **3,655 observations, or 5%, as anomalies** using the 95th-percentile threshold.

The system is therefore best understood as an **inventory investigation and decision-support tool**, rather than a system that automatically determines whether a record is wrong.

---

# 28. Project Summary

> **This project develops a Flask-based unsupervised machine-learning system for detecting unusual inventory, sales, order, and demand patterns. It uses engineered inventory features and combines KDE, Gaussian Mixture Model, K-Means distance, and an IQR baseline into a normalized anomaly score. A 95th-percentile training threshold identifies approximately 5% of observations for investigation. The results are presented through a Flask dashboard with explanations, recommendations, analytics, and CSV download functionality.**

---

## Final Takeaway

**The purpose of this project is not to say "this record is wrong."**

The purpose is:

> **"This record looks unusual compared with the learned inventory patterns, so it should be investigated."**

That distinction is what makes the project an **unsupervised inventory anomaly-detection system** rather than a conventional classification or regression application.
