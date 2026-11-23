
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from feature_engineering import engineer_features, validate_columns, FEATURE_COLUMNS
from anomaly_detection import score_models, explain_record, recommend_action

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Inventory Intelligence AI",
    page_icon="📦",
    layout="wide"
)

@st.cache_resource
def load_models():
    model_dir = ROOT / "models"

    preprocessing = joblib.load(model_dir / "preprocessing.pkl")
    models = joblib.load(model_dir / "models.pkl")
    metadata = json.loads(
        (model_dir / "metadata.json").read_text()
    )

    return preprocessing, models, metadata


def analyze_inventory(raw, threshold):
    preprocessing, models, metadata = load_models()

    missing = validate_columns(raw)

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df = engineer_features(raw)

    X = preprocessing.transform(df[FEATURE_COLUMNS])
    scores = score_models(models, X)

    for key, value in scores.items():
        df[key] = value

    df["Anomaly Score"] = df["anomaly_score"].round(6)

    df["Status"] = np.where(
        df["anomaly_score"] >= threshold,
        "ANOMALY",
        "NORMAL"
    )

    df["Key Indicators"] = df.apply(explain_record, axis=1)
    df["Recommended Action"] = df.apply(recommend_action, axis=1)

    return df


st.sidebar.title("📦 Inventory AI")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Upload Dataset", "Anomaly Records", "About Model"]
)

st.title("📦 Inventory Anomaly Detection System")
st.caption("Machine Learning Powered Inventory Monitoring")

try:
    preprocessing, models, metadata = load_models()
    threshold = float(metadata["threshold"])

    if page == "Dashboard":
        st.subheader("Inventory Overview")

        data_path = ROOT / "data" / "retail_store_inventory.csv"

        if not data_path.exists():
            data_path = ROOT / "retail_store_inventory.csv"

        if data_path.exists():
            raw = pd.read_csv(data_path)

            with st.spinner("Analyzing inventory data..."):
                results = analyze_inventory(raw, threshold)

            st.session_state["results"] = results

            total = len(results)
            anomalies = int((results["Status"] == "ANOMALY").sum())
            normal = total - anomalies
            percentage = anomalies / total * 100 if total else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records", f"{total:,}")
            c2.metric("Anomalies", f"{anomalies:,}")
            c3.metric("Normal Records", f"{normal:,}")
            c4.metric("Anomaly Rate", f"{percentage:.2f}%")

            st.divider()

            left, right = st.columns(2)

            with left:
                st.subheader("Normal vs Anomalies")
                st.bar_chart(
                    results["Status"].value_counts()
                )

            with right:
                st.subheader("Anomaly Score Distribution")
                st.histogram if False else None
                st.bar_chart(
                    results["Anomaly Score"].round(1).value_counts().sort_index()
                )

            st.subheader("Recent Anomaly Records")
            st.dataframe(
                results[results["Status"] == "ANOMALY"].head(20),
                use_container_width=True
            )
        else:
            st.warning("Dataset file was not found. Upload a CSV dataset.")

    elif page == "Upload Dataset":
        st.subheader("Upload Inventory CSV")

        uploaded_file = st.file_uploader(
            "Choose your inventory CSV file",
            type=["csv"]
        )

        if uploaded_file:
            raw = pd.read_csv(uploaded_file)

            st.write("Dataset Preview")
            st.dataframe(raw.head(), use_container_width=True)

            if st.button("Run Anomaly Detection", type="primary"):
                with st.spinner("Running machine learning models..."):
                    results = analyze_inventory(raw, threshold)

                st.session_state["results"] = results
                st.success("Analysis completed successfully!")

                st.metric(
                    "Detected Anomalies",
                    int((results["Status"] == "ANOMALY").sum())
                )

                st.dataframe(
                    results[results["Status"] == "ANOMALY"],
                    use_container_width=True
                )

                st.download_button(
                    "Download Analysis Results",
                    results.to_csv(index=False).encode("utf-8"),
                    "inventory_anomaly_results.csv",
                    "text/csv"
                )

    elif page == "Anomaly Records":
        st.subheader("Detected Inventory Anomalies")

        results = st.session_state.get("results")

        if results is not None:
            anomalies = results[results["Status"] == "ANOMALY"]

            if "Category" in anomalies.columns:
                categories = ["All"] + sorted(
                    anomalies["Category"].dropna().astype(str).unique()
                )

                selected = st.selectbox("Filter by Category", categories)

                if selected != "All":
                    anomalies = anomalies[
                        anomalies["Category"].astype(str) == selected
                    ]

            st.dataframe(anomalies, use_container_width=True)

            st.download_button(
                "Download Anomalies",
                anomalies.to_csv(index=False).encode("utf-8"),
                "detected_anomalies.csv",
                "text/csv"
            )
        else:
            st.info("Run an analysis from Dashboard or Upload Dataset first.")

    elif page == "About Model":
        st.subheader("Machine Learning Architecture")

        st.write("""
        This project uses unsupervised machine learning.

        - Kernel Density Estimation (KDE)
        - Gaussian Mixture Model (GMM)
        - K-Means Clustering
        - Interquartile Range (IQR)
        - Principal Component Analysis (PCA)
        """)

        st.metric("Anomaly Threshold", f"{threshold:.5f}")

        st.info(
            "An anomaly is an unusual observation that requires "
            "investigation. It does not prove fraud or theft."
        )

except FileNotFoundError as error:
    st.error(f"Model files not found: {error}")
    st.info("Check your model artifact folder and file names.")

except Exception as error:
    st.error(f"Application error: {error}")
