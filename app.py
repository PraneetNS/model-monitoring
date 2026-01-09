import streamlit as st
import pandas as pd
import joblib

from ml_monitor.monitoring.alert_monitoring import format_alert_for_humans
from ml_monitor.monitoring.monitor import ModelMonitor
from ml_monitor.monitoring.alerts import AlertManager, AlertThresholds
from ml_monitor.drift_detection.statistical import StatisticalDriftDetector
from ml_monitor.utils.visualization import plot_feature_distribution


# ==================================================
# Page config
# ==================================================
st.set_page_config(page_title="ML Model Monitoring", layout="wide")
st.title("📊 ML Model Monitoring & Drift Detection")

# ==================================================
# Sidebar – Dataset selection
# ==================================================
st.sidebar.header("Dataset Selection")

dataset_choice = st.sidebar.selectbox(
    "Choose dataset",
    [
        "Loan Default Dataset",
        "California Housing Dataset",
    ],
)

DATASET_CONFIG = {
    "Loan Default Dataset": {
        "reference": "data/reference.csv",
        "current": "data/current.csv",
        "model": "model_numeric.pkl",
        "type": "classification",
        "description": "Binary classification: loan default prediction",
    },
    "California Housing Dataset": {
        "reference": "data/reference_california.csv",
        "current": "data/current_california.csv",
        "model": "models/model_california.pkl",
        "type": "regression",
        "description": "Regression: median house value prediction",
    },
}

cfg = DATASET_CONFIG[dataset_choice]
dataset_type = cfg["type"]

st.sidebar.markdown(f"**Task type:** {dataset_type.capitalize()}")
st.sidebar.caption(cfg["description"])

run_monitoring = st.sidebar.button("Run Monitoring")

# ==================================================
# Helper: load & prepare data
# ==================================================
def load_and_prepare(path: str):
    df = pd.read_csv(path)
    df = df.select_dtypes(include=["number"])

    if df.shape[1] < 2:
        return df, None

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


# ==================================================
# Main logic
# ==================================================
if run_monitoring:
    try:
        # -----------------------------
        # Load data
        # -----------------------------
        st.subheader("📥 Data Loading")

        X_ref, y_ref = load_and_prepare(cfg["reference"])
        X_cur, y_cur = load_and_prepare(cfg["current"])

        col1, col2 = st.columns(2)
        col1.metric("Reference rows", X_ref.shape[0])
        col2.metric("Current rows", X_cur.shape[0])

        st.success("Data loaded successfully")

        # -----------------------------
        # Load model
        # -----------------------------
        st.subheader("🤖 Model Loading")

        model = joblib.load(cfg["model"])
        st.success("Model loaded")

        # -----------------------------
        # Setup monitoring
        # -----------------------------
        drift_detector = StatisticalDriftDetector()
        alert_manager = AlertManager(
            thresholds=AlertThresholds(
                accuracy=0.75,
                prediction_shift=0.3,
            )
        )

        monitor = ModelMonitor(
            model=model,
            drift_detector=drift_detector,
            alert_manager=alert_manager,
        )

        monitor.set_reference(X_ref, y_ref)
        results = monitor.monitor(X_cur, y_cur)

        st.session_state.monitor = monitor
        st.session_state.results = results

        # ==================================================
        # PERFORMANCE SECTION
        # ==================================================
        st.divider()
        st.header("📈 Model Performance")

        perf = results.get("performance")

        if not perf:
            st.info("Performance metrics are unavailable (labels not provided).")

        elif dataset_type == "classification":
            acc = perf.get("accuracy", 0)
            prec = perf.get("precision", 0)
            rec = perf.get("recall", 0)
            f1 = perf.get("f1_score", 0)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.1%}")
            col2.metric("Precision", f"{prec:.2f}")
            col3.metric("Recall", f"{rec:.2f}")
            col4.metric("F1 Score", f"{f1:.2f}")

            if acc >= 0.8:
                st.success("✅ Classification performance is healthy.")
            else:
                st.warning("⚠️ Classification performance may be degrading.")

            st.caption(
                "Accuracy shows overall correctness. "
                "Precision & Recall show error trade-offs. "
                "F1 balances both."
            )

        else:  # REGRESSION
            r2 = perf.get("r2_score")
            mae = perf.get("mae")
            rmse = perf.get("rmse")

            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{r2:.2f}")
            col2.metric("MAE", f"{mae:.2f}")
            col3.metric("RMSE", f"{rmse:.2f}")

            if r2 >= 0.6:
                st.success("✅ Model predictions are stable.")
            else:
                st.warning("⚠️ Predictive power may be degrading.")

            st.caption(
                "R² measures explained variance. "
                "MAE shows average error. "
                "RMSE penalizes large errors."
            )

        # ==================================================
        # DATA DRIFT SECTION
        # ==================================================
        st.divider()
        st.header("🧠 Data Drift Analysis")

        drift = results.get("drift", {})

        if not drift or not drift.get("drift_detected"):
            st.success("✅ No significant data drift detected.")
        else:
            st.warning("⚠️ Data drift detected")

            drift_score = drift.get("drift_score", 0)
            st.metric("Drift Score", f"{drift_score:.3f}")

            feature_results = drift.get("feature_results", {})
            drifted_features = [
                f for f, v in feature_results.items()
                if v.get("drift_detected")
            ]

            st.write(
                f"**{len(drifted_features)} features show statistical change:**"
            )

            for f in drifted_features:
                st.write(f"• {f}")

            st.caption(
                "Drift indicates changes in data distribution. "
                "It does not mean the model is broken, but it should be monitored."
            )

            # Feature visualization
            if drifted_features:
                feature = st.selectbox(
                    "Inspect feature distribution",
                    drifted_features,
                )

                fig = plot_feature_distribution(
                    reference_data=X_ref,
                    current_data=X_cur,
                    feature_name=feature,
                )
                st.pyplot(fig)

        # ==================================================
        # PREDICTION BEHAVIOR
        # ==================================================
        st.divider()
        st.header("📊 Prediction Behavior")

        preds = results.get("predictions", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{preds.get('mean', 0):.2f}")
        col2.metric("Std Dev", f"{preds.get('std', 0):.2f}")
        col3.metric("Min", f"{preds.get('min', 0):.2f}")
        col4.metric("Max", f"{preds.get('max', 0):.2f}")

        if dataset_type == "regression":
            st.info(
                "Prediction statistics show how estimated values are distributed. "
                "Large shifts in mean or spread may indicate market or population changes."
            )
        else:
            st.info(
                "Prediction distribution helps detect abnormal model behavior "
                "even if accuracy appears stable."
            )

        # ==================================================
        # ALERTS
        # ==================================================
        st.divider()
        st.header("🚨 Alerts (Human-Readable)")

        alerts = results.get("alerts", [])

        if not alerts:
            st.success("No alerts triggered.")
        else:
            for alert in alerts:
                human = format_alert_for_humans(alert)

                st.warning(f"⚠️ {human['title']}")
                st.write(human["message"])

                if human.get("affected_features"):
                    st.write(
                        "**Affected features:** "
                        + ", ".join(human["affected_features"])
                    )

                if human.get("recommended_action"):
                    st.info(f"👉 {human['recommended_action']}")

    except Exception as e:
        st.error(f"Error: {e}")

# ==================================================
# ALERT HISTORY
# ==================================================
st.divider()
st.header("🕒 Alert History")

monitor = st.session_state.get("monitor")

if monitor:
    history = monitor.get_history(limit=10)

    if not history:
        st.info("No historical alerts yet.")
    else:
        for record in reversed(history):
            ts = record["timestamp"]
            for alert in record.get("alerts", []):
                human = format_alert_for_humans(alert)
                with st.expander(f"{ts} — {human['title']}"):
                    st.write(human["message"])
