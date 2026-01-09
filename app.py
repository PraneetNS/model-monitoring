import streamlit as st
import pandas as pd
import joblib

from ml_monitor.monitoring.alert_monitoring import format_alert_for_humans
from ml_monitor.monitoring.monitor import ModelMonitor
from ml_monitor.monitoring.alerts import AlertManager, AlertThresholds
from ml_monitor.drift_detection.statistical import StatisticalDriftDetector
from ml_monitor.utils.visualization import plot_feature_distribution

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="ML Model Monitoring", layout="wide")

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📊 ML Model Monitoring & Drift Detection")

st.sidebar.header("Configuration")

model_path = st.sidebar.text_input(
    "Trained model / pipeline", "model_numeric.pkl"
)

reference_path = st.sidebar.text_input(
    "Reference data CSV", "data/reference.csv"
)

current_path = st.sidebar.text_input(
    "Current data CSV", "data/current.csv"
)

run_monitoring = st.sidebar.button("Run Monitoring")

# --------------------------------------------------
# Helper
# --------------------------------------------------
def load_and_prepare(path: str):
    df = pd.read_csv(path)

    # KEEP ONLY NUMERIC COLUMNS (IMPORTANT)
    df = df.select_dtypes(include=["number"])

    if df.shape[1] < 2:
        return df, None

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y

# --------------------------------------------------
# Main logic
# --------------------------------------------------
if run_monitoring:
    try:
        # -----------------------------
        # Load data
        # -----------------------------
        st.subheader("📥 Loading data")

        X_ref, y_ref = load_and_prepare(reference_path)
        X_cur, y_cur = load_and_prepare(current_path)

        st.write("Reference data shape:", X_ref.shape)
        st.write("Current data shape:", X_cur.shape)

        st.success("Data loaded successfully")

        # -----------------------------
        # Load model
        # -----------------------------
        st.subheader("🤖 Loading model / pipeline")

        model = joblib.load(model_path)
        st.success("Model loaded")

        # -----------------------------
        # Setup monitoring
        # -----------------------------
        drift_detector = StatisticalDriftDetector()

        thresholds = AlertThresholds(
            accuracy=0.75,
            prediction_shift=0.3,
        )

        alert_manager = AlertManager(thresholds=thresholds)

        monitor = ModelMonitor(
            model=model,
            drift_detector=drift_detector,
            alert_manager=alert_manager,
        )

        st.session_state.monitor = monitor

        # -----------------------------
        # Run monitoring (OLD SIMPLE FLOW)
        # -----------------------------
        st.subheader("🔍 Running monitoring")

        monitor.set_reference(X_ref, y_ref)
        results = monitor.monitor(X_cur, y_cur)

        st.session_state.results = results

        # ==================================================
        # PERFORMANCE (HUMAN FRIENDLY)
        # ==================================================
        st.subheader("📈 Model Performance (Easy Summary)")

        perf = results.get("performance")

        if not perf:
            st.info("No labels provided — performance metrics unavailable.")
        else:
            accuracy = perf.get("accuracy", 0.0)
            f1 = perf.get("f1_score", 0.0)

            if accuracy >= 0.8:
                st.success("✅ Model performance is healthy.")
            else:
                st.warning("⚠️ Model performance may be degrading.")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Overall Accuracy", f"{accuracy:.1%}")

            with col2:
                st.metric("Prediction Quality (F1 Score)", f"{f1:.2f}")

        # ==================================================
        # DATA DRIFT
        # ==================================================
        st.subheader("🧠 Data Changes (Easy Summary)")

        drift = results.get("drift", {})

        if not drift or not drift.get("drift_detected"):
            st.success("✅ Incoming data is consistent with historical data.")
        else:
            st.warning("⚠️ Incoming data is different from historical data.")

            feature_results = drift.get("feature_results", {})
            drifted_features = [
                f for f, v in feature_results.items()
                if v.get("drift_detected")
            ]

            if drifted_features:
                st.write("**Main changes detected in:**")
                for f in drifted_features:
                    st.write(f"• {f}")

                st.subheader("📊 Feature Distribution Changes")
                feature = st.selectbox(
                    "Inspect a feature",
                    drifted_features,
                )

                fig = plot_feature_distribution(
                    reference_data=X_ref,
                    current_data=X_cur,
                    feature_name=feature,
                )
                st.pyplot(fig)

        # ==================================================
        # ALERTS + ACTIONS
        # ==================================================
        st.subheader("🚨 Alerts & Recommended Actions")

        if "selected_action" not in st.session_state:
            st.session_state.selected_action = None

        alerts = results.get("alerts", [])

        if not alerts:
            st.success("No alerts triggered.")
        else:
            for alert in alerts:
                human_alert = format_alert_for_humans(alert)

                st.warning(f"**{human_alert['title']}**")
                st.write(human_alert["message"])

                if human_alert.get("affected_features"):
                    st.write(
                        "**Affected features:** "
                        + ", ".join(human_alert["affected_features"])
                    )

                for action in human_alert.get("actions", []):
                    if st.button(action["label"], key=action["id"]):
                        st.session_state.selected_action = action["id"]

        # ==================================================
        # ACTION DETAILS
        # ==================================================
        selected = st.session_state.get("selected_action")

        if selected == "review_features":
            st.subheader("📊 Feature Distribution Review")
            st.write("Inspect how feature distributions changed over time.")

        elif selected == "monitor_performance":
            st.subheader("📈 Performance Monitoring")
            st.write("Track performance metrics across monitoring cycles.")

        elif selected == "plan_retraining":
            st.subheader("🔁 Retraining Strategy")
            st.write(
                "Retraining is recommended if drift persists across "
                "multiple monitoring runs."
            )

    except Exception as e:
        st.error(f"Error: {e}")

# ==================================================
# ALERT HISTORY
# ==================================================
st.subheader("🕒 Alert History")

monitor = st.session_state.get("monitor")

if monitor:
    history = monitor.get_history(limit=10)

    for record in reversed(history):
        ts = record["timestamp"]
        for alert in record.get("alerts", []):
            human_alert = format_alert_for_humans(alert)
            with st.expander(f"{ts} — {human_alert['title']}"):
                st.write(human_alert["message"])
