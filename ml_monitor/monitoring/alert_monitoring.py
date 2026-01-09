def format_alert_for_humans(alert: dict) -> dict:
    alert_type = alert.get("type", "unknown")
    level = alert.get("level", "info")

    if alert_type == "data_drift":
        feature_results = alert["details"].get("feature_results", {})
        drifted_features = [
            f for f, v in feature_results.items() if v.get("drift_detected")
        ]

        return {
            "level": level,
            "title": "Data Change Detected",
            "message": (
                "The incoming data is different from what the model was trained on."
            ),
            "affected_features": drifted_features[:5],
            "recommendation": "Review data trends and continue monitoring.",
            "actions": [
                {"id": "review_features", "label": "Review feature distributions"},
                {"id": "monitor_performance", "label": "Monitor model performance"},
                {"id": "plan_retraining", "label": "Plan retraining if drift continues"},
            ],
        }

    return {
        "level": level,
        "title": "System Notification",
        "message": "An event was detected in the monitoring system.",
        "actions": [],
    }
def format_alert_for_humans(alert):
    alert_type = alert.get("type", "system")
    details = alert.get("details", {})

    if alert_type == "prediction_mean_anomaly":
        mean = details.get("mean", 0)

        return {
            "title": "Unusual Prediction Trend",
            "message": (
                f"The model is predicting higher values than usual "
                f"(average prediction ≈ {mean:.2f}). "
                "This may indicate a shift in real-world conditions."
            ),
            "recommended_action": "Review recent data trends and continue monitoring.",
        }

    if alert_type == "data_drift":
        features = list(details.get("feature_results", {}).keys())

        return {
            "title": "Data Pattern Change Detected",
            "message": (
                "Incoming data looks different from historical data. "
                "This suggests the environment has changed."
            ),
            "affected_features": features[:5],
            "recommended_action": "Inspect feature distributions and monitor performance.",
        }

    return {
        "title": "System Notification",
        "message": "An event was detected in the monitoring system.",
        "recommended_action": "No immediate action required.",
    }
