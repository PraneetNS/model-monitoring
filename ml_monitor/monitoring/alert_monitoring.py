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
