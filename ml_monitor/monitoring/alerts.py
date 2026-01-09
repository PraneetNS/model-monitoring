"""
Alert Manager

Manages alerts and notifications for model monitoring.
All alert evaluation and severity classification lives here.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class AlertLevel(Enum):
    """
    Alert severity levels with extensible support.
    
    Maps to Python logging levels for integration.
    Supports standard levels (INFO, WARNING, CRITICAL) and can accept
    custom string levels that map to appropriate log levels.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def log_level(self) -> int:
        """
        Get the corresponding Python logging level.
        
        Returns:
            Python logging level constant
        """
        mapping = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "critical": logging.CRITICAL,
        }
        return mapping.get(self.value, logging.WARNING)

    @classmethod
    def from_string(cls, level_str: str) -> "AlertLevel":
        """
        Get AlertLevel from string value (case-insensitive).
        
        Supports both standard enum values and custom string levels.
        Custom levels default to WARNING log level.
        
        Args:
            level_str: String representation of alert level
            
        Returns:
            AlertLevel enum member (or creates a pseudo-member for custom levels)
            
        Example:
            >>> AlertLevel.from_string("info")  # Returns AlertLevel.INFO
            >>> AlertLevel.from_string("ERROR")  # Returns custom level mapped to ERROR
        """
        level_str_lower = level_str.lower()
        
        # Try to find matching enum member
        for level in cls:
            if level.value.lower() == level_str_lower:
                return level
        
        # For custom levels, create a pseudo-enum-like object
        # This allows extensibility while maintaining type safety for standard levels
        class CustomAlertLevel:
            def __init__(self, value: str, log_level: int = logging.WARNING):
                self.value = value.lower()
                self._log_level = log_level
            
            @property
            def log_level(self) -> int:
                return self._log_level
        
        # Map common custom levels to appropriate log levels
        custom_mappings = {
            "error": logging.ERROR,
            "debug": logging.DEBUG,
            "fatal": logging.CRITICAL,
        }
        
        log_level = custom_mappings.get(level_str_lower, logging.WARNING)
        return CustomAlertLevel(level_str, log_level)

    @staticmethod
    def get_log_level(level: Any) -> int:
        """
        Get Python logging level from AlertLevel enum or string.
        
        This method provides extensibility by accepting both AlertLevel
        enum members and string values.
        
        Args:
            level: AlertLevel enum member or string level name
            
        Returns:
            Python logging level constant
        """
        if isinstance(level, AlertLevel):
            return level.log_level
        elif isinstance(level, str):
            alert_level = AlertLevel.from_string(level)
            if hasattr(alert_level, "log_level"):
                return alert_level.log_level
            return logging.WARNING
        else:
            return logging.WARNING


@dataclass
class AlertThresholds:
    """
    Thresholds for built-in alert evaluation.

    Examples should configure these (or rely on defaults) and only call
    AlertManager; no threshold logic should live in example scripts.
    """

    accuracy: float = 0.7
    r2_score: float = 0.5
    prediction_shift: float = 0.2
    prediction_mean_threshold: float = 1.5  # For regression prediction mean anomalies
    drift_level: AlertLevel = AlertLevel.WARNING


class AlertManager:
    """
    Manages alerts based on monitoring results with integrated logging.
    """

    def __init__(
        self,
        thresholds: Optional[AlertThresholds] = None,
        logger: Optional[logging.Logger] = None,
        log_alerts: bool = True,
    ):
        """
        Initialize alert manager with configurable thresholds and logging.

        Args:
            thresholds: Optional AlertThresholds instance; uses defaults if not provided.
            logger: Optional Python logger instance; creates one if not provided.
            log_alerts: Whether to automatically log alerts to Python logging (default: True).
        """
        self.thresholds = thresholds or AlertThresholds()
        self.alert_rules: List[Callable] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.logger = logger or logging.getLogger(__name__)
        self.log_alerts = log_alerts

    def add_rule(self, rule: Callable) -> None:
        """
        Add a custom alert rule.

        Args:
            rule: Function that takes monitoring results and returns
                  alert dict or None
        """
        self.alert_rules.append(rule)

    def check_alerts(self, monitoring_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check monitoring results against alert rules.

        Args:
            monitoring_results: Results from monitoring run

        Returns:
            List of triggered alerts
        """
        alerts: List[Dict[str, Any]] = []

        # Check drift alerts
        if "drift" in monitoring_results:
            drift_results = monitoring_results["drift"]
            if drift_results.get("drift_detected", False):
                alerts.append(
                    {
                        "level": self.thresholds.drift_level.value,
                        "type": "data_drift",
                        "message": (
                            f"Data drift detected (score: "
                            f"{drift_results.get('drift_score', 0):.4f})"
                        ),
                        "details": drift_results,
                    }
                )

        # Check performance degradation
        if "performance" in monitoring_results:
            perf = monitoring_results["performance"]
            if "accuracy" in perf and perf["accuracy"] < self.thresholds.accuracy:
                alerts.append(
                    {
                        "level": AlertLevel.CRITICAL.value,
                        "type": "performance_degradation",
                        "message": f"Low accuracy detected: {perf['accuracy']:.4f}",
                        "details": perf,
                    }
                )
            elif "r2_score" in perf and perf["r2_score"] < self.thresholds.r2_score:
                alerts.append(
                    {
                        "level": AlertLevel.WARNING.value,
                        "type": "performance_degradation",
                        "message": f"Low R2 score detected: {perf['r2_score']:.4f}",
                        "details": perf,
                    }
                )

        # Check prediction shift
        if "prediction_shift" in monitoring_results:
            shift = abs(monitoring_results["prediction_shift"])
            if shift > self.thresholds.prediction_shift:
                alerts.append(
                    {
                        "level": AlertLevel.WARNING.value,
                        "type": "prediction_shift",
                        "message": f"Significant prediction shift detected: {shift:.4f}",
                        "details": {"shift": monitoring_results["prediction_shift"]},
                    }
                )

        # Check prediction mean anomaly (for regression)
        if "predictions" in monitoring_results:
            pred_mean = monitoring_results["predictions"].get("mean", 0)
            if abs(pred_mean) > self.thresholds.prediction_mean_threshold:
                alerts.append(
                    {
                        "level": AlertLevel.CRITICAL.value,
                        "type": "prediction_mean_anomaly",
                        "message": f"Prediction mean is unusually high: {pred_mean:.4f}",
                        "details": monitoring_results["predictions"],
                    }
                )

        # Run custom rules (if provided)
        # Note: Custom rules should not contain threshold logic - use AlertThresholds instead
        for rule in self.alert_rules:
            alert = rule(monitoring_results)
            if alert:
                alerts.append(alert)

        # Store and log alerts
        for alert in alerts:
            alert["timestamp"] = monitoring_results.get("timestamp")
            self.alert_history.append(alert)
            
            # Log alert if logging is enabled
            if self.log_alerts:
                self._log_alert(alert)

        return alerts

    def _log_alert(self, alert: Dict[str, Any]) -> None:
        """
        Log an alert at the appropriate log level.
        
        Supports both standard AlertLevel enum values and custom string levels.
        
        Args:
            alert: Alert dictionary with 'level' and 'message' keys
        """
        try:
            level_str = alert.get("level", "warning")
            log_level = AlertLevel.get_log_level(level_str)
            
            message = alert.get("message", "Alert triggered")
            alert_type = alert.get("type", "unknown")
            
            # Format log message
            log_message = f"[{alert_type.upper()}] {message}"
            
            # Include details if available (truncate if too long)
            if "details" in alert:
                details_str = str(alert["details"])
                if len(details_str) > 200:
                    details_str = details_str[:200] + "..."
                log_message += f" | Details: {details_str}"
            
            # Log at appropriate level
            self.logger.log(log_level, log_message)
            
        except Exception as e:
            # Fallback to warning if level parsing fails
            self.logger.warning(f"Failed to log alert: {e} | Alert: {alert}")

    def get_alert_history(self, level: Optional[AlertLevel] = None) -> List[Dict[str, Any]]:
        """
        Get alert history.

        Args:
            level: Filter by alert level (optional)

        Returns:
            List of alerts
        """
        if level:
            return [alert for alert in self.alert_history if alert["level"] == level.value]
        return self.alert_history
