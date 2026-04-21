import warnings

from .anomaly import anomaly


def find_anomaly(*args, **kwargs):
    """Backward-compatible wrapper for the pre-v2 anomaly entry point."""
    warnings.warn(
        "traffic_anomaly.find_anomaly.find_anomaly() is deprecated; use traffic_anomaly.anomaly.anomaly() or traffic_anomaly.anomaly() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return anomaly(*args, **kwargs)
