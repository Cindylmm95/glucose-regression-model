from __future__ import annotations

from collections import OrderedDict
from datetime import time as clock_time
from typing import Iterable

import numpy as np


HISTORY_LENGTH = 24
DIFF_LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12, 24)


def _validate_history(readings: Iterable[float]) -> np.ndarray:
    history = np.asarray(list(readings), dtype=float)

    if history.shape != (HISTORY_LENGTH,):
        raise ValueError("The sequence must contain exactly 24 readings.")
    if not np.isfinite(history).all():
        raise ValueError("The sequence contains non-numeric values.")
    if np.any((history < 40) | (history > 400)):
        raise ValueError("Readings must remain between 40 and 400 mg/dL.")

    return history


def _slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    return float(np.dot(values, centered) / denominator)


def build_feature_row(
    readings: Iterable[float],
    current_time: clock_time,
) -> OrderedDict[str, float]:
    history = _validate_history(readings)
    features: OrderedDict[str, float] = OrderedDict()

    for lag in range(1, HISTORY_LENGTH + 1):
        features[f"lag{lag}"] = float(history[-lag])

    minutes_since_midnight = current_time.hour * 60 + current_time.minute
    angle = 2 * np.pi * minutes_since_midnight / 1440

    features["tod_sin"] = float(np.sin(angle))
    features["tod_cos"] = float(np.cos(angle))

    for lag in DIFF_LAGS:
        features[f"diff{lag}"] = float(history[-1] - history[-1 - lag])

    for window in ROLLING_WINDOWS:
        segment = history[-window:]
        features[f"mean{window}"] = float(segment.mean())
        features[f"std{window}"] = float(segment.std(ddof=0))
        features[f"slope{window}"] = _slope(segment)
        features[f"min{window}"] = float(segment.min())
        features[f"max{window}"] = float(segment.max())

    if len(features) != 51:
        raise RuntimeError("Feature engineering did not generate 51 variables.")

    return features
