from __future__ import annotations

import numpy as np


def generate_sequence(
    current_glucose: float,
    scenario: str,
    intensity: float,
    variability: float,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, 24)

    if scenario == "Estable":
        base = np.full(24, current_glucose)
        wave = np.sin(t * 3 * np.pi) * variability * 0.5

    elif scenario == "Ascenso gradual":
        total_change = 22.0 * intensity
        base = current_glucose - total_change * (1 - t**1.15)
        wave = np.sin(t * 2 * np.pi) * variability * 0.35

    elif scenario == "Descenso gradual":
        total_change = 22.0 * intensity
        base = current_glucose + total_change * (1 - t**1.15)
        wave = np.sin(t * 2 * np.pi) * variability * 0.35

    elif scenario == "Variación rápida":
        total_change = 30.0 * intensity
        base = current_glucose - total_change * (1 - t**2.8)
        wave = np.sin(t * 5 * np.pi) * variability

    else:
        raise ValueError("El escenario seleccionado no es válido.")

    noise = rng.normal(0.0, variability * 0.18, size=24)
    sequence = base + wave + noise
    sequence[-1] = current_glucose
    sequence = np.clip(sequence, 40.0, 400.0)

    return np.round(sequence, 1).tolist()
