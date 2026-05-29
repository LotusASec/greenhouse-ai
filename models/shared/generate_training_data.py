#!/usr/bin/env python3
"""
Shared training data generator for Phase 2A.

Generates synthetic sensor data CSVs using rule-based labels.

Usage:
    python generate_training_data.py
    python generate_training_data.py --output-dir /path/to/data
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("data-generator")

# TR-11 feature vector order (FROZEN — do not reorder)
IRRIGATION_FEATURES = ["temperature", "humidity", "soil_moisture", "light", "ec", "ph"]
NUTRITION_FEATURES  = ["ec", "ph", "temperature", "humidity"]
ANOMALY_FEATURES    = ["temperature", "humidity", "soil_moisture", "light", "ec", "ph"]


def _clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


# ---------------------------------------------------------------------------
# Irrigation data
# ---------------------------------------------------------------------------

def generate_irrigation_data(
    n_train: int = 8000, n_test: int = 2000, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uniform sensor sampling across full operating range for class balance."""
    rng = np.random.RandomState(seed)
    n = n_train + n_test

    temperature   = rng.uniform(18.0, 35.0, n)
    humidity      = rng.uniform(40.0, 90.0, n)
    soil_moisture = rng.uniform(20.0, 80.0, n)
    light         = rng.uniform(0.0, 1000.0, n)
    ec            = rng.uniform(1.0, 3.5, n)
    ph            = rng.uniform(5.5, 7.0, n)

    # Ground truth labels — TR-07
    irrigate = (soil_moisture < 35) | ((soil_moisture < 50) & (temperature > 28))
    amount_liters = np.maximum(0.0, (50.0 - soil_moisture) * 0.15)
    amount_liters = np.where(irrigate, amount_liters, 0.0)

    df = pd.DataFrame({
        "temperature":   temperature,
        "humidity":      humidity,
        "soil_moisture": soil_moisture,
        "light":         light,
        "ec":            ec,
        "ph":            ph,
        "irrigate":      irrigate.astype(int),
        "amount_liters": amount_liters,
    })

    pos_rate = irrigate.mean()
    log.info(
        "Irrigation labels — irrigate=True: %.1f%%, irrigate=False: %.1f%%",
        pos_rate * 100, (1 - pos_rate) * 100,
    )

    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Nutrition data
# ---------------------------------------------------------------------------

def generate_nutrition_data(
    n_train: int = 8000, n_test: int = 2000, seed: int = 43
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wider ec/ph range to ensure every class exceeds 5%."""
    rng = np.random.RandomState(seed)
    n = n_train + n_test

    # Extended range so K_deficiency (ph>6.8, ec<1.8) reaches ≥5%
    ec          = rng.uniform(0.5, 4.0, n)
    ph          = rng.uniform(4.5, 8.0, n)
    temperature = rng.uniform(18.0, 35.0, n)
    humidity    = rng.uniform(40.0, 90.0, n)

    # Ground truth labels — TR-07 (priority order matters)
    labels = np.where(
        ec < 1.2,
        "N_deficiency",
        np.where(
            ph < 5.8,
            "P_deficiency",
            np.where(
                (ph > 6.8) & (ec < 1.8),
                "K_deficiency",
                "normal",
            ),
        ),
    )

    df = pd.DataFrame({
        "ec":          ec,
        "ph":          ph,
        "temperature": temperature,
        "humidity":    humidity,
        "label":       labels,
    })

    dist = pd.Series(labels).value_counts(normalize=True).sort_index()
    log.info("Nutrition label distribution:\n%s", dist.to_string())
    min_pct = dist.min() * 100
    if min_pct < 5.0:
        log.warning("Min class %.1f%% < 5%% — consider adjusting ranges", min_pct)

    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Anomaly data
# ---------------------------------------------------------------------------

def _gen_normal_sensors(n: int, rng: np.random.RandomState) -> pd.DataFrame:
    """
    Uniform sensor data across the full config operating range.
    Uniform training ensures IF learns the complete normal space
    and doesn't flag edge-of-range values as anomalies.
    """
    return pd.DataFrame({
        "temperature":   rng.uniform(18.0, 35.0, n),
        "humidity":      rng.uniform(40.0, 90.0, n),
        "soil_moisture": rng.uniform(20.0, 80.0, n),
        "light":         rng.uniform(0.0, 1000.0, n),
        "ec":            rng.uniform(1.0,  3.5, n),
        "ph":            rng.uniform(5.5,  7.0, n),
    })


def generate_anomaly_data(
    n_normal: int = 9500,
    n_test_normal: int = 475,
    n_inject: int = 25,
    seed: int = 44,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normal training data: Gaussian within realistic bounds.
    Injected anomalies: values far outside normal operating range.
    """
    rng_train = np.random.RandomState(seed)
    rng_test  = np.random.RandomState(seed + 1)
    rng_inj   = np.random.RandomState(seed + 2)

    train_normal = _gen_normal_sensors(n_normal, rng_train)
    test_normal  = _gen_normal_sensors(n_test_normal, rng_test)

    # Five anomaly types (5 rows each for n_inject=25)
    n_each = n_inject // 5
    n_rem  = n_inject - n_each * 5

    # Injected anomalies — multi-feature extreme values.
    # Single-feature extremes are not reliably detected by IsolationForest
    # in 6-dimensional space (1/6 feature selection probability per tree node).
    # Multi-feature extremes ensure all trees quickly isolate the anomaly.
    # Each anomaly type has ≥3 features far outside the training distribution.
    # IsolationForest in 6-d space requires multiple extreme dimensions to
    # reliably isolate a point regardless of random feature selection.
    frames = []

    # 1 — Drought extreme: hot + low humidity + no water (3 extreme)
    f1 = pd.DataFrame({
        "temperature":   rng_inj.uniform(52.0, 65.0, n_each),
        "humidity":      rng_inj.uniform(2.0,  8.0,  n_each),
        "soil_moisture": rng_inj.uniform(1.0,  5.0,  n_each),
        "light":         rng_inj.uniform(900.0, 1200.0, n_each),
        "ec":            rng_inj.uniform(2.0,   2.5,  n_each),
        "ph":            rng_inj.uniform(6.0,   6.5,  n_each),
    })
    frames.append(f1)

    # 2 — Chemical + heat spike: high temp + low humidity + extreme EC + extreme pH (4 extreme)
    f2 = pd.DataFrame({
        "temperature":   rng_inj.uniform(50.0, 65.0, n_each),
        "humidity":      rng_inj.uniform(2.0,  8.0,  n_each),
        "soil_moisture": rng_inj.uniform(54.0, 56.0, n_each),
        "light":         rng_inj.uniform(695.0, 705.0, n_each),
        "ec":            rng_inj.uniform(8.0,  12.0, n_each),
        "ph":            rng_inj.uniform(1.5,   3.0,  n_each),
    })
    frames.append(f2)

    # 3 — Multi-sensor failure: high temp + low humidity + low soil + high EC (4 extreme)
    f3 = pd.DataFrame({
        "temperature":   rng_inj.uniform(55.0, 70.0, n_each),
        "humidity":      rng_inj.uniform(2.0,  8.0,  n_each),
        "soil_moisture": rng_inj.uniform(1.0,  5.0,  n_each),
        "light":         rng_inj.uniform(695.0, 705.0, n_each),
        "ec":            rng_inj.uniform(7.0,  10.0, n_each),
        "ph":            rng_inj.uniform(6.2,   6.4,  n_each),
    })
    frames.append(f3)

    # 4 — All-high extreme (flooding + heat + over-fertilization + alkaline, 6 extreme)
    f4 = pd.DataFrame({
        "temperature":   rng_inj.uniform(48.0, 58.0,   n_each),
        "humidity":      rng_inj.uniform(95.0, 105.0,  n_each),
        "soil_moisture": rng_inj.uniform(90.0, 100.0,  n_each),
        "light":         rng_inj.uniform(1800.0, 2200.0, n_each),
        "ec":            rng_inj.uniform(7.0,  10.0,   n_each),
        "ph":            rng_inj.uniform(10.0, 13.0,   n_each),
    })
    frames.append(f4)

    # 5 — All-low extreme (sensor failure / freezing / total depletion, 6 extreme)
    n5 = n_each + n_rem
    f5 = pd.DataFrame({
        "temperature":   rng_inj.uniform(1.0,  5.0, n5),
        "humidity":      rng_inj.uniform(1.0,  5.0, n5),
        "soil_moisture": rng_inj.uniform(1.0,  5.0, n5),
        "light":         rng_inj.uniform(0.0,  1.0, n5),
        "ec":            rng_inj.uniform(0.0,  0.1, n5),
        "ph":            rng_inj.uniform(1.0,  2.0, n5),
    })
    frames.append(f5)

    test_inject = pd.concat(frames, ignore_index=True)

    log.info(
        "Anomaly data — train_normal: %d, test_normal: %d, injected: %d",
        len(train_normal), len(test_normal), len(test_inject),
    )
    return train_normal, test_normal, test_inject


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 2A training data")
    parser.add_argument("--output-dir", default="data", help="Directory for CSV files")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Generating irrigation data...")
    irr_train, irr_test = generate_irrigation_data()
    irr_train.to_csv(out / "irrigation_train.csv", index=False)
    irr_test.to_csv(out / "irrigation_test.csv", index=False)

    log.info("Generating nutrition data...")
    nut_train, nut_test = generate_nutrition_data()
    nut_train.to_csv(out / "nutrition_train.csv", index=False)
    nut_test.to_csv(out / "nutrition_test.csv", index=False)

    log.info("Generating anomaly data...")
    anm_train, anm_test_normal, anm_test_inject = generate_anomaly_data()
    anm_train.to_csv(out / "anomaly_normal.csv", index=False)
    anm_test_normal.to_csv(out / "anomaly_test_normal.csv", index=False)
    anm_test_inject.to_csv(out / "anomaly_test_inject.csv", index=False)

    log.info("Done. Files written to %s/", out)
    for f in sorted(out.iterdir()):
        rows = pd.read_csv(f).shape[0]
        log.info("  %-32s %5d rows", f.name, rows)


if __name__ == "__main__":
    main()
