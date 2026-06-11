"""Unit tests for SensorSimulator — spike/freeze/drift anomaly injection.

Regression coverage for P1-3: inject_anomaly("spike") set _spike_next
but generate_reading() never consumed it, so spikes were never emitted.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Load from exact path — simulator/ is not a package
_SIM_PATH = (
    Path(__file__).parent.parent.parent / "simulator" / "sensor_simulator.py"
)
_spec = importlib.util.spec_from_file_location("sensor_simulator_module", _SIM_PATH)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["sensor_simulator_module"] = _mod
_spec.loader.exec_module(_mod)
SensorSimulator = _mod.SensorSimulator

_CONFIG_PATH = str(
    Path(__file__).parent.parent.parent / "simulator" / "config" / "node1.yaml"
)


@pytest.fixture
def sim():
    # No db_path → no SQLite side effects
    return SensorSimulator(_CONFIG_PATH)


def _profile_bounds(sim_obj):
    return {
        name: (float(p["min"]), float(p["max"]))
        for name, p in sim_obj._sensor_profile.items()
    }


def test_normal_reading_within_bounds(sim):
    reading = sim.generate_reading()
    for sensor, (lo, hi) in _profile_bounds(sim).items():
        assert lo <= reading["sensors"][sensor] <= hi


def test_spike_produces_out_of_bounds_values(sim):
    sim.inject_anomaly("spike")
    reading = sim.generate_reading()
    for sensor, (lo, hi) in _profile_bounds(sim).items():
        assert reading["sensors"][sensor] > hi, (
            f"{sensor}: expected spike above {hi}, got {reading['sensors'][sensor]}"
        )


def test_spike_is_one_shot(sim):
    sim.inject_anomaly("spike")
    sim.generate_reading()                      # consumes the spike
    reading = sim.generate_reading()            # back to normal
    for sensor, (lo, hi) in _profile_bounds(sim).items():
        assert lo <= reading["sensors"][sensor] <= hi


def test_reset_anomaly_clears_pending_spike(sim):
    sim.inject_anomaly("spike")
    sim.reset_anomaly()
    reading = sim.generate_reading()
    for sensor, (lo, hi) in _profile_bounds(sim).items():
        assert lo <= reading["sensors"][sensor] <= hi


def test_freeze_repeats_identical_values(sim):
    sim.inject_anomaly("freeze")
    r1 = sim.generate_reading()
    r2 = sim.generate_reading()
    assert r1["sensors"] == r2["sensors"]
    sim.reset_anomaly()
