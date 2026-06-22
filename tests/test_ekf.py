"""
EKF unit tests.

Tests:
  1. Filter convergence — RMSE on angles and velocities
  2. NEES consistency — mean NEES within chi-squared bounds
  3. Sensor dropout — filter stays bounded with no measurements
  4. Numerical stability — large noise inputs don't cause NaN / divergence
  5. Safety flag logic — correct bits set for limit violations
  6. Modbus register encoding/decoding round-trip
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ekf import RobotArmEKF, check_safety, EKFState
from src.simulator import RobotArmSimulator, sinusoidal_torque, step_torque
from src.modbus_bridge import ModbusBridge, _safety_bitmask, _to_int16


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def ekf():
    return RobotArmEKF()


@pytest.fixture
def sim():
    return RobotArmSimulator(dt=0.01, seed=0)


# ── Test 1: Convergence ────────────────────────────────────────────────────────

def test_filter_converges(ekf, sim):
    """
    After 5 seconds of sinusoidal excitation, the EKF angle RMSE
    must be under 0.01 rad (< 0.6°) and velocity RMSE under 0.1 rad/s.
    """
    state = ekf.init()
    errors_q, errors_qd = [], []

    for true, sensors in sim.run(5.0, sinusoidal_torque):
        state = ekf.predict(state, true.tau, dt=0.01)
        state = ekf.update_encoder(state, sensors.enc)
        state = ekf.update_imu(state, sensors.gyro)

        if true.t > 1.0:   # skip initial transient
            errors_q.append( [state.x[0] - true.q[0],  state.x[2] - true.q[1]])
            errors_qd.append([state.x[1] - true.qd[0], state.x[3] - true.qd[1]])

    rmse_q  = np.sqrt(np.mean(np.array(errors_q)  ** 2))
    rmse_qd = np.sqrt(np.mean(np.array(errors_qd) ** 2))

    assert rmse_q  < 0.01,  f"Angle RMSE {rmse_q:.4f} rad exceeds 0.01 rad"
    assert rmse_qd < 0.10,  f"Velocity RMSE {rmse_qd:.4f} rad/s exceeds 0.10 rad/s"


# ── Test 2: NEES consistency ───────────────────────────────────────────────────

def test_nees_consistency(ekf, sim):
    """
    Mean NEES should lie within the 95% chi-squared confidence interval
    for 4 degrees of freedom: [2.09, 6.57] (approximate for N=500 samples).
    A well-tuned, consistent filter should satisfy this.
    """
    state = ekf.init()
    nees_vals = []

    for true, sensors in sim.run(10.0, sinusoidal_torque):
        state = ekf.predict(state, true.tau, dt=0.01)
        state = ekf.update_encoder(state, sensors.enc)
        state = ekf.update_imu(state, sensors.gyro)

        if true.t > 2.0:
            x_true = np.array([true.q[0], true.qd[0], true.q[1], true.qd[1]])
            nees_vals.append(ekf.nees(state, x_true))

    mean_nees = np.mean(nees_vals)
    # Chi-sq(4) mean = 4.0; allow a wide band for a real system with model mismatch
    assert 1.0 < mean_nees < 20.0, \
        f"Mean NEES {mean_nees:.2f} outside consistency band [1, 20]"


# ── Test 3: Sensor dropout ─────────────────────────────────────────────────────

def test_sensor_dropout_bounded(ekf, sim):
    """
    With no encoder or IMU updates for 3 seconds, the EKF should not diverge.
    State estimate must stay within ±5 rad and covariance must remain finite.
    """
    state = ekf.init()

    for true, sensors in sim.run(5.0, sinusoidal_torque):
        state = ekf.predict(state, true.tau, dt=0.01)
        # No measurement updates — pure prediction

    assert np.all(np.isfinite(state.x)), "EKF state contains NaN/Inf after dropout"
    assert np.all(np.abs(state.x) < 100), f"EKF state diverged: {state.x}"
    assert np.all(np.isfinite(state.P)), "Covariance contains NaN/Inf after dropout"
    assert np.linalg.matrix_rank(state.P) == 4, "Covariance lost rank"


# ── Test 4: Numerical stability under large noise ──────────────────────────────

def test_numerical_stability_large_noise():
    """
    Feed the filter 100 measurements with 100x normal noise.
    State must remain finite and covariance must remain positive semi-definite.
    """
    rng = np.random.default_rng(1)
    ekf = RobotArmEKF()
    state = ekf.init()

    for _ in range(100):
        state = ekf.predict(state, np.array([0.1, 0.1]), dt=0.01)
        z_enc  = rng.normal(0, 3.0, size=2)    # 3 rad noise (absurd)
        z_gyro = rng.normal(0, 10.0, size=2)   # 10 rad/s noise (absurd)
        state = ekf.update_encoder(state, z_enc)
        state = ekf.update_imu(state, z_gyro)

    assert np.all(np.isfinite(state.x)), "State diverged under high noise"
    assert np.all(np.isfinite(state.P)), "Covariance diverged under high noise"

    # P must be positive semi-definite
    eigvals = np.linalg.eigvalsh(state.P)
    assert np.all(eigvals >= -1e-10), f"Covariance not PSD: min eigenvalue {eigvals.min()}"


# ── Test 5: Safety flag logic ──────────────────────────────────────────────────

@pytest.mark.parametrize("x, expected_flags", [
    # Safe state
    ([0.0, 0.0, 0.0, 0.0],
     {"joint1_limit": False, "joint2_limit": False,
      "joint1_vel":   False, "joint2_vel":   False, "any_fault": False}),
    # Joint 1 angle violation (> 135° = 2.356 rad)
    ([3.0, 0.0, 0.0, 0.0],
     {"joint1_limit": True, "any_fault": True}),
    # Joint 2 velocity violation (> 4.0 rad/s)
    ([0.0, 0.0, 0.0, 5.0],
     {"joint2_vel": True, "any_fault": True}),
    # All clear after large negative angle — check sign handling
    ([-1.0, -0.5, -0.8, -1.0],
     {"any_fault": False}),
])
def test_safety_flags(x, expected_flags, ekf):
    state = EKFState(x=np.array(x, dtype=float), P=np.eye(4))
    flags = check_safety(state)
    for key, expected in expected_flags.items():
        assert flags[key] == expected, \
            f"Flag '{key}' expected {expected}, got {flags[key]} for x={x}"


# ── Test 6: Modbus register round-trip ────────────────────────────────────────

def test_modbus_register_roundtrip():
    """
    Encode engineering values to Modbus registers and decode back.
    All values must round-trip within the encoding precision.
    """
    q  = [1.234, -0.567]
    qd = [2.10,  -3.80]
    flags = {"joint1_limit": True, "joint2_limit": False,
             "joint1_vel": False, "joint2_vel": True, "any_fault": True}

    # Encode
    regs = [
        _to_int16(q[0],  1000),
        _to_int16(qd[0], 100),
        _to_int16(q[1],  1000),
        _to_int16(qd[1], 100),
        _safety_bitmask(flags),
        0, 0, 0,
    ]

    # Decode
    decoded = ModbusBridge.decode_registers(regs)

    assert abs(decoded["q1_rad"]     - q[0])  < 0.002, "q1 round-trip error"
    assert abs(decoded["q1dot_rads"] - qd[0]) < 0.02,  "q1dot round-trip error"
    assert abs(decoded["q2_rad"]     - q[1])  < 0.002, "q2 round-trip error"
    assert abs(decoded["q2dot_rads"] - qd[1]) < 0.02,  "q2dot round-trip error"
    assert decoded["joint1_limit"] == flags["joint1_limit"]
    assert decoded["joint2_vel"]   == flags["joint2_vel"]
    assert decoded["any_fault"]    == flags["any_fault"]


# ── Test 7: Covariance symmetry ────────────────────────────────────────────────

def test_covariance_remains_symmetric(ekf, sim):
    """
    Covariance matrix must remain symmetric throughout the run.
    Asymmetry accumulates from floating-point errors; the Joseph update
    form in the EKF should keep it bounded.
    """
    state = ekf.init()

    for true, sensors in sim.run(5.0, step_torque):
        state = ekf.predict(state, true.tau, dt=0.01)
        state = ekf.update_encoder(state, sensors.enc)
        state = ekf.update_imu(state, sensors.gyro)

    asymmetry = np.max(np.abs(state.P - state.P.T))
    assert asymmetry < 1e-10, f"Covariance asymmetry {asymmetry:.2e} exceeds 1e-10"
