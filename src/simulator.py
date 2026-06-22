"""
2-DOF robot arm physics simulator.

Generates ground-truth joint trajectories and simulates noisy sensor readings:
  - Encoder: quantised position (2048 counts/rev)
  - IMU (gyro): velocity with additive white Gaussian noise + bias drift

Typical usage
-------------
sim = RobotArmSimulator(dt=0.01, seed=42)
for t, state, sensors in sim.run(duration=10.0, torque_fn=my_torque_fn):
    ...
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Generator


# ── Physical parameters ────────────────────────────────────────────────────────
INERTIA  = np.array([0.15, 0.08])   # kg·m²
DAMPING  = np.array([0.05, 0.03])   # N·m·s/rad
GRAVITY_COEFF = np.array([0.12, 0.06])  # gravity torque coefficient [N·m/rad approximation]

# ── Sensor parameters ──────────────────────────────────────────────────────────
ENCODER_COUNTS_PER_REV = 2048
ENCODER_STD_RAD = (2 * np.pi / ENCODER_COUNTS_PER_REV)  # quantisation noise ~0.003 rad

IMU_NOISE_STD  = 0.03    # rad/s white noise
IMU_BIAS_DRIFT = 0.001   # rad/s² bias random walk


@dataclass
class TrueState:
    t:    float
    q:    np.ndarray   # [q1, q2]  rad
    qd:   np.ndarray   # [q1d, q2d] rad/s
    qdd:  np.ndarray   # [q1dd, q2dd] rad/s²
    tau:  np.ndarray   # applied torque


@dataclass
class SensorReading:
    t:       float
    enc:     np.ndarray   # noisy quantised position [q1, q2]
    gyro:    np.ndarray   # noisy velocity [q1d, q2d]
    # fault flags for testing safety interlock
    enc_fault:  bool = False
    gyro_fault: bool = False


class RobotArmSimulator:
    """
    Simulates a 2-DOF planar robot arm with realistic sensor noise.

    Parameters
    ----------
    dt      : integration step [s]
    seed    : RNG seed for reproducibility
    """

    def __init__(self, dt: float = 0.01, seed: int = 0):
        self.dt   = dt
        self.rng  = np.random.default_rng(seed)
        self._imu_bias = np.zeros(2)   # accumulated gyro bias

    def reset(self, q0=(0.0, 0.0), qd0=(0.0, 0.0)) -> None:
        self._q   = np.array(q0,  dtype=float)
        self._qd  = np.array(qd0, dtype=float)
        self._imu_bias = np.zeros(2)

    # ── Dynamics ───────────────────────────────────────────────────────────────

    def _dynamics(self, q: np.ndarray, qd: np.ndarray,
                  tau: np.ndarray) -> np.ndarray:
        """
        Returns joint accelerations [rad/s²].

        Model:  I * qdd + b * qd + g(q) = tau
        Gravity term uses first-order linearisation:  g(q) ≈ g_coeff * sin(q)
        """
        g = GRAVITY_COEFF * np.sin(q)
        qdd = (tau - DAMPING * qd - g) / INERTIA
        return qdd

    def _rk4_step(self, q: np.ndarray, qd: np.ndarray,
                  tau: np.ndarray, dt: float):
        """4th-order Runge-Kutta integration."""
        def deriv(q_, qd_):
            return qd_, self._dynamics(q_, qd_, tau)

        k1q, k1v = deriv(q, qd)
        k2q, k2v = deriv(q + 0.5*dt*k1q, qd + 0.5*dt*k1v)
        k3q, k3v = deriv(q + 0.5*dt*k2q, qd + 0.5*dt*k2v)
        k4q, k4v = deriv(q + dt*k3q,     qd + dt*k3v)

        q_new  = q  + (dt/6) * (k1q + 2*k2q + 2*k3q + k4q)
        qd_new = qd + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
        return q_new, qd_new

    # ── Sensor models ──────────────────────────────────────────────────────────

    def _read_encoder(self, q_true: np.ndarray) -> np.ndarray:
        """Quantise angle to encoder resolution + add Gaussian noise."""
        q_counts = np.round(q_true / (2 * np.pi) * ENCODER_COUNTS_PER_REV)
        q_quant  = q_counts * (2 * np.pi / ENCODER_COUNTS_PER_REV)
        return q_quant + self.rng.normal(0, ENCODER_STD_RAD * 0.1, size=2)

    def _read_gyro(self, qd_true: np.ndarray) -> np.ndarray:
        """IMU gyro: white noise + slowly drifting bias."""
        self._imu_bias += self.rng.normal(0, IMU_BIAS_DRIFT * self.dt, size=2)
        noise = self.rng.normal(0, IMU_NOISE_STD, size=2)
        return qd_true + noise + self._imu_bias

    # ── Main generator ─────────────────────────────────────────────────────────

    def run(
        self,
        duration: float,
        torque_fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        q0=(0.0, 0.0),
        qd0=(0.0, 0.0),
        fault_schedule: dict | None = None,
    ) -> Generator[tuple[TrueState, SensorReading], None, None]:
        """
        Simulate the arm for `duration` seconds.

        Parameters
        ----------
        duration      : total simulation time [s]
        torque_fn     : callable(t, q, qd) -> [tau1, tau2]
        q0, qd0       : initial joint angles and velocities
        fault_schedule: {time: 'enc_fault'|'gyro_fault'} for interlock testing

        Yields
        ------
        (TrueState, SensorReading) at each time step
        """
        self.reset(q0, qd0)
        fault_schedule = fault_schedule or {}

        q, qd = self._q.copy(), self._qd.copy()
        t = 0.0
        steps = int(duration / self.dt)

        for _ in range(steps):
            tau = np.asarray(torque_fn(t, q, qd), dtype=float)
            qdd = self._dynamics(q, qd, tau)

            true = TrueState(t=t, q=q.copy(), qd=qd.copy(), qdd=qdd, tau=tau)

            # Determine active faults
            enc_fault  = fault_schedule.get(round(t, 3)) == 'enc_fault'
            gyro_fault = fault_schedule.get(round(t, 3)) == 'gyro_fault'

            enc  = np.array([np.nan, np.nan]) if enc_fault  else self._read_encoder(q)
            gyro = np.array([np.nan, np.nan]) if gyro_fault else self._read_gyro(qd)

            sensors = SensorReading(t=t, enc=enc, gyro=gyro,
                                    enc_fault=enc_fault, gyro_fault=gyro_fault)

            yield true, sensors

            # Advance physics
            q, qd = self._rk4_step(q, qd, tau, self.dt)
            t += self.dt


# ── Built-in torque profiles for demos / tests ────────────────────────────────

def sinusoidal_torque(t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Slow sinusoidal excitation — good for filter tuning demo."""
    return np.array([
        0.3 * np.sin(0.5 * t),
        0.2 * np.sin(0.8 * t + 0.5),
    ])


def step_torque(t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Step inputs to excite transient dynamics."""
    tau1 = 0.5 if t < 3.0 else -0.3
    tau2 = 0.3 if t < 5.0 else  0.4
    return np.array([tau1, tau2])


def velocity_ramp_torque(t: float, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """Ramp up until velocity limit is breached — triggers PLC interlock."""
    return np.array([0.8 * min(t / 5.0, 1.0), 0.6 * min(t / 4.0, 1.0)])
