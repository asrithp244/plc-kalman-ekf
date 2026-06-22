"""
Extended Kalman Filter for 2-DOF robot arm joint state estimation.

State vector x = [q1, q1_dot, q2, q2_dot]  (angles [rad], velocities [rad/s])

Process model: rigid-body Euler integration with simplified friction.
    q_dot_next  = q_dot + dt * (tau - b*q_dot) / I
    q_next      = q + dt * q_dot

Measurement model:
    Encoder:    z_enc = [q1, q2]                        (position only)
    IMU:        z_imu = [q1_dot, q2_dot]                (velocity proxy via differentiation)

Both sensor streams are fused sequentially in the update step.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Physical constants (single rigid link approximation per joint) ─────────────
INERTIA   = np.array([0.15, 0.08])   # kg·m²  — link 1, link 2
DAMPING   = np.array([0.05, 0.03])   # N·m·s/rad — viscous friction
DT_NOM    = 0.01                      # nominal control step [s]

# ── Noise tuning ───────────────────────────────────────────────────────────────
# Process noise (model uncertainty): [q1, q1d, q2, q2d]
Q_DIAG = np.array([1e-5, 1e-3, 1e-5, 1e-3])

# Encoder measurement noise: [q1, q2]  (std ≈ 0.001 rad ≈ 0.06°)
R_ENC_DIAG = np.array([1e-6, 1e-6])

# IMU (gyro) measurement noise: [q1_dot, q2_dot]  (std ≈ 0.03 rad/s)
R_IMU_DIAG = np.array([9e-4, 9e-4])


@dataclass
class EKFState:
    """Mutable EKF state passed between prediction/update steps."""
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))   # [q1, q1d, q2, q2d]
    P: np.ndarray = field(default_factory=lambda: np.eye(4) * 0.1)
    t: float = 0.0   # last update time [s]


class RobotArmEKF:
    """
    EKF for a 2-DOF serial robot arm.

    Usage
    -----
    ekf = RobotArmEKF()
    state = ekf.init(q0=[0.0, 0.0], q_dot0=[0.0, 0.0])

    for each control cycle:
        state = ekf.predict(state, torque, dt)
        if encoder_available:
            state = ekf.update_encoder(state, z_enc)
        if imu_available:
            state = ekf.update_imu(state, z_imu)
    """

    def __init__(self,
                 inertia: np.ndarray = INERTIA,
                 damping: np.ndarray = DAMPING,
                 Q_diag: np.ndarray = Q_DIAG,
                 R_enc_diag: np.ndarray = R_ENC_DIAG,
                 R_imu_diag: np.ndarray = R_IMU_DIAG):
        self.I = inertia
        self.b = damping
        self.Q = np.diag(Q_diag)
        self.R_enc = np.diag(R_enc_diag)
        self.R_imu = np.diag(R_imu_diag)

        # Encoder measurement matrix  H_enc @ x = [q1, q2]
        self.H_enc = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ], dtype=float)

        # IMU measurement matrix  H_imu @ x = [q1_dot, q2_dot]
        self.H_imu = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=float)

    # ── Public API ─────────────────────────────────────────────────────────────

    def init(self,
             q0: list[float] = (0.0, 0.0),
             q_dot0: list[float] = (0.0, 0.0),
             t0: float = 0.0) -> EKFState:
        """Return a freshly initialised EKF state."""
        x0 = np.array([q0[0], q_dot0[0], q0[1], q_dot0[1]], dtype=float)
        P0 = np.diag([1e-4, 1e-2, 1e-4, 1e-2])   # small initial uncertainty
        return EKFState(x=x0, P=P0, t=t0)

    def predict(self, state: EKFState, torque: np.ndarray, dt: float) -> EKFState:
        """
        EKF predict step.

        Parameters
        ----------
        state  : current EKF state
        torque : applied joint torques  [tau1, tau2]  (N·m)
        dt     : time since last predict call (s)
        """
        x = state.x.copy()
        P = state.P.copy()

        # ── Nonlinear process model f(x, u) ───────────────────────────────────
        q1,  q1d, q2, q2d = x
        tau1, tau2 = torque

        # Joint accelerations  (simplified: no cross-coupling inertia)
        q1dd = (tau1 - self.b[0] * q1d) / self.I[0]
        q2dd = (tau2 - self.b[1] * q2d) / self.I[1]

        # Euler integration
        x_pred = np.array([
            q1  + dt * q1d,
            q1d + dt * q1dd,
            q2  + dt * q2d,
            q2d + dt * q2dd,
        ])

        # ── Jacobian F = ∂f/∂x  (analytic) ───────────────────────────────────
        #   ∂q_next/∂q   = 1          ∂q_next/∂qdot  = dt
        #   ∂qdot_next/∂q= 0          ∂qdot_next/∂qdot = 1 - dt*b/I
        a1 = 1.0 - dt * self.b[0] / self.I[0]
        a2 = 1.0 - dt * self.b[1] / self.I[1]

        F = np.array([
            [1,  dt, 0,   0],
            [0,  a1, 0,   0],
            [0,   0, 1,  dt],
            [0,   0, 0,  a2],
        ], dtype=float)

        P_pred = F @ P @ F.T + self.Q * dt   # scale Q by dt → diffusion model

        return EKFState(x=x_pred, P=P_pred, t=state.t + dt)

    def update_encoder(self, state: EKFState, z_enc: np.ndarray) -> EKFState:
        """
        EKF update with encoder measurement  z_enc = [q1_meas, q2_meas].
        Linear measurement model → standard Kalman gain (EKF reduces to KF here).
        """
        return self._linear_update(state, z_enc, self.H_enc, self.R_enc)

    def update_imu(self, state: EKFState, z_imu: np.ndarray) -> EKFState:
        """
        EKF update with IMU (gyro) measurement  z_imu = [q1dot_meas, q2dot_meas].
        Linear measurement model.
        """
        return self._linear_update(state, z_imu, self.H_imu, self.R_imu)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def nees(self, state: EKFState, x_true: np.ndarray) -> float:
        """
        Normalised Estimation Error Squared (NEES).
        Should be χ²(4) distributed ≈ 4.0 on average for a consistent filter.
        Values >> 4 → filter under-confident (Q/R too small).
        Values << 4 → filter over-confident.
        """
        e = x_true - state.x
        P_inv = np.linalg.inv(state.P)
        return float(e @ P_inv @ e)

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _linear_update(state: EKFState,
                       z: np.ndarray,
                       H: np.ndarray,
                       R: np.ndarray) -> EKFState:
        """Standard linear Kalman update step."""
        x, P = state.x.copy(), state.P.copy()

        y  = z - H @ x                         # innovation
        S  = H @ P @ H.T + R                   # innovation covariance
        K  = P @ H.T @ np.linalg.inv(S)        # Kalman gain
        x_upd = x + K @ y
        P_upd = (np.eye(len(x)) - K @ H) @ P   # Joseph form for numerical stability

        return EKFState(x=x_upd, P=P_upd, t=state.t)


# ── Safety thresholds (mirrors PLC register definitions) ──────────────────────
JOINT_LIMIT_RAD   = np.array([2.356, 2.094])   # 135°, 120°
VELOCITY_LIMIT    = np.array([3.0, 4.0])        # rad/s


def check_safety(state: EKFState) -> dict[str, bool]:
    """
    Return a dict of safety flags derived from the EKF state estimate.
    These are written to Modbus holding registers by the bridge.
    """
    q   = np.array([state.x[0], state.x[2]])
    qd  = np.array([state.x[1], state.x[3]])
    return {
        "joint1_limit": bool(abs(q[0])  > JOINT_LIMIT_RAD[0]),
        "joint2_limit": bool(abs(q[1])  > JOINT_LIMIT_RAD[1]),
        "joint1_vel":   bool(abs(qd[0]) > VELOCITY_LIMIT[0]),
        "joint2_vel":   bool(abs(qd[1]) > VELOCITY_LIMIT[1]),
        "any_fault":    bool(
            abs(q[0])  > JOINT_LIMIT_RAD[0] or
            abs(q[1])  > JOINT_LIMIT_RAD[1] or
            abs(qd[0]) > VELOCITY_LIMIT[0]  or
            abs(qd[1]) > VELOCITY_LIMIT[1]
        ),
    }
