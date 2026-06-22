"""
Modbus TCP bridge between the Python EKF pipeline and OpenPLC Runtime.

Register map (all holding registers, base address 0):
┌──────┬──────────────────────────────────┬──────────────┐
│ Addr │ Signal                           │ Scale        │
├──────┼──────────────────────────────────┼──────────────┤
│  0   │ Joint 1 angle       [rad * 1000] │ INT16 / 1000 │
│  1   │ Joint 1 velocity    [rad/s * 100]│ INT16 / 100  │
│  2   │ Joint 2 angle       [rad * 1000] │ INT16 / 1000 │
│  3   │ Joint 2 velocity    [rad/s * 100]│ INT16 / 100  │
│  4   │ Safety flags bitmask             │ UINT16 bits  │
│  5   │ Filter health (NEES * 10)        │ UINT16 / 10  │
│  6   │ Timestamp low  word  [ms]        │ UINT16       │
│  7   │ Timestamp high word  [ms]        │ UINT16       │
└──────┴──────────────────────────────────┴──────────────┘

Safety flags bitmask (register 4):
  bit 0: joint1_limit
  bit 1: joint2_limit
  bit 2: joint1_vel
  bit 3: joint2_vel
  bit 4: any_fault

The OpenPLC program reads these registers over Modbus TCP (unit ID 1)
and implements the interlock logic in Structured Text.
"""

import threading
import time
import struct
import logging
from typing import Optional

from pymodbus.server import StartTcpServer
from pymodbus.datastore import (
    ModbusSlaveContext,
    ModbusServerContext,
    ModbusSequentialDataBlock,
)
from pymodbus.device import ModbusDeviceIdentification

log = logging.getLogger(__name__)

# ── Register addresses ─────────────────────────────────────────────────────────
REG_J1_ANGLE   = 0
REG_J1_VEL     = 1
REG_J2_ANGLE   = 2
REG_J2_VEL     = 3
REG_SAFETY     = 4
REG_NEES       = 5
REG_TS_LOW     = 6
REG_TS_HIGH    = 7
NUM_REGISTERS  = 8

# Safety flag bit positions
BIT_J1_LIMIT = 0
BIT_J2_LIMIT = 1
BIT_J1_VEL   = 2
BIT_J2_VEL   = 3
BIT_ANY_FAULT = 4


def _to_int16(value: float, scale: float) -> int:
    """Convert float to scaled signed 16-bit integer, clamped to range."""
    scaled = int(round(value * scale))
    return max(-32768, min(32767, scaled))


def _to_uint16(value: float, scale: float) -> int:
    """Convert float to scaled unsigned 16-bit integer."""
    scaled = int(round(abs(value) * scale))
    return min(65535, scaled)


def _safety_bitmask(flags: dict[str, bool]) -> int:
    mask = 0
    if flags.get("joint1_limit"): mask |= (1 << BIT_J1_LIMIT)
    if flags.get("joint2_limit"): mask |= (1 << BIT_J2_LIMIT)
    if flags.get("joint1_vel"):   mask |= (1 << BIT_J1_VEL)
    if flags.get("joint2_vel"):   mask |= (1 << BIT_J2_VEL)
    if flags.get("any_fault"):    mask |= (1 << BIT_ANY_FAULT)
    return mask


class ModbusBridge:
    """
    Runs a Modbus TCP server in a background thread.
    Call update() from the main EKF loop to write fresh values.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 502):
        self.host = host
        self.port = port
        self._store = self._build_datastore()
        self._context = ModbusServerContext(slaves=self._store, single=True)
        self._lock = threading.Lock()
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

    # ── Server lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the Modbus TCP server in a background thread."""
        self._running = True
        self._server_thread = threading.Thread(
            target=self._serve, daemon=True, name="modbus-server"
        )
        self._server_thread.start()
        log.info("Modbus TCP server starting on %s:%d", self.host, self.port)

    def stop(self) -> None:
        self._running = False
        log.info("Modbus TCP server stopped")

    def _serve(self) -> None:
        identity = ModbusDeviceIdentification()
        identity.VendorName  = "VirtualPLC"
        identity.ProductCode = "VPEKF"
        identity.ProductName = "Virtual PLC EKF Bridge"
        identity.ModelName   = "2DOF-ARM"
        StartTcpServer(
            context=self._context,
            identity=identity,
            address=(self.host, self.port),
        )

    # ── Register update ────────────────────────────────────────────────────────

    def update(self,
               q: list[float],
               qd: list[float],
               safety_flags: dict[str, bool],
               nees: float = 0.0) -> None:
        """
        Write current EKF estimates to Modbus holding registers.

        Parameters
        ----------
        q            : [q1, q2] estimated joint angles [rad]
        qd           : [q1d, q2d] estimated joint velocities [rad/s]
        safety_flags : output of ekf.check_safety()
        nees         : current NEES value (filter health)
        """
        ts_ms = int(time.monotonic() * 1000) & 0xFFFFFFFF   # 32-bit ms counter
        ts_low  = ts_ms & 0xFFFF
        ts_high = (ts_ms >> 16) & 0xFFFF

        registers = [
            _to_int16(q[0],  1000),          # REG_J1_ANGLE
            _to_int16(qd[0], 100),           # REG_J1_VEL
            _to_int16(q[1],  1000),          # REG_J2_ANGLE
            _to_int16(qd[1], 100),           # REG_J2_VEL
            _safety_bitmask(safety_flags),   # REG_SAFETY
            _to_uint16(nees, 10),            # REG_NEES
            ts_low,                          # REG_TS_LOW
            ts_high,                         # REG_TS_HIGH
        ]

        with self._lock:
            # Function code 3 → holding registers in pymodbus SlaveContext
            self._store.setValues(3, 0, registers)

    def read_registers(self) -> list[int]:
        """Read back current register values (for testing / dashboard)."""
        with self._lock:
            return self._store.getValues(3, 0, NUM_REGISTERS)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_datastore() -> ModbusSlaveContext:
        block = ModbusSequentialDataBlock(0, [0] * NUM_REGISTERS)
        return ModbusSlaveContext(hr=block, zero_mode=True)

    @staticmethod
    def decode_registers(regs: list[int]) -> dict:
        """
        Decode raw register values back to engineering units.
        Useful for the dashboard and for debugging.
        """
        def from_int16(v: int) -> int:
            return v if v < 32768 else v - 65536

        safety_mask = regs[REG_SAFETY]
        ts = (regs[REG_TS_HIGH] << 16) | regs[REG_TS_LOW]

        return {
            "q1_rad":      from_int16(regs[REG_J1_ANGLE]) / 1000.0,
            "q1dot_rads":  from_int16(regs[REG_J1_VEL])   / 100.0,
            "q2_rad":      from_int16(regs[REG_J2_ANGLE]) / 1000.0,
            "q2dot_rads":  from_int16(regs[REG_J2_VEL])   / 100.0,
            "joint1_limit": bool(safety_mask & (1 << BIT_J1_LIMIT)),
            "joint2_limit": bool(safety_mask & (1 << BIT_J2_LIMIT)),
            "joint1_vel":   bool(safety_mask & (1 << BIT_J1_VEL)),
            "joint2_vel":   bool(safety_mask & (1 << BIT_J2_VEL)),
            "any_fault":    bool(safety_mask & (1 << BIT_ANY_FAULT)),
            "nees":        regs[REG_NEES] / 10.0,
            "timestamp_ms": ts,
        }


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run the bridge standalone for testing.
    You can then connect OpenPLC or any Modbus client to localhost:502.
    """
    import sys
    import importlib
    import numpy as np
    sys.path.insert(0, str(__file__.replace("src/modbus_bridge.py", "")))

    from src.ekf import RobotArmEKF, check_safety
    from src.simulator import RobotArmSimulator, sinusoidal_torque

    logging.basicConfig(level=logging.INFO)

    bridge = ModbusBridge(host="127.0.0.1", port=5020)  # non-root port for dev
    bridge.start()

    ekf = RobotArmEKF()
    sim = RobotArmSimulator(dt=0.01, seed=0)
    state = ekf.init()

    print("Bridge running on port 5020. Press Ctrl+C to stop.")
    try:
        for true, sensors in sim.run(duration=60.0, torque_fn=sinusoidal_torque):
            state = ekf.predict(state, true.tau, dt=0.01)
            if not sensors.enc_fault:
                state = ekf.update_encoder(state, sensors.enc)
            if not sensors.gyro_fault:
                state = ekf.update_imu(state, sensors.gyro)

            flags = check_safety(state)
            nees  = ekf.nees(state, np.array([true.q[0], true.qd[0],
                                               true.q[1], true.qd[1]]))
            bridge.update(
                q=[state.x[0], state.x[2]],
                qd=[state.x[1], state.x[3]],
                safety_flags=flags,
                nees=nees,
            )
            time.sleep(0.01)
    except KeyboardInterrupt:
        bridge.stop()
