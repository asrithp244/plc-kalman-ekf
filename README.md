# Virtual PLC - EKF Joint State Estimator with IEC 61131-3 Safety Interlocks

> **A fully software-simulated industrial robot work cell that bridges the IT/OT gap:**
> a Python Extended Kalman Filter estimates joint states from noisy sensors and
> writes them to a real soft PLC over Modbus TCP, which runs actual IEC 61131-3
> Structured Text safety interlock logic.

---

## Why This Project Exists

Every robotics team eventually hits the same wall: the software engineers can write
ROS nodes and the controls engineers can write PLC ladder logic, but almost nobody
can do both in the same repo. This project is a deliberate attempt to own both sides
of that wall.

The engineering problem is real: industrial robot arms use encoders and IMUs that
are noisy, quantised, and occasionally drop out. A raw encoder reading fed into a
safety interlock will trigger false alarms constantly. The correct fix is sensor
fusion - but in a real plant, that fusion runs in a PC-class machine that must
somehow talk to a DIN-rail PLC running at 20ms scan cycles. Modbus TCP is the
standard bridge, and IEC 61131-3 Structured Text is the language the PLC speaks.

This project implements the whole stack, entirely in software, on a single laptop.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Process                          │
│                                                             │
│   ┌──────────────┐    ┌───────────┐    ┌────────────────┐  │
│   │  Simulator   │───▶│    EKF    │───▶│  Modbus Bridge │  │
│   │ (RK4 + noise)│    │(4-state)  │    │  (pymodbus)    │  │
│   └──────────────┘    └───────────┘    └───────┬────────┘  │
│                                                 │           │
└─────────────────────────────────────────────────┼───────────┘
                                                  │ Modbus TCP
                               ┌──────────────────▼──────────┐
                               │   OpenPLC Runtime (Docker)  │
                               │   IEC 61131-3 Structured    │
                               │   Text - robot_safety.st    │
                               │                             │
                               │   - Joint limit interlock   │
                               │   - Velocity limit interlock│
                               │   - Filter divergence alarm │
                               │   - Watchdog (stale data)   │
                               │   - Latching safety relay   │
                               └─────────────────────────────┘
                                                  │
                               ┌──────────────────▼──────────┐
                               │   Dash Dashboard (port 8050)│
                               │   Real-time plots:          │
                               │   - Raw vs EKF vs Truth     │
                               │   - NEES filter health      │
                               │   - Joint space safety zone │
                               │   - Alarm log               │
                               └─────────────────────────────┘
```

---

## EKF Design

**State vector:** `x = [q1, q1_dot, q2, q2_dot]` - joint angles (rad) and velocities (rad/s)

**Process model** (nonlinear rigid-body dynamics):

```
q_ddot_i = (tau_i - b_i * q_dot_i - g_i * sin(q_i)) / I_i
```

Linearised Jacobian `F = df/dx` is computed analytically each step - no finite
differences. The diffusion process noise `Q*dt` ensures the filter stays bounded
during sensor dropout.

**Measurement models** (both linear, so EKF reduces to standard KF in update):

| Sensor  | Measurement              | Noise std    |
|---------|--------------------------|--------------|
| Encoder | `z = [q1, q2]`           | 0.001 rad    |
| IMU     | `z = [q1_dot, q2_dot]`   | 0.03 rad/s   |

**Filter health:** NEES (Normalised Estimation Error Squared) is monitored. For a
consistent filter, NEES ~ chi-squared(4) = 4.0. Values above 18.5 (99.9% bound) trigger
a `ALARM_FILTER_DIV` in the PLC.

---

## PLC Program (robot_safety.st)

Written in **IEC 61131-3 Structured Text** - the same language used on Siemens S7,
Allen-Bradley CompactLogix, and Beckhoff TwinCAT PLCs.

Key interlock logic:

- **Joint limit:** de-energises drive if EKF angle estimate exceeds 135 deg / 120 deg
- **Velocity limit:** de-energises drive if velocity exceeds 3.0 / 4.0 rad/s
- **Filter health:** alarm if NEES > 50 (filter has diverged, estimates unreliable)
- **Watchdog:** if the Modbus timestamp stops updating for > 1 second, assert fault
- **Latching relay:** once a fault fires, it stays latched until reset is pressed
  AND all conditions are clear - mirrors IEC 62061 SIL-rated relay behaviour

---

## Quickstart

### Option A - Dashboard only (no Docker needed)

```bash
# Install Python deps
pip install -r requirements.txt

# Run the live dashboard (runs its own internal sim loop)
python dashboard/app.py
```

Open **http://localhost:8050** - you'll see the EKF tracking real-time, the filter
diverging after ~15 seconds when the velocity ramp torque kicks in, and safety
alarms firing.

### Option B - Full stack with real OpenPLC

```bash
# Requires Docker Desktop
docker compose -f docker/docker-compose.yml up

# OpenPLC Web UI: http://localhost:8080
#   -> Programs -> Upload -> select plc/robot_safety.st -> Compile -> Start PLC
# Dashboard:     http://localhost:8050
# Modbus bridge: port 5020 (connect any Modbus client to inspect registers)
```

### Run tests

```bash
pytest tests/ -v
```

Expected output: 7 tests pass in < 10 seconds.

---

## Register Map (Modbus Holding Registers, unit ID 1)

| Address | Signal               | Scale      | Type   |
|---------|----------------------|------------|--------|
| 0       | Joint 1 angle        | x1000 rad  | INT16  |
| 1       | Joint 1 velocity     | x100 rad/s | INT16  |
| 2       | Joint 2 angle        | x1000 rad  | INT16  |
| 3       | Joint 2 velocity     | x100 rad/s | INT16  |
| 4       | Safety flags bitmask | bits 0-4   | UINT16 |
| 5       | NEES x10             | /10        | UINT16 |
| 6       | Timestamp low word   | ms         | UINT16 |
| 7       | Timestamp high word  | ms         | UINT16 |

---

## Project Structure

```
plc_kalman/
├── src/
│   ├── ekf.py           # Extended Kalman Filter (numpy only)
│   ├── simulator.py     # 2-DOF arm physics + sensor noise (RK4)
│   └── modbus_bridge.py # Modbus TCP server bridging EKF -> PLC
├── plc/
│   └── robot_safety.st  # IEC 61131-3 Structured Text safety program
├── dashboard/
│   └── app.py           # Plotly Dash live monitoring dashboard
├── tests/
│   └── test_ekf.py      # pytest: convergence, NEES, dropout, stability
├── docker/
│   ├── docker-compose.yml
│   ├── Dockerfile.ekf
│   └── Dockerfile.dashboard
└── requirements.txt
```

---

## Technical Highlights for Reviewers

- **No ML frameworks** - EKF implemented from scratch with only numpy. Analytic
  Jacobian, Joseph-form covariance update for numerical stability.
- **Real industrial protocol** - Modbus TCP with correct INT16 scaling, not a toy
  socket. Any hardware Modbus master (SCADA, HMI, real PLC) can connect.
- **Real PLC language** - IEC 61131-3 Structured Text, not pseudocode. Paste
  `robot_safety.st` into OpenPLC or CODESYS and it compiles.
- **Safety engineering patterns** - latching relay, watchdog timer, de-energise-
  to-trip output - all standard IEC 62061 / IEC 61511 patterns.
- **Quantitative filter evaluation** - NEES chi-squared consistency test, not just
  a pretty plot. The test suite enforces filter health numerically.

---


