(* ============================================================
   Virtual PLC — Robot Arm Safety Interlock
   IEC 61131-3 Structured Text
   Target runtime: OpenPLC Runtime (IEC 61131-3 compliant)

   This program reads joint state estimates from Modbus TCP
   holding registers (written by the Python EKF bridge) and
   implements the safety interlock logic for a 2-DOF robot arm.

   Register map mirrors src/modbus_bridge.py:
     %IW0  = Joint 1 angle       [rad * 1000, INT16]
     %IW1  = Joint 1 velocity    [rad/s * 100, INT16]
     %IW2  = Joint 2 angle       [rad * 1000, INT16]
     %IW3  = Joint 2 velocity    [rad/s * 100, INT16]
     %IW4  = Safety flags bitmask [UINT16]
     %IW5  = Filter NEES * 10    [UINT16]
     %IW6  = Timestamp low word
     %IW7  = Timestamp high word

   Output coils:
     %QX0.0 = DRIVE_ENABLE      (FALSE = e-stop asserted)
     %QX0.1 = ALARM_JOINT_LIMIT
     %QX0.2 = ALARM_VEL_LIMIT
     %QX0.3 = ALARM_FILTER_DIVERGE
     %QX0.4 = ALARM_WATCHDOG

   Hardware E-STOP input:
     %IX0.0 = ESTOP_HW          (FALSE = button pressed)
   ============================================================ *)

PROGRAM robot_safety

VAR
    (* ── Raw Modbus inputs ─────────────────────────────────── *)
    raw_j1_angle    AT %IW0 : INT;
    raw_j1_vel      AT %IW1 : INT;
    raw_j2_angle    AT %IW2 : INT;
    raw_j2_vel      AT %IW3 : INT;
    safety_flags    AT %IW4 : WORD;
    raw_nees        AT %IW5 : WORD;
    ts_low          AT %IW6 : WORD;
    ts_high         AT %IW7 : WORD;

    (* ── Hardware inputs ───────────────────────────────────── *)
    ESTOP_HW        AT %IX0.0 : BOOL;   (* FALSE = e-stop pressed *)

    (* ── Outputs ───────────────────────────────────────────── *)
    DRIVE_ENABLE       AT %QX0.0 : BOOL;
    ALARM_JOINT_LIMIT  AT %QX0.1 : BOOL;
    ALARM_VEL_LIMIT    AT %QX0.2 : BOOL;
    ALARM_FILTER_DIV   AT %QX0.3 : BOOL;
    ALARM_WATCHDOG     AT %QX0.4 : BOOL;

    (* ── Engineering unit conversions ──────────────────────── *)
    j1_angle_mrad   : REAL;   (* milliradians, before scaling *)
    j1_angle_rad    : REAL;
    j1_vel_crad     : REAL;   (* centi-rad/s, before scaling *)
    j1_vel_rads     : REAL;
    j2_angle_rad    : REAL;
    j2_vel_rads     : REAL;
    nees_val        : REAL;

    (* ── Safety flag bits (extracted from bitmask) ─────────── *)
    bit_j1_limit    : BOOL;
    bit_j2_limit    : BOOL;
    bit_j1_vel      : BOOL;
    bit_j2_vel      : BOOL;
    bit_any_fault   : BOOL;

    (* ── Watchdog: detect stale data from EKF bridge ───────── *)
    ts_prev_low     : WORD;
    ts_prev_high    : WORD;
    ts_stale_count  : INT;
    WATCHDOG_MAX    : INT := 50;   (* 50 * 20ms scan = 1s timeout *)
    watchdog_ok     : BOOL;

    (* ── Latching alarm registers ──────────────────────────── *)
    alarm_latched   : BOOL;
    ALARM_RESET     AT %IX0.1 : BOOL;   (* momentary reset input *)

    (* ── Filter divergence threshold ───────────────────────── *)
    NEES_LIMIT      : REAL := 50.0;   (* chi-sq(4) 99.9% ≈ 18.5; 50 is very loose *)
END_VAR

(* ============================================================
   SECTION 1: Scale raw register values to engineering units
   ============================================================ *)

(* Signed 16-bit to REAL, then divide by scale factor *)
j1_angle_rad  := INT_TO_REAL(raw_j1_angle) / 1000.0;
j1_vel_rads   := INT_TO_REAL(raw_j1_vel)   / 100.0;
j2_angle_rad  := INT_TO_REAL(raw_j2_angle) / 1000.0;
j2_vel_rads   := INT_TO_REAL(raw_j2_vel)   / 100.0;
nees_val      := WORD_TO_REAL(raw_nees)    / 10.0;

(* ============================================================
   SECTION 2: Extract safety flag bits from bitmask
   ============================================================ *)

bit_j1_limit  := (safety_flags AND 16#0001) <> 0;
bit_j2_limit  := (safety_flags AND 16#0002) <> 0;
bit_j1_vel    := (safety_flags AND 16#0004) <> 0;
bit_j2_vel    := (safety_flags AND 16#0008) <> 0;
bit_any_fault := (safety_flags AND 16#0010) <> 0;

(* ============================================================
   SECTION 3: Watchdog — detect frozen timestamp from EKF bridge
   If ts_low and ts_high do not change for WATCHDOG_MAX scans,
   the Python bridge has stalled → assert fault.
   ============================================================ *)

IF (ts_low = ts_prev_low) AND (ts_high = ts_prev_high) THEN
    ts_stale_count := ts_stale_count + 1;
ELSE
    ts_stale_count := 0;
    ts_prev_low    := ts_low;
    ts_prev_high   := ts_high;
END_IF;

watchdog_ok := (ts_stale_count < WATCHDOG_MAX);
ALARM_WATCHDOG := NOT watchdog_ok;

(* ============================================================
   SECTION 4: Individual alarm conditions
   ============================================================ *)

ALARM_JOINT_LIMIT := bit_j1_limit OR bit_j2_limit;
ALARM_VEL_LIMIT   := bit_j1_vel   OR bit_j2_vel;
ALARM_FILTER_DIV  := (nees_val > NEES_LIMIT) AND (nees_val > 0.0);

(* ============================================================
   SECTION 5: Latching alarm logic
   An alarm once set stays set until ALARM_RESET is pulsed AND
   all current fault conditions are cleared (safe state).
   This mirrors industrial safety relay behaviour.
   ============================================================ *)

IF ALARM_JOINT_LIMIT OR ALARM_VEL_LIMIT OR ALARM_FILTER_DIV OR ALARM_WATCHDOG THEN
    alarm_latched := TRUE;
END_IF;

IF ALARM_RESET AND
   NOT ALARM_JOINT_LIMIT AND
   NOT ALARM_VEL_LIMIT   AND
   NOT ALARM_FILTER_DIV  AND
   NOT ALARM_WATCHDOG     THEN
    alarm_latched := FALSE;
END_IF;

(* ============================================================
   SECTION 6: Drive enable — de-energise to stop
   DRIVE_ENABLE is TRUE only when:
     • No alarm latched
     • Hardware e-stop not pressed (ESTOP_HW = TRUE)
     • Watchdog healthy
   This is a "normally de-energised" safety output — any power
   loss or PLC fault automatically removes drive enable.
   ============================================================ *)

DRIVE_ENABLE := ESTOP_HW AND (NOT alarm_latched) AND watchdog_ok;

END_PROGRAM


(* ============================================================
   CONFIGURATION — wires program to a 20ms cyclic task
   ============================================================ *)

CONFIGURATION default

    RESOURCE res ON PLC
        TASK cyclic_task(INTERVAL := T#20ms, PRIORITY := 1);
        PROGRAM inst WITH cyclic_task : robot_safety;
    END_RESOURCE

END_CONFIGURATION
