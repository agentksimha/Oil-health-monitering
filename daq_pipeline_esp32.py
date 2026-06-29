"""
Oil Health DAQ Pipeline — ESP-32 Edition
=========================================
Aligned with ohi-final.ipynb (v4, June 2026).

Feature vector (6): [temperature, soot_pct, capacitance_pF, tbn, oxidation_index, dilution]

Sensor → raw packet flow on real ESP-32:
  temperature    : °C              (DS18B20)
  raw_adc        : 0–4095          (turbidity / soot photodiode)
  dilution_adc   : 0–4095          (MQ-3 alcohol sensor, fuel-dilution path)
  capacitance_pF : pF              (touch-capacitive dielectric probe)

Processing chain per tick:
  1. raw_adc       → Voltage = (4095−ADC)/4095×100  → Soot % (poly-3 calibration)
  2. temperature history → TBN % remaining  (second-order Arrhenius, Eₐ=60 kJ/mol)
  3. temperature history → Oxidation Index  (first-order Arrhenius accumulation, Eb=55 kJ/mol)
  4. dilution_adc   → Fuel Dilution %  (−21.4218 + 0.022253 × ADC, linear)
  5. [temp, soot, cap_pF, tbn, OI, dilution] → KMeans (k=3) → HEALTHY/WARNING/DEGRADED
  6. PCA PC1 loadings → data-driven OHI weights → OHI 0–100
  7. Store to SQLite (WAL mode)

Serial protocol from ESP-32 firmware (oil_health_monitor.ino):
  The firmware emits two lines per tick:
    [DBG]  human-readable debug line  (ignored by this script)
    [CSV]  engine_hours,temperature,raw_adc,dilution_adc,capacitance_pF

  Only lines starting with "[CSV] " are parsed.

Usage:
  # Live ESP-32 over USB serial (auto-detect port, or specify):
  python daq_pipeline_esp32.py --source serial
  python daq_pipeline_esp32.py --source serial --port COM4          # Windows
  python daq_pipeline_esp32.py --source serial --port /dev/ttyUSB0  # Linux
  python daq_pipeline_esp32.py --source serial --port /dev/tty.usbserial-0001  # Mac

  # Simulation mode (no hardware needed):
  python daq_pipeline_esp32.py --source sim

  # Stop a running instance gracefully from another terminal:
  python daq_pipeline_esp32.py --stop

  # Offline retrain after a live session:
  python daq_pipeline_esp32.py --retrain

ON/OFF control:
  • Ctrl+C   — graceful stop at any time
  • --stop   — write sentinel file; running process detects and shuts down
  • SIGTERM  — also triggers graceful stop
"""

import argparse
import json
import math
import os
import pathlib
import platform
import signal
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ── Paths (laptop-friendly — everything lives next to this script) ────────────
BASE_DIR      = pathlib.Path(__file__).parent / "oil_health_data"
DATA_DIR      = BASE_DIR / "data"
DB_PATH       = str(DATA_DIR / "oil_health.db")
STOP_SENTINEL = str(BASE_DIR / ".stop")
# TBN_MODEL_PATH / OI_CAP_MODEL_PATH removed — both models now use
# Arrhenius kinetics driven by temperature history; no .npz files needed.

# ── Serial CSV prefix (must match firmware printCSV()) ────────────────────────
CSV_PREFIX = "[CSV] "

# ── Soot calibration data (from Soot_Calibration.csv, Option B) ──────────────
# Voltage = (4095 − Raw_ADC) / 4095 × 100
SOOT_CAL_DATA = [
    # (Raw_ADC, Voltage, Soot_pct)
    (3100, 24.50, 0.0350),
    (2150, 47.20, 0.0470),
    (1220, 70.00, 0.0706),
    (1058, 74.14, 0.0676),
    (1170, 70.40, 0.0667),
    (1450, 64.49, 0.0570),
    (2633, 35.53, 0.0400),
    ( 901, 78.00, 0.0800),
    ( 400, 90.23, 0.1140),
    ( 431, 89.69, 0.1080),
    ( 602, 85.30, 0.0930),
    ( 336, 91.55, 0.1300),
    ( 156, 96.19, 0.1680),
    ( 133, 96.75, 0.2180),
]

# ── Feature definitions ───────────────────────────────────────────────────────
FEATURES = ["temperature", "soot_pct", "capacitance_pF", "tbn", "oxidation_index", "dilution"]

THRESHOLDS = {
    "temperature":     {"direction": "below", "max": 110.0},
    "soot_pct":        {"direction": "below", "max":   0.10},
    "capacitance_pF":  {"direction": "above", "min":  50.0},
    "tbn":             {"direction": "above", "min":  40.0},
    "oxidation_index": {"direction": "below", "max":  30.0},
    "dilution":        {"direction": "below", "max":   5.0},
}

# ── Pipeline config ───────────────────────────────────────────────────────────
STREAM_INTERVAL  = 1.0    # seconds between ticks
RING_BUFFER_SIZE = 500    # rolling window for offline training
MIN_TRAIN        = 30     # minimum samples before first KMeans fit
RETRAIN_EVERY    = 30     # retrain every N ticks after MIN_TRAIN is reached

# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class ESP32Packet:
    """Raw sensor packet from the ESP-32 node (real or simulated)."""
    timestamp:      float
    engine_hours:   float
    temperature:    float   # °C
    raw_adc:        int     # 0–4095, turbidity sensor (soot path)
    dilution_adc:   int     # 0–4095, MQ-3 (fuel-dilution path)
    capacitance_pF: float   # pF

@dataclass
class ProcessedReading:
    """Fully processed reading stored to SQLite per tick."""
    timestamp:       float
    engine_hours:    float
    temperature:     float
    raw_adc:         int
    dilution_adc:    int
    soot_pct:        float
    capacitance_pF:  float
    analexrs:        float
    tbn:             float
    oxidation_index: float
    dilution:        float
    cluster:         int
    cluster_label:   str
    ohi:             int
    pca_weights:     str    # JSON string

# ── Model: Soot calibration ───────────────────────────────────────────────────
class SootCalibrationModel:
    """Raw ADC → Voltage → Soot % via degree-3 polynomial fit."""
    ADC_MAX  = 4095
    V_MIN    =  24.5
    V_MAX    =  96.75
    SOOT_MIN =   0.0
    SOOT_MAX =   0.25

    def __init__(self, cal_data: list = None):
        data      = cal_data or SOOT_CAL_DATA
        voltages  = np.array([r[1] for r in data])
        soot_pcts = np.array([r[2] for r in data])
        self.coeffs = np.polyfit(voltages, soot_pcts, 3)
        y_hat = np.polyval(self.coeffs, voltages)
        ss_res = np.sum((soot_pcts - y_hat) ** 2)
        ss_tot = np.sum((soot_pcts - soot_pcts.mean()) ** 2)
        r2   = 1 - ss_res / ss_tot
        rmse = float(np.sqrt(np.mean((soot_pcts - y_hat) ** 2)))
        print(f"[SOOT]  Poly-3 fit  R²={r2:.4f}  RMSE={rmse:.5f}")

    def adc_to_voltage(self, raw_adc: int) -> float:
        return (self.ADC_MAX - raw_adc) / self.ADC_MAX * 100.0

    def predict(self, raw_adc: int) -> float:
        v    = max(self.V_MIN, min(self.V_MAX, self.adc_to_voltage(raw_adc)))
        soot = float(np.polyval(self.coeffs, v))
        return round(max(self.SOOT_MIN, min(self.SOOT_MAX, soot)), 6)

# ── Model: TBN — second-order Arrhenius kinetics ─────────────────────────────
class TBNModel:
    """
    Predicts TBN % remaining using second-order Arrhenius thermal kinetics.

    Physical basis (Honda patent US7826987 / US8464576):
      d[TBN]/dt = −k₁(T) · TBN²
      k₁(T) = A · exp(−Eₐ / R·T)   [T in Kelvin]

    Integrated discretely per tick (Euler step, dt in hours):
      TBN[t+1] = TBN[t] / (1 + TBN[t] · k₁(T[t]) · dt)

    Parameters:
      Eₐ = 60,000 J/mol  A = 1.081e4 h⁻¹  R = 8.314 J/mol·K
    """

    EA      = 60_000.0   # J/mol — activation energy
    A       = 1.081e4    # h⁻¹  — pre-exponential factor
    R       = 8.314      # J/mol·K
    TBN_MIN = 20.0       # % — lower bound (fully depleted)
    TBN_MAX = 100.0      # % — upper bound (fresh oil)

    def __init__(self, tbn_init: float = 92.78):
        self.tbn_init = tbn_init
        self._tbn = tbn_init
        print(f"[TBN]   Second-order Arrhenius  Eₐ={self.EA/1000:.0f} kJ/mol  A={self.A}")

    def reset(self, tbn_init: float = None):
        """Reset state to fresh oil. Call at start of each session."""
        if tbn_init is not None:
            self.tbn_init = tbn_init
        self._tbn = self.tbn_init

    def k1(self, temperature_C: float) -> float:
        """Arrhenius rate constant k₁ at given temperature (°C)."""
        T_K = temperature_C + 273.15
        return self.A * math.exp(-self.EA / (self.R * T_K))

    def step(self, temperature_C: float, dt_hours: float = 0.4) -> float:
        """
        Advance TBN by one tick of duration dt_hours at temperature_C.
        Returns current TBN % (clamped to [TBN_MIN, TBN_MAX]).
        """
        k = self.k1(temperature_C)
        self._tbn = self._tbn / (1.0 + self._tbn * k * dt_hours)
        self._tbn = max(self.TBN_MIN, min(self.TBN_MAX, self._tbn))
        return round(self._tbn, 4)

    @property
    def current(self) -> float:
        """Current TBN % without advancing the state."""
        return round(self._tbn, 4)

# ── Model: Oxidation Index — first-order Arrhenius accumulation ───────────────
class OxidationIndexModel:
    """
    Predicts Oxidation Index (OI) using first-order Arrhenius accumulation kinetics.

    Physical basis (US patent 6920779; ASTM D974):
      d[OI]/dt = k₂(T) · (OI_max − OI)
      k₂(T) = B · exp(−Eb / R·T)   [T in Kelvin]

    Integrated discretely per tick (Euler step, dt in hours):
      OI[t+1] = OI[t] + k₂(T[t]) · (OI_max − OI[t]) · dt

    Parameters:
      Eb = 55,000 J/mol  B = 5.843e4 h⁻¹  OI_max = 100.0  OI_init = 16.6
    """

    EB      = 55_000.0   # J/mol — activation energy for oxidation
    B       = 5.843e4    # h⁻¹  — pre-exponential factor
    R       = 8.314      # J/mol·K
    OI_MAX  = 100.0      # saturation (fully oxidised)
    OI_INIT =  16.6      # fresh oil baseline (Gomółka & Augustynowicz 2019)

    def __init__(self):
        self._oi = self.OI_INIT
        print(f"[OI]    First-order Arrhenius accumulation  Eb={self.EB/1000:.0f} kJ/mol  B={self.B}")

    def reset(self):
        """Reset to fresh oil OI. Call at start of each session."""
        self._oi = self.OI_INIT

    def k2(self, temperature_C: float) -> float:
        """Arrhenius rate constant k₂ at given temperature (°C)."""
        T_K = temperature_C + 273.15
        return self.B * math.exp(-self.EB / (self.R * T_K))

    def step(self, temperature_C: float, dt_hours: float = 0.4) -> float:
        """
        Advance OI by one tick of duration dt_hours at temperature_C.
        Returns current OI (clamped to [OI_INIT, OI_MAX]).
        """
        k = self.k2(temperature_C)
        self._oi = self._oi + k * (self.OI_MAX - self._oi) * dt_hours
        self._oi = max(self.OI_INIT, min(self.OI_MAX, self._oi))
        return round(self._oi, 4)

    @property
    def current(self) -> float:
        """Current OI without advancing the state."""
        return round(self._oi, 4)

# ── Model: Fuel Dilution ──────────────────────────────────────────────────────
class DilutionModel:
    """Dilution % = −21.4218 + 0.022253 × dilution_adc  (linear, R²=0.78)."""
    INTERCEPT = -21.4218
    SLOPE     =  0.022253
    ADC_MIN   = 1130
    ADC_MAX   = 1328

    def predict(self, dilution_adc: int) -> float:
        adc = max(self.ADC_MIN, min(self.ADC_MAX, dilution_adc))
        return round(self.INTERCEPT + self.SLOPE * adc, 4)

# ── OHI Engine: KMeans + PCA ──────────────────────────────────────────────────
class OHIEngine:
    """KMeans (k=3, k-means++) + PCA PC1 loadings → Oil Health Index 0–100."""
    FEAT_MIN = np.array([ 40.0,  0.000,  20.0,  20.0,   0.0,  0.0], dtype=np.float32)
    FEAT_MAX = np.array([150.0,  0.250, 120.0, 100.0, 100.0, 15.0], dtype=np.float32)

    def __init__(self):
        self.scaler  = MinMaxScaler()
        self.kmeans  = None
        self.pca     = PCA(n_components=1)
        self.weights = {f: 1.0 / len(FEATURES) for f in FEATURES}
        self.labels  = []
        self.trained = False
        self._lock   = threading.Lock()

    def train(self, X: np.ndarray, verbose: bool = True):
        if X.shape[0] < 9:
            return
        with self._lock:
            X_norm = self.scaler.fit_transform(X)
            self.kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10,
                                 max_iter=300, random_state=42)
            self.kmeans.fit(X_norm)

            # Label clusters from physical thresholds
            centroids_raw = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
            vscores = []
            for c in centroids_raw:
                v = 0
                for i, feat in enumerate(FEATURES):
                    t, val = THRESHOLDS.get(feat, {}), c[i]
                    if t.get("direction") == "above" and val < t.get("min", 0):
                        v += 2
                    elif t.get("direction") == "below" and val > t.get("max", 9999):
                        v += 1
                vscores.append(v)
            order = np.argsort(vscores)
            lmap  = [""] * 3
            for rank, ci in enumerate(order):
                lmap[ci] = ["HEALTHY", "WARNING", "DEGRADED"][min(rank, 2)]
            self.labels = lmap

            self.pca.fit(X_norm)
            loadings = np.abs(self.pca.components_[0])
            w = loadings / (loadings.sum() + 1e-9)
            self.weights = {feat: round(float(w[i]), 4) for i, feat in enumerate(FEATURES)}
            self.trained = True

        if verbose:
            print(f"  [OHI] retrained  inertia={self.kmeans.inertia_:.3f}  "
                  f"PC1_var={self.pca.explained_variance_ratio_[0]:.3f}  "
                  f"labels={self.labels}")
            print(f"  [OHI] weights={self.weights}")

    def predict(self, feat_vec: np.ndarray):
        """Returns (cluster_int, label_str, ohi_int)."""
        with self._lock:
            if not self.trained or self.kmeans is None:
                return 0, "HEALTHY", 50

            X_norm = np.clip(
                (feat_vec.reshape(1, -1) - self.scaler.data_min_) /
                (self.scaler.data_max_ - self.scaler.data_min_ + 1e-8),
                0.0, 1.0
            )
            cluster = int(self.kmeans.predict(X_norm)[0])
            label   = self.labels[cluster] if self.labels else "UNKNOWN"

            health = np.zeros(len(FEATURES), dtype=np.float32)
            for i, feat in enumerate(FEATURES):
                t    = THRESHOLDS.get(feat, {})
                norm = float(np.clip(
                    (feat_vec[i] - self.FEAT_MIN[i]) /
                    (self.FEAT_MAX[i] - self.FEAT_MIN[i] + 1e-8), 0, 1))
                health[i] = norm if t.get("direction") == "above" else 1.0 - norm

            ohi = sum(self.weights.get(f, 1 / len(FEATURES)) * float(health[i])
                      for i, f in enumerate(FEATURES))
            return cluster, label, max(0, min(100, int(ohi * 100)))

# ── SQLite store ──────────────────────────────────────────────────────────────
class OilHealthDB:
    """WAL-mode SQLite store. Schema matches ohi-final.ipynb exactly."""

    def __init__(self, path: str = DB_PATH):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-4096")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id               INTEGER PRIMARY KEY,
                timestamp        REAL    NOT NULL,
                engine_hours     REAL    NOT NULL,
                temperature      REAL    NOT NULL,
                raw_adc          INTEGER NOT NULL,
                dilution_adc     INTEGER NOT NULL,
                soot_pct         REAL    NOT NULL,
                capacitance_pF   REAL    NOT NULL,
                analexrs         REAL    NOT NULL,
                tbn              REAL    NOT NULL,
                oxidation_index  REAL    NOT NULL,
                dilution         REAL    NOT NULL,
                cluster          INTEGER NOT NULL,
                cluster_label    TEXT    NOT NULL,
                ohi              INTEGER NOT NULL,
                pca_weights      TEXT    NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_readings_hours ON readings (engine_hours)"
        )
        self._conn.commit()
        print(f"[DB]    SQLite ready  →  {path}")

    def insert(self, r: ProcessedReading):
        with self._lock:
            self._conn.execute(
                """INSERT INTO readings
                   (timestamp, engine_hours, temperature, raw_adc, dilution_adc, soot_pct,
                    capacitance_pF, analexrs, tbn, oxidation_index, dilution,
                    cluster, cluster_label, ohi, pca_weights)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r.timestamp, r.engine_hours, r.temperature, r.raw_adc, r.dilution_adc,
                 r.soot_pct, r.capacitance_pF, r.analexrs, r.tbn, r.oxidation_index,
                 r.dilution, r.cluster, r.cluster_label, r.ohi, r.pca_weights),
            )
            self._conn.commit()

    def load_for_training(self) -> np.ndarray:
        cols = ", ".join(FEATURES)
        with self._lock:
            cur  = self._conn.execute(f"SELECT {cols} FROM readings ORDER BY id")
            rows = cur.fetchall()
        return np.array(rows, dtype=np.float32) if rows else np.empty((0, len(FEATURES)), dtype=np.float32)

    def close(self):
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()
        print("[DB]    Connection closed.")

# ── Serial port auto-detection ────────────────────────────────────────────────
def auto_detect_port() -> Optional[str]:
    """
    Attempt to find the ESP32 USB-serial port automatically.
    Returns the first likely candidate, or None if nothing found.
    """
    import glob
    system = platform.system()
    candidates = []
    if system == "Windows":
        # Try COM3–COM20 (pyserial can open by name)
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = (p.description or "").lower()
            if any(k in desc for k in ("cp210", "ch340", "ch341", "ftdi", "uart", "esp")):
                candidates.append(p.device)
        if not candidates:
            candidates = [p.device for p in ports]
    elif system == "Darwin":  # macOS
        candidates = (glob.glob("/dev/tty.usbserial-*") +
                      glob.glob("/dev/tty.SLAB_USBtoUART*") +
                      glob.glob("/dev/tty.wchusbserial*"))
    else:  # Linux
        candidates = (glob.glob("/dev/ttyUSB*") +
                      glob.glob("/dev/ttyACM*"))

    for c in candidates:
        print(f"[SERIAL] Auto-detected port candidate: {c}")
        return c
    return None

# ── Sensor stream: real serial ────────────────────────────────────────────────
def make_serial_stream(port: Optional[str] = None, baud: int = 115200,
                       engine_hours_start: float = 0.0) -> Callable:
    """
    Read one CSV packet per tick from the ESP-32 over USB-UART.

    The firmware emits:
      [DBG]  ... human readable ...
      [CSV]  engine_hours,temperature,raw_adc,dilution_adc,capacitance_pF

    Only lines that start with "[CSV] " are parsed; all others are printed
    to the console as pass-through debug output.
    """
    import serial

    if port is None:
        port = auto_detect_port()
        if port is None:
            raise RuntimeError(
                "Could not auto-detect ESP32 serial port.\n"
                "Pass --port explicitly, e.g.:\n"
                "  Windows : --port COM4\n"
                "  Linux   : --port /dev/ttyUSB0\n"
                "  Mac     : --port /dev/tty.usbserial-0001"
            )

    ser = serial.Serial(port, baud, timeout=2)
    time.sleep(2)  # allow ESP32 to reset after DTR toggle
    ser.reset_input_buffer()
    print(f"[SERIAL] Opened {port} @ {baud} baud")
    print(f"[SERIAL] Waiting for [CSV] lines from firmware…")

    def read() -> Optional[ESP32Packet]:
        while True:
            raw = ser.readline()
            if not raw:
                return None  # timeout — no data
            try:
                line = raw.decode(errors="replace").strip()
            except Exception:
                continue

            if not line:
                continue

            # Pass [DBG] lines through to the console so developers can see them
            if line.startswith("[DBG]"):
                print(f"  {line}")
                continue

            # Parse [CSV] lines
            if line.startswith(CSV_PREFIX):
                payload = line[len(CSV_PREFIX):]
                try:
                    parts = [p.strip() for p in payload.split(",")]
                    if len(parts) < 5:
                        print(f"[SERIAL] Short CSV ({len(parts)} fields): {line!r}")
                        continue
                    return ESP32Packet(
                        timestamp      = time.time(),
                        engine_hours   = float(parts[0]) + engine_hours_start,
                        temperature    = float(parts[1]),
                        raw_adc        = int(float(parts[2])),
                        dilution_adc   = int(float(parts[3])),
                        capacitance_pF = float(parts[4]),
                    )
                except (ValueError, IndexError) as e:
                    print(f"[SERIAL] Bad CSV ({e}): {line!r}")
                    continue

            # Any other line — ignore silently (e.g. boot messages)

    return read

# ── Sensor stream: simulation ─────────────────────────────────────────────────
def make_simulation_stream(seed: int = 42, hours_per_step: float = 0.4) -> Callable:
    """
    Mirrors generate_esp32_packets() from ohi-final.ipynb exactly.
    Degradation over 250 simulated engine hours:
      temperature    : 70°C  → 130°C
      raw_adc        : 3100  → 133       (soot build-up)
      capacitance_pF : 80 pF → 25 pF
      dilution_adc   : 1130  → 1328
    """
    rng  = np.random.default_rng(seed)
    step = [0]

    def read() -> Optional[ESP32Packet]:
        t        = step[0]
        step[0] += 1
        eng_hours = t * hours_per_step
        progress  = min(eng_hours / 250.0, 1.0)

        temperature    = 70.0   + 60.0  * progress**1.5  + rng.normal(0, 1.5)
        raw_adc_f      = 3100.0 - 2967.0 * progress**1.2 + rng.normal(0, 30)
        capacitance_pF = 80.0   - 55.0  * progress**1.2  + rng.normal(0, 1.0)
        dilution_adc_f = 1130.0 + 198.0 * progress**1.2  + rng.normal(0, 10)

        return ESP32Packet(
            timestamp      = time.time(),
            engine_hours   = round(eng_hours, 3),
            temperature    = round(max(50.0, temperature), 2),
            raw_adc        = int(max(133, min(3100, raw_adc_f))),
            dilution_adc   = int(max(1130, min(1328, dilution_adc_f))),
            capacitance_pF = round(max(5.0, min(120.0, capacitance_pF)), 2),
        )

    return read

# ── Main DAQ pipeline ─────────────────────────────────────────────────────────
class OilHealthDAQ:
    """
    Data acquisition and inference loop for the ESP-32 oil health monitor.

    Hot-path per tick (target ≤ 5 ms on any modern laptop):
      1. Read ESP32Packet from stream
      2. Soot model    : raw_adc → soot_pct          (~1 µs)
      3. TBN model     : cap_pF  → tbn               (~5 µs)
      4. OI model      : cap_pF  → oxidation_index   (~1 µs)
      5. Dilution      : dilution_adc → dilution      (~1 µs)
      6. OHI predict   : KMeans + weighted score      (~10 µs)
      7. DB insert     : SQLite WAL write             (~0.05 ms)
    """

    def __init__(self, stream_fn: Callable, on_result: Callable = None,
                 dt_hours: float = 0.4):
        self.stream_fn = stream_fn
        self.on_result = on_result or self._default_print
        self.dt_hours  = dt_hours   # engine-hours per tick (matches ESP-32 firmware)

        self.soot_model     = SootCalibrationModel()
        self.tbn_model      = TBNModel(tbn_init=92.78)   # second-order Arrhenius
        self.oi_model       = OxidationIndexModel()       # first-order Arrhenius
        self.dilution_model = DilutionModel()
        self.ohi_engine     = OHIEngine()
        self.db             = OilHealthDB()

        # Reset stateful thermal models to fresh-oil initial conditions
        self.tbn_model.reset()
        self.oi_model.reset()

        self._ring: List[np.ndarray] = []
        self._tick_count = 0
        self._running    = False
        self._lock       = threading.Lock()

        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\n[DAQ]   Signal {signum} received — stopping pipeline…")
        self.stop()

    def start(self, interval: float = STREAM_INTERVAL):
        """Block and run the acquisition loop until stop() is called."""
        self._running = True
        print(f"[DAQ]   Pipeline started  (interval={interval}s)")
        print("[DAQ]   Press Ctrl+C  or  python daq_pipeline_esp32.py --stop  to stop.\n")

        while self._running:
            if os.path.exists(STOP_SENTINEL):
                os.remove(STOP_SENTINEL)
                print("[DAQ]   Stop sentinel detected.")
                break

            t0 = time.monotonic()
            try:
                self._tick()
            except Exception as e:
                print(f"[ERROR] Tick exception: {e}")
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))

        self._shutdown()

    def stop(self):
        """Signal the loop to exit on the next iteration. Thread-safe."""
        self._running = False

    def _shutdown(self):
        self.db.close()
        print("[DAQ]   Shutdown complete.")
        print("[DAQ]   Run  python daq_pipeline_esp32.py --retrain  to retrain on collected data.")

    def _tick(self):
        pkt = self.stream_fn()
        if pkt is None:
            return

        soot_pct = self.soot_model.predict(pkt.raw_adc)
        tbn      = self.tbn_model.step(pkt.temperature, self.dt_hours)
        oi       = self.oi_model.step(pkt.temperature, self.dt_hours)
        dilution = self.dilution_model.predict(pkt.dilution_adc)
        analexrs = 0.0   # retained in DB schema for backward compat; not computed

        feat_vec = np.array(
            [pkt.temperature, soot_pct, pkt.capacitance_pF, tbn, oi, dilution],
            dtype=np.float32
        )

        # Periodic retrain on the ring buffer
        self._ring.append(feat_vec.copy())
        if len(self._ring) > RING_BUFFER_SIZE:
            self._ring.pop(0)
        n = len(self._ring)
        if n >= MIN_TRAIN and (n == MIN_TRAIN or self._tick_count % RETRAIN_EVERY == 0):
            self.ohi_engine.train(np.array(self._ring, dtype=np.float32))

        cluster, label, ohi = self.ohi_engine.predict(feat_vec)
        self._tick_count += 1

        r = ProcessedReading(
            timestamp       = pkt.timestamp,
            engine_hours    = pkt.engine_hours,
            temperature     = pkt.temperature,
            raw_adc         = pkt.raw_adc,
            dilution_adc    = pkt.dilution_adc,
            soot_pct        = soot_pct,
            capacitance_pF  = pkt.capacitance_pF,
            analexrs        = round(analexrs, 2),
            tbn             = tbn,
            oxidation_index = oi,
            dilution        = dilution,
            cluster         = cluster,
            cluster_label   = label,
            ohi             = ohi,
            pca_weights     = json.dumps(self.ohi_engine.weights),
        )
        self.db.insert(r)
        self.on_result(r)

    @staticmethod
    def _default_print(r: ProcessedReading):
        icon = {"HEALTHY": "✓", "WARNING": "⚠", "DEGRADED": "✗"}.get(r.cluster_label, "?")
        bar  = "█" * (r.ohi // 5) + "░" * (20 - r.ohi // 5)
        print(
            f"[{r.cluster_label:<8}] {icon}  OHI:{r.ohi:3d}/100  [{bar}]  "
            f"h={r.engine_hours:6.1f}  "
            f"T={r.temperature:.1f}°C  "
            f"Soot={r.soot_pct:.4f}  "
            f"Cap={r.capacitance_pF:.1f}pF  "
            f"TBN={r.tbn:.1f}%  "
            f"OI={r.oxidation_index:.1f}  "
            f"Dil={r.dilution:.3f}%"
        )

    def offline_train(self) -> dict:
        data = self.db.load_for_training()
        if data.shape[0] == 0:
            return {"error": "No data in database."}
        self.ohi_engine.train(data, verbose=True)
        return {
            "n_samples":   data.shape[0],
            "trained":     self.ohi_engine.trained,
            "ohi_weights": self.ohi_engine.weights,
        }

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ESP-32 Oil Health DAQ Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python daq_pipeline_esp32.py                                       # simulation
  python daq_pipeline_esp32.py --source serial                       # auto-detect port
  python daq_pipeline_esp32.py --source serial --port COM4           # Windows
  python daq_pipeline_esp32.py --source serial --port /dev/ttyUSB0   # Linux
  python daq_pipeline_esp32.py --source serial --port /dev/tty.usbserial-0001  # Mac
  python daq_pipeline_esp32.py --stop                                # stop running instance
  python daq_pipeline_esp32.py --retrain                             # retrain from DB
        """
    )
    parser.add_argument("--source", choices=["sim", "serial"], default="sim",
                        help="Sensor source: sim (default) or serial")
    parser.add_argument("--port",  default=None,
                        help="Serial port — auto-detected if omitted")
    parser.add_argument("--baud",  type=int, default=115200,
                        help="Serial baud rate (default: 115200)")
    parser.add_argument("--engine-hours-start", type=float, default=0.0,
                        dest="engine_hours_start",
                        help="ECU hour-meter offset at session start (e.g. 1200)")
    parser.add_argument("--interval", type=float, default=STREAM_INTERVAL,
                        help=f"Seconds between sensor reads (default: {STREAM_INTERVAL})")
    parser.add_argument("--stop", action="store_true",
                        help="Send stop signal to a running pipeline instance and exit")
    parser.add_argument("--retrain", action="store_true",
                        help="Run offline retrain from DB then exit (pipeline must be stopped first)")
    args = parser.parse_args()

    # ── --stop ────────────────────────────────────────────────────────────────
    if args.stop:
        os.makedirs(str(BASE_DIR), exist_ok=True)
        pathlib.Path(STOP_SENTINEL).touch()
        print(f"[STOP]  Sentinel written to {STOP_SENTINEL}")
        print("[STOP]  The running pipeline will detect this and shut down on the next tick.")
        raise SystemExit(0)

    # ── --retrain ─────────────────────────────────────────────────────────────
    if args.retrain:
        print("[RETRAIN]  Loading data from SQLite for offline retrain…")
        if not os.path.exists(DB_PATH):
            print(f"[ERROR]  No database found at {DB_PATH}.\n"
                  "         Run the pipeline in live or sim mode first.")
            raise SystemExit(1)
        db   = OilHealthDB()
        data = db.load_for_training()
        db.close()
        if data.shape[0] == 0:
            print("[ERROR]  readings table is empty. Run the pipeline first.")
            raise SystemExit(1)
        print(f"[RETRAIN]  {data.shape[0]} samples loaded.")
        engine = OHIEngine()
        engine.train(data, verbose=True)
        print(f"[RETRAIN]  Done.  OHI weights: {engine.weights}")
        raise SystemExit(0)

    # ── Live acquisition ──────────────────────────────────────────────────────
    if args.source == "serial":
        stream = make_serial_stream(
            port               = args.port,          # None → auto-detect
            baud               = args.baud,
            engine_hours_start = args.engine_hours_start,
        )
    else:
        stream = make_simulation_stream()
        print("[SIM]   Simulation mode — degradation over 250 engine hours.")
        print("[SIM]   No hardware required.\n")

    daq = OilHealthDAQ(stream_fn=stream)
    daq.start(interval=args.interval)
