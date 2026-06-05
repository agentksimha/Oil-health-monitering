"""
Oil Health DAQ Pipeline
=======================
- Ingests real-time sensor stream (serial / MQTT / simulated)
- Buffers readings into a rolling ring buffer (acquisition only — no retraining on hot path)
- Spawns Rust inference subprocess for <1ms latency KMeans predictions
- PCA on PC1 loadings derives data-driven feature weights for the Oil Health Index
- Retraining is FULLY OFFLINE via OfflineTrainer — never triggered during inference

Architecture:
  Sensor → StreamReader → RingBuffer → [Rust subprocess] → Dashboard
                                              ↑
                                     (centroids.bin — written offline)
                                              ↑
                                      OfflineTrainer (idle / scheduled / CLI)
                                        1. Elbow method → optimal K
                                        2. KMeans on full feature space
                                        3. PCA PC1 loadings → OHI weights
                                        4. Refit interpolation splines
                                        5. Atomic binary export → Rust reloads

Sensor stream variants:
  - Serial  : CSV lines "engine_hours,visc,dielectric,soot[,tbn,tan]"
              engine_hours_start offsets cumulative ECU hours to the current session
              tbn_tan_from_sensor=True passes TBN/TAN directly from hardware
  - MQTT    : JSON payload with engine_hours, viscosity, dielectric, soot [,tbn, tan]
  - Sim     : synthetic degradation curve for desktop testing
"""

import subprocess
import threading
import struct
import time
import json
import csv
import os
import io
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH       = "/home/pi/oil_health/models/centroids.bin"
INTERP_PATH      = "/home/pi/oil_health/models/interpolation.npz"   # persisted spline knots
DATA_LOG_PATH    = "/home/pi/oil_health/data/readings.csv"
LAB_DATA_PATH    = "/home/pi/oil_health/data/lab_samples.csv"        # sparse lab measurements
RUST_BINARY      = "/home/pi/oil_health/oil_health_core"
RING_BUFFER_SIZE = 500       # rolling window kept in memory for offline training
STREAM_INTERVAL  = 1.0       # seconds between sensor reads (target: 1-2s on RPi)

# Offline training config — used only by OfflineTrainer, never on the hot path
ELBOW_K_RANGE    = range(2, 7)   # test k=2..6 during offline elbow search
N_CLUSTERS       = 3             # default; overridden by elbow result after first offline run

# Feature split:
#   DIRECT_FEATURES  — measurable in real-time from sensors
#   DERIVED_FEATURES — computed via interpolation model; updated when lab samples arrive
DIRECT_FEATURES  = ["viscosity", "dielectric", "soot"]
DERIVED_FEATURES = ["tbn", "tan"]          # TAN is the oxidation proxy
FEATURES         = ["tbn", "tan", "viscosity", "dielectric", "soot"]  # order must match Rust

# Bootstrap knots from paper (used before any real lab data is collected)
# Format: {feature: {"t": [...hours], "y": [...values]}}
PAPER_BOOTSTRAP_KNOTS: dict = {
    # Exact values from Table 8, Engine 1 (Balashanmugam & Gobalakichenin, 2016)
    "tbn": {
        "t": [0,     50,   100,  150,  200,  250 ],
        "y": [10.74, 8.79, 8.95, 7.81, 7.96, 6.21],
    },
    "tan": {
        "t": [0,    50,   100,  150,  200,  250 ],
        "y": [3.34, 3.52, 4.04, 3.98, 3.65, 4.36],
    },
}

# Physical thresholds from paper (used to label clusters post-hoc)
THRESHOLDS = {
    "tbn":        {"min": 5.0,  "direction": "above"},  # TBN < 5 = degraded
    "tan":        {"max": 4.5,  "direction": "below"},  # TAN > 4.5 = degraded
    "viscosity":  {"max": 18.5, "direction": "below"},  # Visc > 18.5 = degraded
    "dielectric": {"max": 3.5,  "direction": "below"},  # DC > 3.5 = degraded
    "soot":       {"max": 4.0,  "direction": "below"},  # Soot > 4% = degraded
}

# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class SensorReading:
    timestamp:    float
    engine_hours: float  # cumulative engine operating hours — the x-axis for interpolation
    # Direct real-time sensor measurements
    viscosity:    float   # Kinematic viscosity at 100 degC cSt
    dielectric:   float   # Dielectric constant
    soot:         float   # Soot %
    # Derived via interpolation model — filled in by InterpolationModel.estimate()
    tbn:          float = 0.0   # Total Base Number mg KOH/g
    tan:          float = 0.0   # Total Acid Number mg KOH/g  (oxidation proxy)
    engine_id:    str   = "engine_1"

    def to_array(self) -> np.ndarray:
        """Returns [tbn, tan, viscosity, dielectric, soot] — order must match Rust"""
        return np.array([self.tbn, self.tan, self.viscosity,
                         self.dielectric, self.soot], dtype=np.float32)

    def to_csv_row(self) -> list:
        return [self.timestamp, self.engine_hours, self.engine_id,
                self.tbn, self.tan, self.viscosity, self.dielectric, self.soot]


@dataclass
class LabSample:
    """Sparse lab measurement - arrives infrequently (every 50-250 engine hours)"""
    engine_hours: float
    tbn:          float
    tan:          float
    engine_id:    str = "engine_1"


@dataclass
class OilHealthResult:
    timestamp:    float
    cluster:      int
    state:        str        # HEALTHY / WARNING / DEGRADED
    health_score: int        # 0-100
    distance:     float
    latency_us:   int
    reading:      SensorReading


# ── Model export (Python → Rust binary format) ────────────────────────────────
MAGIC = b"OHI1"   # 4-byte magic number

def export_model_binary(centroids: np.ndarray,
                         cluster_labels: list[int],
                         feat_min: np.ndarray,
                         feat_max: np.ndarray,
                         path: str = None):
    """
    Binary layout:
      4B  magic
      N_CLUSTERS * N_FEATURES * 4B  centroids (f32 LE)
      N_CLUSTERS * 1B               cluster labels (u8)
      N_FEATURES * 4B               feat_min (f32 LE)
      N_FEATURES * 4B               feat_max (f32 LE)
    """
    path = path or MODEL_PATH  # resolve at call time so patching MODEL_PATH works
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    buf = io.BytesIO()
    buf.write(MAGIC)

    for v in centroids.flatten():
        buf.write(struct.pack("<f", float(v)))
    for label in cluster_labels:
        buf.write(struct.pack("B", int(label)))
    for v in feat_min:
        buf.write(struct.pack("<f", float(v)))
    for v in feat_max:
        buf.write(struct.pack("<f", float(v)))

    # Atomic write via rename
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(buf.getvalue())
    os.replace(tmp_path, path)
    print(f"[MODEL] Exported to {path} ({buf.tell()} bytes)")


# ── Cluster labelling (post-hoc from paper thresholds) ───────────────────────
def label_clusters(kmeans: KMeans,
                   scaler: MinMaxScaler,
                   n_features: int) -> list[int]:
    """
    Assign 0=Healthy / 1=Warning / 2=Degraded to each cluster
    by checking how many threshold violations the centroid represents
    """
    centroids_raw = scaler.inverse_transform(kmeans.cluster_centers_)
    feat_names = FEATURES[:n_features]
    scores = []

    for centroid in centroids_raw:
        violations = 0
        for i, feat in enumerate(feat_names):
            t = THRESHOLDS.get(feat)
            if t is None:
                continue
            val = centroid[i]
            if t["direction"] == "above" and val < t.get("min", 0):
                violations += 2          # critical: TBN below minimum
            elif t["direction"] == "below" and val > t.get("max", 999):
                violations += 1
        scores.append(violations)

    # Sort: 0 violations = Healthy (label 0), most = Degraded (label 2)
    # Use actual k from kmeans, not the global N_CLUSTERS constant —
    # these can differ when elbow picks a different K than the default.
    k = len(kmeans.cluster_centers_)
    order = np.argsort(scores)
    labels = [0] * k
    for rank, cluster_idx in enumerate(order):
        if rank == 0:
            labels[cluster_idx] = 0   # Healthy
        elif rank == k - 1:
            labels[cluster_idx] = 2   # Degraded
        else:
            labels[cluster_idx] = 1   # Warning
    return labels


# ── Interpolation model (singleton) ──────────────────────────────────────────
class InterpolationModel:
    """
    Singleton loaded ONCE at engine startup.
    Holds per-feature CubicSpline fits for TBN and TAN (derived/lab-measured params).

    Lifecycle:
      1. Startup  — load from INTERP_PATH if it exists, else bootstrap from paper knots
      2. Runtime  — estimate(engine_hours) returns TBN, TAN for any timestamp in ~0.1ms
      3. Update   — add_lab_sample() called when a new sparse lab result arrives;
                    refit() rebuilds the splines and persists them atomically
                    This is cheap (microseconds for <100 knots) so it runs inline,
                    not in a background thread.

    Why singleton: spline knots are engine-specific state that must be consistent
    across every estimate() call. Loading from disk on each sample would cost
    10-50ms (file I/O) and risk mid-series inconsistency.
    """
    _instance: Optional["InterpolationModel"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "InterpolationModel":
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._initialised = False
                cls._instance = obj
        return cls._instance

    def initialise(self, path: str = INTERP_PATH) -> "InterpolationModel":
        """
        Call once at engine startup. Thread-safe via internal lock.
        """
        with self._lock:
            if self._initialised:
                return self
            # knot store: {feature: {"t": np.ndarray, "y": np.ndarray}}
            self._knots: dict[str, dict] = {}
            self._splines: dict[str, CubicSpline] = {}
            self._extrapolation_poly: dict[str, np.poly1d] = {}
            self._path = path

            if os.path.exists(path):
                self._load(path)
                print(f"[INTERP] Loaded splines from {path}")
            else:
                self._bootstrap_from_paper()
                print("[INTERP] No saved splines found — bootstrapped from paper knots")

            self._initialised = True
        return self

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(self, engine_hours: float) -> dict[str, float]:
        """
        Evaluate fitted splines at the given engine_hours.
        Returns dict {"tbn": ..., "tan": ...}.
        Falls back to linear extrapolation beyond the last knot.
        ~0.1ms on RPi 4 (spline eval is pure numpy).
        """
        result = {}
        for feat in DERIVED_FEATURES:
            spline = self._splines.get(feat)
            knots  = self._knots.get(feat, {})
            t_arr  = knots.get("t", np.array([]))
            y_arr  = knots.get("y", np.array([]))

            if spline is None or len(t_arr) < 2:
                # No data yet — return last known value or 0
                result[feat] = float(y_arr[-1]) if len(y_arr) else 0.0
                continue

            t_min, t_max = float(t_arr[0]), float(t_arr[-1])

            if t_min <= engine_hours <= t_max:
                # Interpolation — spline is exact here
                val = float(spline(engine_hours))
            else:
                # Extrapolation beyond training range:
                # Use linear fit on last 2 knots to avoid spline oscillation
                poly = self._extrapolation_poly.get(feat)
                if poly is not None:
                    val = float(poly(engine_hours))
                else:
                    val = float(spline(np.clip(engine_hours, t_min, t_max)))

            # Clamp to physical bounds
            if feat == "tbn":
                val = max(0.0, val)
            elif feat == "tan":
                val = max(0.0, val)

            result[feat] = round(val, 4)
        return result

    def add_lab_sample(self, sample: LabSample):
        """
        Ingest a new sparse lab measurement, refit splines immediately,
        and persist updated knots to disk.
        Thread-safe; called from DAQ tick when a new lab value arrives.
        """
        with self._lock:
            for feat in DERIVED_FEATURES:
                val = getattr(sample, feat)
                knots = self._knots.setdefault(feat, {"t": np.array([]), "y": np.array([])})

                t_arr = knots["t"]
                y_arr = knots["y"]

                # Deduplicate: if a knot at this hour already exists, update it
                existing_idx = np.where(np.isclose(t_arr, sample.engine_hours, atol=0.5))
                if len(existing_idx[0]):
                    y_arr[existing_idx[0][0]] = val
                else:
                    t_arr = np.append(t_arr, sample.engine_hours)
                    y_arr = np.append(y_arr, val)
                    # Keep sorted by time
                    sort_idx = np.argsort(t_arr)
                    t_arr, y_arr = t_arr[sort_idx], y_arr[sort_idx]

                knots["t"], knots["y"] = t_arr, y_arr

            self._refit_all()
            self._save(self._path)
            print(f"[INTERP] Updated splines with lab sample at {sample.engine_hours:.1f}h")

    def knot_counts(self) -> dict[str, int]:
        return {f: len(self._knots.get(f, {}).get("t", [])) for f in DERIVED_FEATURES}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _bootstrap_from_paper(self):
        """Seed knots from paper data so the model works on day 1."""
        for feat, data in PAPER_BOOTSTRAP_KNOTS.items():
            self._knots[feat] = {
                "t": np.array(data["t"], dtype=np.float64),
                "y": np.array(data["y"], dtype=np.float64),
            }
        self._refit_all()
        self._save(self._path)

    def _refit_all(self):
        """Rebuild CubicSpline for every derived feature. Called under lock."""
        for feat in DERIVED_FEATURES:
            knots = self._knots.get(feat, {})
            t_arr = knots.get("t", np.array([]))
            y_arr = knots.get("y", np.array([]))
            n = len(t_arr)
            if n < 2:
                self._splines[feat] = None
                continue
            # CubicSpline needs >=2 unique points; use 'clamped' bc when n>=4
            # to enforce zero second-derivative at the knots (smooth degradation)
            bc = "clamped" if n >= 4 else "not-a-knot"
            self._splines[feat] = CubicSpline(t_arr, y_arr, bc_type=bc, extrapolate=False)

            # Linear extrapolation beyond last two knots (avoids spline curl)
            if n >= 2:
                slope = (float(y_arr[-1]) - float(y_arr[-2])) / \
                        (float(t_arr[-1]) - float(t_arr[-2]) + 1e-9)
                intercept = float(y_arr[-1]) - slope * float(t_arr[-1])
                self._extrapolation_poly[feat] = np.poly1d([slope, intercept])

    def _save(self, path: str):
        """Atomic save of knot arrays to .npz"""
        if not path:
            return
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        save_dict = {}
        for feat in DERIVED_FEATURES:
            knots = self._knots.get(feat, {"t": np.array([]), "y": np.array([])})
            save_dict[f"{feat}_t"] = knots["t"]
            save_dict[f"{feat}_y"] = knots["y"]
        # np.savez appends .npz automatically, so strip it before passing to savez
        tmp_base = path[:-4] + "_tmp" if path.endswith(".npz") else path + "_tmp"
        np.savez(tmp_base, **save_dict)   # writes tmp_base.npz
        os.replace(tmp_base + ".npz", path)

    def _load(self, path: str):
        """Load knot arrays from .npz and refit."""
        data = np.load(path)
        for feat in DERIVED_FEATURES:
            t_key, y_key = f"{feat}_t", f"{feat}_y"
            if t_key in data and y_key in data:
                self._knots[feat] = {
                    "t": data[t_key].astype(np.float64),
                    "y": data[y_key].astype(np.float64),
                }
        self._refit_all()


# ── Offline trainer (never runs during inference) ─────────────────────────────
class OfflineTrainer:
    """
    Runs ONLY when the pipeline is idle or explicitly triggered from CLI.
    Must never be called from OilHealthDAQ._tick() or any hot-path code.

    Responsibilities:
      1. Elbow method — finds optimal K (k=2..6) from inertia curve.
         Expensive O(k * n * iter) — fine offline, unacceptable during inference.
      2. KMeans fit — on the full normalized feature space (all 5 features).
         Clustering is NOT done on PCA components; all features are kept.
      3. PCA (PC1 loadings) — run AFTER clustering purely to derive feature
         weights for the Oil Health Index score formula.  The cluster assignments
         themselves do not use PCA at all.
      4. Interpolation refit — rebuilds TBN/TAN splines from the full lab CSV
         so the running interpolation model is consistent with the new clusters.
      5. Binary export — atomic write of centroids.bin; Rust reloads on next
         restart (triggered by the caller, not here).
    """

    def __init__(self, interp_model: "InterpolationModel",
                 lab_logger: "LabSampleLogger"):
        self.scaler       = MinMaxScaler()
        self.kmeans       = None
        self.pca          = PCA(n_components=1)   # only PC1 needed for OHI weights
        self.is_trained   = False
        self.optimal_k    = N_CLUSTERS
        self.ohi_weights: dict[str, float] = {}   # populated after train()
        self._interp_model = interp_model
        self._lab_logger   = lab_logger

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self, data: np.ndarray) -> dict:
        """
        Full offline retrain.  Call this only when the DAQ loop is stopped
        or from a separate offline process — never from _tick().

        Steps:
          1. Normalize all 5 features with MinMaxScaler
          2. Elbow method → pick optimal_k
          3. KMeans on full normalized feature space with optimal_k
          4. Post-hoc cluster labelling from paper thresholds
          5. PCA on X_norm → PC1 loadings → OHI feature weights
          6. Export centroids.bin for Rust
          7. Refit interpolation splines from all logged lab samples

        Returns a diagnostics dict (logged by caller).
        """
        # Bootstrap mode: allow as few as k*3 samples (used when seeding from paper data)
        min_samples = max(self.optimal_k if self.kmeans else N_CLUSTERS, 2) * 3
        if len(data) < min_samples:
            return {"error": f"Need at least {min_samples} samples, have {len(data)}"}

        # 1. Normalize
        X_norm = self.scaler.fit_transform(data)

        # 2. Elbow method — find optimal K
        self.optimal_k = self._elbow(X_norm)
        print(f"[OFFLINE] Elbow → optimal K = {self.optimal_k}")

        # 3. KMeans on full feature space (all 5 dims, NOT PCA-reduced)
        #    sklearn uses Euclidean internally; Rust uses Manhattan at inference.
        #    On MinMax-normalized data the cluster boundaries are nearly identical.
        self.kmeans = KMeans(
            n_clusters  = self.optimal_k,
            init        = "k-means++",
            n_init      = 10,
            max_iter    = 300,
            random_state= 42,
        )
        self.kmeans.fit(X_norm)

        # 4. Label clusters from physical thresholds (Healthy / Warning / Degraded)
        cluster_labels = label_clusters(self.kmeans, self.scaler, data.shape[1])

        # 5. PCA — run on the same X_norm, but ONLY to derive OHI weights from PC1.
        #    PC1 = primary degradation axis.  |loading_i| / Σ|loadings| → weight_i.
        #    This makes the OHI score data-driven rather than hardcoded.
        self.pca.fit(X_norm)
        loadings = np.abs(self.pca.components_[0])
        weights  = loadings / (loadings.sum() + 1e-9)
        self.ohi_weights = {feat: round(float(w), 4) for feat, w in zip(FEATURES, weights)}

        # 6. Export binary model for Rust
        export_model_binary(
            centroids      = self.kmeans.cluster_centers_,
            cluster_labels = cluster_labels,
            feat_min       = self.scaler.data_min_,
            feat_max       = self.scaler.data_max_,
        )

        # 7. Refit interpolation splines from all persisted lab samples.
        #    Ensures the running interpolation model reflects all real measurements
        #    accumulated since the last offline run.
        self._refit_interpolation()

        self.is_trained = True

        return {
            "n_samples":      len(data),
            "optimal_k":      self.optimal_k,
            "inertia":        round(self.kmeans.inertia_, 4),
            "cluster_labels": cluster_labels,
            "cluster_sizes":  np.bincount(self.kmeans.labels_).tolist(),
            "pc1_var_ratio":  round(float(self.pca.explained_variance_ratio_[0]), 4),
            "ohi_weights":    self.ohi_weights,
        }

    def compute_ohi(self, reading: "SensorReading") -> int:
        """
        Oil Health Index 0-100 using PCA-derived weights.
        Higher = healthier.  Falls back to equal weights if not yet trained.

        Formula: OHI = 100 × Σ(weight_i × normalized_health_i)
        where normalized_health_i maps each feature to [0=worst, 1=best].
        """
        weights = self.ohi_weights if self.ohi_weights else \
                  {f: 0.2 for f in FEATURES}   # equal fallback before first train

        raw = reading.to_array()   # [tbn, tan, visc, dc, soot]
        feat_min = np.array([5.0,  2.0, 12.0, 2.0, 0.0])
        feat_max = np.array([11.0, 5.0, 32.0, 4.0, 5.0])

        # For each feature, map to [0=worst, 1=best] according to degradation direction
        # "above" threshold → higher raw = better;  "below" → lower raw = better
        health = np.zeros(len(FEATURES), dtype=np.float32)
        for i, feat in enumerate(FEATURES):
            t = THRESHOLDS.get(feat)
            norm = float(np.clip((raw[i] - feat_min[i]) / (feat_max[i] - feat_min[i] + 1e-8), 0, 1))
            if t and t["direction"] == "above":
                health[i] = norm         # high TBN = healthy
            else:
                health[i] = 1.0 - norm   # low Soot/TAN/Visc/DC = healthy

        ohi = sum(weights.get(feat, 0.2) * float(health[i])
                  for i, feat in enumerate(FEATURES))
        return max(0, min(100, int(ohi * 100)))

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _elbow(self, X_norm: np.ndarray) -> int:
        """
        Elbow method over ELBOW_K_RANGE.
        Picks K where the rate of inertia decrease drops below 15%
        (the 'knee' of the curve).  Falls back to N_CLUSTERS if no clear elbow.
        Only called offline — O(k * n * iter) is too slow for the hot path.
        """
        inertias = {}
        for k in ELBOW_K_RANGE:
            if len(X_norm) < k * 3:
                break
            km = KMeans(n_clusters=k, init="k-means++", n_init=5,
                        max_iter=100, random_state=42)
            km.fit(X_norm)
            inertias[k] = km.inertia_
            print(f"[ELBOW]  k={k}  inertia={km.inertia_:.4f}")

        if len(inertias) < 2:
            return N_CLUSTERS

        ks      = sorted(inertias)
        vals    = [inertias[k] for k in ks]
        drops   = [(vals[i] - vals[i+1]) / (vals[i] + 1e-9) for i in range(len(vals)-1)]
        # Find first k where the marginal gain of adding another cluster < 15%
        for i, drop in enumerate(drops):
            if drop < 0.15:
                chosen = ks[i]
                print(f"[ELBOW]  Chosen K={chosen} (marginal gain fell to {drop:.1%})")
                return chosen

        # Inertia kept dropping steeply all the way to the end.
        # Happens with augmented/synthetic data where clusters are artificially tight.
        # Fall back to N_CLUSTERS (physically grounded: 3 degradation stages).
        print(f"[ELBOW]  No clear knee found — falling back to K={N_CLUSTERS} (physical default)")
        return N_CLUSTERS

    def _refit_interpolation(self):
        """
        Reload all persisted lab samples and refit the interpolation model.
        Called at the end of every offline train so splines stay in sync.
        """
        samples = self._lab_logger.load_all()
        if not samples:
            print("[OFFLINE] No lab samples on disk — interpolation model unchanged.")
            return
        for s in samples:
            self._interp_model.add_lab_sample(s)
        print(f"[OFFLINE] Refit interpolation splines from {len(samples)} lab samples.")


# ── Ring buffer ───────────────────────────────────────────────────────────────
class SensorRingBuffer:
    def __init__(self, maxlen: int = RING_BUFFER_SIZE):
        self._buf  = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, reading: SensorReading):
        with self._lock:
            self._buf.append(reading)

    def to_numpy(self) -> np.ndarray:
        with self._lock:
            return np.array([r.to_array() for r in self._buf], dtype=np.float32)

    def __len__(self):
        with self._lock:
            return len(self._buf)


# ── Rust subprocess wrapper ───────────────────────────────────────────────────
class RustInferenceEngine:
    """
    Manages the Rust binary as a persistent subprocess.
    Communicates via stdin/stdout pipe (CSV in, JSON out).
    Restart is automatic if binary is updated (model retrain).
    """
    def __init__(self, binary_path: str = RUST_BINARY):
        self.binary_path = binary_path
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        if not os.path.exists(self.binary_path):
            print(f"[WARN] Rust binary not found at {self.binary_path}. "
                   "Run: cargo build --release --manifest-path oil_health/Cargo.toml")
            self._proc = None
            return
        self._proc = subprocess.Popen(
            [self.binary_path],
            stdin  = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            bufsize = 0,    # unbuffered — critical for latency
        )
        print(f"[ENGINE] Rust inference engine started (PID {self._proc.pid})")

    def restart(self):
        """Call after model export so Rust reloads updated centroids"""
        with self._lock:
            if self._proc:
                self._proc.terminate()
                self._proc.wait()
            time.sleep(0.05)   # let file system flush
            self._start()

    def predict(self, reading: SensorReading) -> Optional[dict]:
        """
        Send one CSV line, receive one JSON line.
        Total round-trip target: <2ms on RPi 4
        """
        if self._proc is None:
            return self._python_fallback(reading)

        line = f"{reading.tbn},{reading.tan},{reading.viscosity},"  \
               f"{reading.dielectric},{reading.soot}\n"

        with self._lock:
            try:
                self._proc.stdin.write(line.encode())
                self._proc.stdin.flush()
                out = self._proc.stdout.readline()
                return json.loads(out.decode().strip())
            except (BrokenPipeError, json.JSONDecodeError, OSError) as e:
                print(f"[ERROR] Rust engine: {e}. Restarting...")
                self._start()
                return None

    def _python_fallback(self, reading: SensorReading) -> dict:
        """Pure Python Manhattan KMeans — fallback when Rust binary unavailable"""
        # Default centroids (paper values, normalized)
        centroids = np.array([
            [0.82, 0.09, 0.14, 0.08, 0.00],
            [0.55, 0.45, 0.28, 0.35, 0.40],
            [0.10, 0.85, 0.75, 0.80, 0.90],
        ], dtype=np.float32)
        feat_min = np.array([5.0,  2.0, 12.0, 2.0, 0.0])
        feat_max = np.array([11.0, 5.0, 32.0, 4.0, 5.0])

        raw  = reading.to_array()
        norm = np.clip((raw - feat_min) / (feat_max - feat_min + 1e-8), 0, 1)
        dists = np.sum(np.abs(centroids - norm), axis=1)   # Manhattan
        best  = int(np.argmin(dists))
        healthy_dist = float(dists[0])
        health_score = max(0, min(100, int((1.0 - healthy_dist / 5.0) * 100)))
        state = ["HEALTHY", "WARNING", "DEGRADED"][best]
        return {"cluster": best, "state": state,
                "health_score": health_score,
                "distance": float(dists[best]), "latency_us": 0}


# ── Data loggers ──────────────────────────────────────────────────────────────
class DataLogger:
    def __init__(self, path: str = DATA_LOG_PATH):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "engine_hours", "engine_id"] + FEATURES
                )

    def log(self, reading: SensorReading):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(reading.to_csv_row())


class LabSampleLogger:
    """Persists sparse lab measurements so splines survive restarts."""
    def __init__(self, path: str = LAB_DATA_PATH):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(["engine_hours", "engine_id", "tbn", "tan"])

    def log(self, sample: LabSample):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(
                [sample.engine_hours, sample.engine_id, sample.tbn, sample.tan]
            )

    def load_all(self) -> list:
        """Reload historical lab samples on startup to warm up the interpolator."""
        samples = []
        if not os.path.exists(self.path):
            return samples
        with open(self.path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    samples.append(LabSample(
                        engine_hours=float(row["engine_hours"]),
                        tbn=float(row["tbn"]),
                        tan=float(row["tan"]),
                        engine_id=row.get("engine_id", "engine_1"),
                    ))
                except (KeyError, ValueError):
                    continue
        return samples


# ── Main DAQ pipeline ─────────────────────────────────────────────────────────
class OilHealthDAQ:
    """
    Full data acquisition and inference pipeline.
    Designed for Raspberry Pi with 1-2 second update intervals.

    CRITICAL DESIGN CONSTRAINT:
      The _tick() hot path does ZERO training work.
      - Sensor read        : ~1ms
      - Spline estimate    : ~0.1ms
      - Rust KMeans        : ~0.5ms
      Total per tick       : <2ms  (well within 1-2s interval)

      Retraining happens ONLY via offline_trainer.train() called from:
        - CLI: `python daq_pipeline.py --retrain`
        - Scheduled cron when engine is stopped
        - Manual call: daq.offline_train()

    Startup sequence:
      1. InterpolationModel singleton initialised — loads spline knots from disk
         (or bootstraps from paper data) ONCE. No further disk I/O on hot path.
      2. Historical lab samples replayed into interpolator so splines are warm.
      3. Rust inference engine spawned.
      4. Main loop: sensor tick → interpolate TBN/TAN → KMeans via Rust → emit.
         Ring buffer is populated every tick for later offline use.

    Lab sample injection (out-of-band):
      Call daq.ingest_lab_sample(LabSample(...)) whenever a real lab result arrives.
      The interpolator refits splines and persists knots inline (<1ms for <100 knots).
      Next sensor tick will automatically use the updated curve.
    """
    def __init__(self,
                 stream_fn,           # callable() -> SensorReading | None
                 on_result=None):     # callback(OilHealthResult)

        self.stream_fn      = stream_fn
        self.on_result      = on_result or (lambda r: print(r))

        # ── Interpolation singleton: initialise once, reuse forever ────────
        self.interp_model = InterpolationModel().initialise(INTERP_PATH)

        self.buffer         = SensorRingBuffer()
        self.lab_logger     = LabSampleLogger()

        # OfflineTrainer holds PCA weights and elbow state.
        # It is initialized here so OHI weights survive across multiple
        # offline_train() calls, but train() is never called from _tick().
        self.offline_trainer = OfflineTrainer(
            interp_model = self.interp_model,
            lab_logger   = self.lab_logger,
        )

        self.engine     = RustInferenceEngine()
        self.logger     = DataLogger()

        # Replay persisted lab samples so splines reflect all historical data
        self._replay_lab_samples()

        self._sample_count  = 0
        self._running       = False

    def _replay_lab_samples(self):
        """
        On startup, feed all logged lab samples back into the interpolator.
        This re-warms the splines from disk without re-reading the binary .npz
        (the .npz already has the fitted knots, but this also handles cases where
        lab_samples.csv has entries newer than the last .npz save).
        """
        samples = self.lab_logger.load_all()
        if samples:
            for sample in samples:
                self.interp_model.add_lab_sample(sample)
            print(f"[INTERP] Replayed {len(samples)} lab samples. "
                  f"Knot counts: {self.interp_model.knot_counts()}")

    def ingest_lab_sample(self, sample: LabSample):
        """
        Public API: call this whenever a new sparse lab measurement arrives.
        Refits splines and persists knots atomically. Safe to call from any thread.
        Next tick will use updated TBN/TAN estimates automatically.
        """
        self.interp_model.add_lab_sample(sample)
        self.lab_logger.log(sample)
        print(f"[LAB] Ingested sample at {sample.engine_hours:.1f}h — "
              f"TBN={sample.tbn:.2f}, TAN={sample.tan:.2f}")

    def start(self, interval: float = STREAM_INTERVAL):
        self._running = True
        print(f"[DAQ] Pipeline started. Sampling every {interval}s")
        while self._running:
            t0 = time.monotonic()
            self._tick()
            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, interval - elapsed)
            time.sleep(sleep_t)

    def stop(self):
        self._running = False

    def _tick(self):
        # 1. Read raw sensor (direct fields only: viscosity, dielectric, soot)
        reading = self.stream_fn()
        if reading is None:
            return

        # 2. Estimate TBN and TAN from singleton interpolation model
        #    Uses pre-fitted CubicSpline evaluated at reading.engine_hours.
        #    Cost: ~0.1ms (pure numpy spline eval, no disk I/O — singleton loaded at startup).
        #    Skip if the stream already provided TBN/TAN from onboard sensor hardware.
        if reading.tbn == 0.0 and reading.tan == 0.0:
            derived = self.interp_model.estimate(reading.engine_hours)
            reading.tbn = derived["tbn"]
            reading.tan = derived["tan"]

        # 3. Log full reading (direct + derived) to CSV
        self.logger.log(reading)

        # 4. Buffer for offline retraining — NO training happens here
        self.buffer.push(reading)
        self._sample_count += 1

        # 5. Inference via Rust (<1ms)
        raw_result = self.engine.predict(reading)
        if raw_result is None:
            return

        # 6. OHI score using PCA-derived weights (or equal fallback before first offline train)
        health_score = self.offline_trainer.compute_ohi(reading)

        result = OilHealthResult(
            timestamp    = reading.timestamp,
            cluster      = raw_result["cluster"],
            state        = raw_result["state"],
            health_score = health_score,
            distance     = raw_result["distance"],
            latency_us   = raw_result["latency_us"],
            reading      = reading,
        )

        # 7. Emit result to dashboard / alert system
        self.on_result(result)
        # NOTE: No retraining here — ever. Call offline_train() when the engine is idle.

    def offline_train(self) -> dict:
        """
        Trigger a full offline retrain from the current ring buffer.

        WHEN TO CALL:
          - After stopping the DAQ loop (engine shut down for the day)
          - From a cron job / systemd timer during non-operating hours
          - From CLI: python daq_pipeline.py --retrain
          - Manually after accumulating enough new lab samples

        After this returns, call daq.engine.restart() so Rust picks up the
        new centroids.bin.  Restart is intentionally NOT done here to keep
        this method side-effect-free and testable.

        Returns the diagnostics dict from OfflineTrainer.train().
        """
        if self._running:
            print("[WARN] offline_train() called while pipeline is RUNNING. "
                  "Stop the DAQ loop first to avoid data races on the ring buffer.")
        data = self.buffer.to_numpy()
        diagnostics = self.offline_trainer.train(data)
        print(f"[OFFLINE] Training complete: {diagnostics}")
        return diagnostics


# ── Sensor stream implementations ─────────────────────────────────────────────
# Streams supply ONLY direct sensor fields: engine_hours, viscosity, dielectric, soot.
# TBN and TAN are NOT read from sensors — they are estimated by InterpolationModel
# inside OilHealthDAQ._tick() after the reading is returned here.

def make_serial_stream(port: str = "/dev/ttyUSB0", baud: int = 9600,
                       engine_hours_start: float = 0.0,
                       tbn_tan_from_sensor: bool = False):
    """
    Real hardware stream from sensor over serial.

    Expected CSV line formats:
      tbn_tan_from_sensor=False  (default):
        "engine_hours,visc,dielectric,soot"
        TBN/TAN are NOT in the stream; they will be estimated by InterpolationModel.

      tbn_tan_from_sensor=True:
        "engine_hours,visc,dielectric,soot,tbn,tan"
        The sensor unit computes TBN/TAN onboard (e.g. impedance-based estimation).
        Values are passed through directly — InterpolationModel is bypassed.

    engine_hours_start:
        Offset added to every engine_hours value from the sensor.
        Use this when the ECU hour-meter is not reset between sessions, but your
        pipeline run starts mid-life.  Example: ECU reads 1200h at startup →
        pass engine_hours_start=1200.0 so the pipeline sees cumulative hours.
    """
    import serial
    ser = serial.Serial(port, baud, timeout=2)
    n_expected = 6 if tbn_tan_from_sensor else 4

    def read():
        line = ser.readline().decode().strip()
        if not line:
            return None
        vals = [float(v) for v in line.split(",")]
        if len(vals) < n_expected:
            return None

        cumulative_hours = vals[0] + engine_hours_start

        if tbn_tan_from_sensor:
            # Sensor provides all 6 fields; TBN/TAN set directly, skips interpolation
            return SensorReading(
                timestamp    = time.time(),
                engine_hours = cumulative_hours,
                viscosity    = vals[1],
                dielectric   = vals[2],
                soot         = vals[3],
                tbn          = vals[4],   # non-zero → _tick() will NOT overwrite
                tan          = vals[5],
            )
        else:
            # Standard: only 3 direct sensor values; TBN/TAN filled by InterpolationModel
            return SensorReading(
                timestamp    = time.time(),
                engine_hours = cumulative_hours,
                viscosity    = vals[1],
                dielectric   = vals[2],
                soot         = vals[3],
                # tbn=0.0, tan=0.0 → _tick() will call interp_model.estimate()
            )
    return read


def make_mqtt_stream(broker: str = "localhost", topic: str = "oil/sensor"):
    """
    MQTT stream for wireless sensor nodes.
    Payload JSON must include: engine_hours, viscosity, dielectric, soot.
    TBN/TAN are derived — do not include them in the payload.
    """
    import paho.mqtt.client as mqtt
    latest = [None]
    def on_message(client, userdata, msg):
        d = json.loads(msg.payload)
        latest[0] = SensorReading(
            timestamp    = time.time(),
            engine_hours = float(d["engine_hours"]),
            viscosity    = float(d["viscosity"]),
            dielectric   = float(d["dielectric"]),
            soot         = float(d["soot"]),
            engine_id    = d.get("engine_id", "engine_1"),
        )
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(broker)
    client.subscribe(topic)
    client.loop_start()
    def read():
        r = latest[0]; latest[0] = None; return r
    return read


def make_simulation_stream(seed: int = 42, hours_per_step: float = 0.5):
    """
    Simulated stream for testing on desktop / before hardware is available.
    Simulates gradual degradation over ~500 time steps (~250 engine hours).

    Only direct sensor values are generated here (viscosity, dielectric, soot).
    TBN/TAN are intentionally omitted — they will be filled in by the
    InterpolationModel singleton after the reading is returned.

    To test lab sample ingestion, use make_simulation_lab_samples() in parallel.
    """
    rng  = np.random.default_rng(seed)
    step = [0]

    def read():
        t = step[0]
        step[0] += 1
        engine_hours = t * hours_per_step
        progress     = min(engine_hours / 250.0, 1.0)   # 0=fresh, 1=fully degraded at 250h

        viscosity  = 13.9  + 18.0 * progress**2 + rng.normal(0, 0.20)
        dielectric =  2.1  +  1.8 * progress    + rng.normal(0, 0.05)
        soot       =  0.0  +  5.0 * progress**1.5 + rng.normal(0, 0.10)

        return SensorReading(
            timestamp    = time.time(),
            engine_hours = engine_hours,
            viscosity    = max(12.0, viscosity),
            dielectric   = max(2.0,  dielectric),
            soot         = max(0.0,  soot),
        )
    return read


def make_simulation_lab_samples(hours_per_sample: float = 50.0,
                                 seed: int = 99) -> list:
    """
    Generate simulated sparse lab samples for testing ingest_lab_sample().
    Call this at startup and feed results into daq.ingest_lab_sample() one-by-one
    to verify that TBN/TAN estimates update correctly after each lab result.
    Returns a list of LabSample objects at 0, 50, 100, 150, 200, 250 hours.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for h in np.arange(0, 260, hours_per_sample):
        progress = min(h / 250.0, 1.0)
        tbn = max(3.0, 10.74 - 4.53 * progress + rng.normal(0, 0.15))
        tan = max(2.0,  2.50  + 1.86 * progress + rng.normal(0, 0.10))
        samples.append(LabSample(engine_hours=float(h), tbn=tbn, tan=tan))
    return samples


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Oil Health DAQ Pipeline")
    parser.add_argument("--source", choices=["sim", "serial", "mqtt"],
                        default="sim", help="Sensor source")
    parser.add_argument("--port",   default="/dev/ttyUSB0",
                        help="Serial port (--source serial)")
    parser.add_argument("--broker", default="localhost",
                        help="MQTT broker host (--source mqtt)")
    parser.add_argument("--engine-hours-start", type=float, default=0.0,
                        dest="engine_hours_start",
                        help="ECU hour-meter offset at session start (serial only). "
                             "Pass the cumulative hours shown on the ECU when "
                             "the pipeline starts, e.g. --engine-hours-start 1200")
    parser.add_argument("--tbn-tan-from-sensor", action="store_true",
                        dest="tbn_tan_from_sensor",
                        help="Sensor sends TBN and TAN directly (6-field CSV). "
                             "When set, InterpolationModel is bypassed for TBN/TAN.")
    parser.add_argument("--retrain", action="store_true",
                        help="Run offline retrain on buffered data then exit. "
                             "Use after stopping a live session or from cron. "
                             "Pipeline does NOT start in live mode when this flag is set.")
    args = parser.parse_args()

    # ── Offline retrain mode (no live sensor loop) ────────────────────────────
    if args.retrain:
        print("[OFFLINE] Retrain mode — loading buffered data from disk...")
        # Load all logged readings from CSV into a numpy array for retraining
        import pandas as pd
        if not os.path.exists(DATA_LOG_PATH):
            print(f"[ERROR] No data log found at {DATA_LOG_PATH}. "
                  "Run the pipeline in live mode first to collect data.")
            raise SystemExit(1)

        df   = pd.read_csv(DATA_LOG_PATH, usecols=FEATURES)
        data = df.dropna().values.astype(np.float32)
        print(f"[OFFLINE] Loaded {len(data)} samples from {DATA_LOG_PATH}")

        interp  = InterpolationModel().initialise(INTERP_PATH)
        lab_log = LabSampleLogger()
        trainer = OfflineTrainer(interp_model=interp, lab_logger=lab_log)
        diag    = trainer.train(data)
        print(f"[OFFLINE] Complete.\n  Optimal K : {diag.get('optimal_k')}\n"
              f"  Inertia   : {diag.get('inertia')}\n"
              f"  OHI Wts   : {diag.get('ohi_weights')}\n"
              f"  PC1 var   : {diag.get('pc1_var_ratio')}")
        print("[OFFLINE] centroids.bin written. Restart the DAQ pipeline to pick it up.")
        raise SystemExit(0)

    # ── Live acquisition mode ─────────────────────────────────────────────────
    if args.source == "serial":
        stream = make_serial_stream(
            port                = args.port,
            engine_hours_start  = args.engine_hours_start,
            tbn_tan_from_sensor = args.tbn_tan_from_sensor,
        )
    elif args.source == "mqtt":
        stream = make_mqtt_stream(args.broker)
    else:
        stream = make_simulation_stream()

    def on_result(r: OilHealthResult):
        bar = "\u2588" * (r.health_score // 5) + "\u2591" * (20 - r.health_score // 5)
        tbn_tag = "" if r.reading.tbn == 0.0 else "(est)" if not args.tbn_tan_from_sensor else "(sens)"
        print(f"[{r.state:<8}] OHI:{r.health_score:3d}/100  [{bar}]  "
              f"h={r.reading.engine_hours:6.1f}  "
              f"TBN:{r.reading.tbn:.2f}{tbn_tag}  TAN:{r.reading.tan:.2f}  "
              f"Visc:{r.reading.viscosity:.1f}  DC:{r.reading.dielectric:.2f}  "
              f"Soot:{r.reading.soot:.1f}%  ({r.latency_us}\u00b5s)")

    daq = OilHealthDAQ(stream_fn=stream, on_result=on_result)

    # In simulation mode, pre-seed lab samples to demonstrate interpolation.
    # In production: call daq.ingest_lab_sample(LabSample(...)) each time
    # the lab sends a new TBN/TAN measurement.
    if args.source == "sim":
        print("[SIM] Ingesting simulated lab samples to seed interpolation model...")
        for lab in make_simulation_lab_samples():
            daq.ingest_lab_sample(lab)
        print(f"[SIM] Knot counts after seeding: {daq.interp_model.knot_counts()}")
        print("[SIM] TBN/TAN will now be estimated from fitted splines.")
        print("[SIM] To retrain KMeans after this session: python daq_pipeline.py --retrain")
        print()

    try:
        daq.start(interval=1.0)
    except KeyboardInterrupt:
        print("\n[DAQ] Stopped.")
        daq.stop()
        print("[DAQ] Run  python daq_pipeline.py --retrain  to retrain on collected data.")
