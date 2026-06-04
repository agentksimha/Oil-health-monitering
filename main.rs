// oil_health_core - Ultra low latency KMeans inference for Raspberry Pi
// Manhattan distance, no heap allocation in hot path, no sqrt

use std::io::{self, BufRead, Write};
use std::time::Instant;
use std::fs;

const N_FEATURES: usize = 5;   // TBN, TAN, Viscosity, Dielectric, Soot
const N_CLUSTERS: usize = 3;   // Healthy / Warning / Degraded
const MODEL_PATH: &str = "/home/pi/oil_health/models/centroids.bin";

// Oil health cluster labels (set after offline training)
// 0 = Healthy, 1 = Warning, 2 = Degraded (assigned post-hoc from Python)
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum OilState {
    Healthy   = 0,
    Warning   = 1,
    Degraded  = 2,
}

impl OilState {
    fn from_cluster(id: usize, cluster_labels: &[u8; N_CLUSTERS]) -> Self {
        match cluster_labels[id] {
            0 => OilState::Healthy,
            1 => OilState::Warning,
            _ => OilState::Degraded,
        }
    }
    fn as_str(&self) -> &'static str {
        match self {
            OilState::Healthy  => "HEALTHY",
            OilState::Warning  => "WARNING",
            OilState::Degraded => "DEGRADED",
        }
    }
}

// Flat array: [cluster0_f0, cluster0_f1, ..., cluster1_f0, ...]
// Stored in normalized space (MinMax scaled 0-1)
#[derive(Debug)]
pub struct KMeansModel {
    centroids:      [f32; N_CLUSTERS * N_FEATURES],
    cluster_labels: [u8;  N_CLUSTERS],   // post-hoc label from Python
    feat_min:       [f32; N_FEATURES],   // for online normalization
    feat_max:       [f32; N_FEATURES],
}

impl KMeansModel {
    /// Load binary model written by Python trainer
    /// Format: 4B magic + centroids (f32 array) + labels (u8) + min/max (f32)
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let mut cursor = 4usize; // skip magic

        let mut centroids = [0f32; N_CLUSTERS * N_FEATURES];
        for v in centroids.iter_mut() {
            let b = &bytes[cursor..cursor+4];
            *v = f32::from_le_bytes([b[0],b[1],b[2],b[3]]);
            cursor += 4;
        }

        let mut cluster_labels = [0u8; N_CLUSTERS];
        for v in cluster_labels.iter_mut() {
            *v = bytes[cursor];
            cursor += 1;
        }

        let mut feat_min = [0f32; N_FEATURES];
        let mut feat_max = [1f32; N_FEATURES];
        for v in feat_min.iter_mut() {
            let b = &bytes[cursor..cursor+4];
            *v = f32::from_le_bytes([b[0],b[1],b[2],b[3]]);
            cursor += 4;
        }
        for v in feat_max.iter_mut() {
            let b = &bytes[cursor..cursor+4];
            *v = f32::from_le_bytes([b[0],b[1],b[2],b[3]]);
            cursor += 4;
        }

        Ok(KMeansModel { centroids, cluster_labels, feat_min, feat_max })
    }

    /// Fallback hardcoded model (paper data centroids, normalized)
    /// Used on first boot before real training data is available
    pub fn default_from_paper() -> Self {
        // Centroids derived from paper Table 7 & 8 data
        // Features: [TBN, TAN, Viscosity, Dielectric, Soot]
        // Normalized with min=[5,2,12,2,0] max=[11,5,32,4,5]
        let centroids: [f32; N_CLUSTERS * N_FEATURES] = [
            // Cluster 0 - Healthy
            0.82, 0.09, 0.14, 0.08, 0.00,
            // Cluster 1 - Warning
            0.55, 0.45, 0.28, 0.35, 0.40,
            // Cluster 2 - Degraded
            0.10, 0.85, 0.75, 0.80, 0.90,
        ];
        // feat ranges from paper
        let feat_min = [5.0f32,  2.0, 12.0, 2.0, 0.0];
        let feat_max = [11.0f32, 5.0, 32.0, 4.0, 5.0];

        KMeansModel {
            centroids,
            cluster_labels: [0, 1, 2],
            feat_min,
            feat_max,
        }
    }

    /// Normalize a raw sensor reading into [0,1]
    /// No allocation, pure stack ops
    #[inline(always)]
    fn normalize(&self, raw: &[f32; N_FEATURES]) -> [f32; N_FEATURES] {
        let mut out = [0f32; N_FEATURES];
        for i in 0..N_FEATURES {
            let range = self.feat_max[i] - self.feat_min[i];
            out[i] = if range > 1e-6 {
                ((raw[i] - self.feat_min[i]) / range).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        out
    }

    /// Manhattan distance between normalized point and a centroid
    /// SIMD-friendly loop, no sqrt, no heap
    #[inline(always)]
    fn manhattan_dist(point: &[f32; N_FEATURES], centroid_start: usize, centroids: &[f32]) -> f32 {
        let mut dist = 0f32;
        for i in 0..N_FEATURES {
            dist += (point[i] - centroids[centroid_start + i]).abs();
        }
        dist
    }

    /// Core inference - returns (cluster_id, OilState, distance, latency_us)
    /// Designed for <1ms on RPi 4
    pub fn predict(&self, raw: &[f32; N_FEATURES]) -> InferenceResult {
        let t0 = Instant::now();

        let norm = self.normalize(raw);

        let mut best_cluster = 0usize;
        let mut best_dist    = f32::MAX;

        // Only 3 centroids - fully unrollable by compiler
        for c in 0..N_CLUSTERS {
            let dist = Self::manhattan_dist(&norm, c * N_FEATURES, &self.centroids);
            if dist < best_dist {
                best_dist    = dist;
                best_cluster = c;
            }
        }

        // Health score: 0-100 (100 = perfect health)
        // Invert distance from healthy centroid (cluster 0)
        let healthy_dist = Self::manhattan_dist(&norm, 0, &self.centroids);
        // max possible manhattan dist in N_FEATURES dims normalized to [0,1] = N_FEATURES
        let health_score = ((1.0 - (healthy_dist / N_FEATURES as f32)) * 100.0)
            .clamp(0.0, 100.0) as u8;

        let state      = OilState::from_cluster(best_cluster, &self.cluster_labels);
        let latency_us = t0.elapsed().as_micros() as u32;

        InferenceResult {
            cluster:      best_cluster as u8,
            state,
            health_score,
            distance:     best_dist,
            latency_us,
            norm_features: norm,
        }
    }
}

#[derive(Debug)]
pub struct InferenceResult {
    pub cluster:       u8,
    pub state:         OilState,
    pub health_score:  u8,         // 0-100
    pub distance:      f32,
    pub latency_us:    u32,
    pub norm_features: [f32; N_FEATURES],
}

impl InferenceResult {
    /// Compact JSON output for downstream consumers (Python dashboard, MQTT, etc.)
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"cluster":{},"state":"{}","health_score":{},"distance":{:.4},"latency_us":{}}}"#,
            self.cluster,
            self.state.as_str(),
            self.health_score,
            self.distance,
            self.latency_us
        )
    }
}

// ── Main loop: reads CSV lines from stdin, writes JSON to stdout ──────────────
// Input  line: "tbn,tan,visc,dielectric,soot"   e.g. "8.5,3.2,14.1,2.3,1.5"
// Output line: JSON InferenceResult
// Designed to be spawned as subprocess by Python DAQ or called from C via FFI
fn main() {
    let model = if std::path::Path::new(MODEL_PATH).exists() {
        KMeansModel::load(MODEL_PATH).unwrap_or_else(|e| {
            eprintln!("[WARN] Model load failed: {}. Using paper defaults.", e);
            KMeansModel::default_from_paper()
        })
    } else {
        eprintln!("[INFO] No trained model found. Using paper bootstrap centroids.");
        KMeansModel::default_from_paper()
    };

    let stdin  = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') { continue; }

        // Parse CSV: tbn,tan,visc,dielectric,soot
        let vals: Vec<f32> = line.split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if vals.len() < N_FEATURES {
            eprintln!("[WARN] Expected {} features, got {}", N_FEATURES, vals.len());
            continue;
        }

        let raw = [vals[0], vals[1], vals[2], vals[3], vals[4]];
        let result = model.predict(&raw);

        writeln!(out, "{}", result.to_json()).ok();
        out.flush().ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healthy_oil_from_paper() {
        let model = KMeansModel::default_from_paper();
        // Fresh oil from paper Table 6: TBN=10.74, TAN=3.34, Visc=13.87, DC=2.2, Soot=0
        let raw = [10.74, 3.34, 13.87, 2.2, 0.0];
        let r = model.predict(&raw);
        println!("Fresh oil result: {:?}", r);
        assert!(r.health_score > 60, "Fresh oil should be healthy");
    }

    #[test]
    fn test_degraded_oil_from_paper() {
        let model = KMeansModel::default_from_paper();
        // Degraded oil Engine 1 at 250h: TBN=6.21, TAN=4.36, Visc=31.73, DC=3.11, Soot=4.57
        let raw = [6.21, 4.36, 31.73, 3.11, 4.57];
        let r = model.predict(&raw);
        println!("Degraded oil result: {:?}", r);
        assert!(r.health_score < 50, "Degraded oil should score low");
    }

    #[test]
    fn test_latency_under_1ms() {
        let model = KMeansModel::default_from_paper();
        let raw = [8.5, 3.2, 14.1, 2.3, 1.5];
        let r = model.predict(&raw);
        assert!(r.latency_us < 1000, "Inference must be <1ms, got {}us", r.latency_us);
    }
}
