"""
Dual License

For Open-Source Individuals:
MIT License

Copyright (c) 2025 James B. Chapman

Permission is hereby granted, free of charge, to any individual obtaining a copy
of this software and associated documentation files (the "Software"), for personal,
non-commercial use, to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

For Companies:
Commercial use by companies requires a separate license. Contact iconoclastdao@gmail.com
for licensing terms and conditions. Unauthorized commercial use is prohibited.
"""

import numpy as np
import time
from scipy import signal
from typing import Tuple, List, Optional
from sklearn.ensemble import IsolationForest
from pathlib import Path
from ..agents.tcc_logger import TCCLogger

class BioInspiredSecuritySystem:
    def __init__(self, sample_rate: int = 1000, window_size: int = 100,
                 base_threshold: float = 0.1, adaptive_threshold_factor: float = 1.5,
                 log_level: str = "INFO"):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.base_threshold = base_threshold
        self.adaptive_threshold = base_threshold
        self.adaptive_threshold_factor = adaptive_threshold_factor
        self.execution_data = np.array([])
        self.time_points = np.array([])
        self.baseline_frequencies: Optional[np.ndarray] = None
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_history: List[float] = []
        self.logger = TCCLogger(log_level=log_level)
        self.start_time = time.time()

    def _generate_signal(self, data: bytes, is_malicious: bool = False, seed: Optional[int] = None) -> np.ndarray:
        start_time = time.time_ns()
        try:
            if seed is not None:
                np.random.seed(seed)
            t = np.linspace(0, self.window_size / self.sample_rate, self.window_size)
            base_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
            if is_malicious:
                anomaly = 0.8 * np.sin(2 * np.pi * np.random.uniform(40, 60) * t)
                signal_out = base_signal + anomaly
            else:
                signal_out = base_signal + np.random.normal(0, 0.15, self.window_size)
            self.execution_data = signal_out
            self.time_points = t
            self.logger.log(
                "generate_signal", data, signal_out.tobytes(),
                {"is_malicious": is_malicious, "seed": seed, "execution_time_ns": time.time_ns() - start_time}
            )
            return signal_out
        except Exception as e:
            self.logger.log(
                "generate_signal", data, b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time}, "ERROR", "SIGNAL_GENERATION_FAILED"
            )
            raise

    def _compute_fft(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time_ns()
        try:
            window = signal.windows.hann(self.window_size)
            fft_result = np.fft.fft(data * window)
            freqs = np.fft.fftfreq(self.window_size, d=1 / self.sample_rate)
            magnitudes = np.abs(fft_result)
            mask = freqs > 0
            self.logger.log(
                "analyze_frequency", data.tobytes(), magnitudes[mask].tobytes(),
                {"freq_count": len(freqs[mask]), "execution_time_ns": time.time_ns() - start_time}
            )
            return freqs[mask], magnitudes[mask]
        except Exception as e:
            self.logger.log(
                "analyze_frequency", data.tobytes(), b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time}, "ERROR", "FFT_COMPUTATION_FAILED"
            )
            raise

    def analyze_frequency_signature(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._compute_fft(self.execution_data)

    def establish_baseline(self, num_samples: int = 10) -> np.ndarray:
        start_time = time.time_ns()
        try:
            all_magnitudes = []
            for i in range(num_samples):
                self._generate_signal(b"baseline", is_malicious=False, seed=i)
                _, magnitudes = self.analyze_frequency_signature()
                all_magnitudes.append(magnitudes)
            self.baseline_frequencies = np.mean(all_magnitudes, axis=0)
            self.isolation_forest.fit(np.array(all_magnitudes))
            self.logger.log(
                "establish_baseline", b"", self.baseline_frequencies.tobytes(),
                {"num_samples": num_samples, "execution_time_ns": time.time_ns() - start_time}
            )
            return self.baseline_frequencies
        except Exception as e:
            self.logger.log(
                "establish_baseline", b"", b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time}, "ERROR", "BASELINE_FAILED"
            )
            raise

    def detect_anomaly(self, data: bytes) -> Tuple[bool, float, str]:
        if self.baseline_frequencies is None:
            self.logger.log(
                "detect_anomaly", data, b"",
                {"error": "Baseline not established"}, "ERROR", "NO_BASELINE"
            )
            raise ValueError("Baseline not established. Run establish_baseline() first.")

        start_time = time.time_ns()
        try:
            self._generate_signal(data)
            freqs, current_magnitudes = self.analyze_frequency_signature()
            deviation = np.abs(current_magnitudes - self.baseline_frequencies)
            fft_score = np.mean(deviation / (self.baseline_frequencies + 1e-8))
            ml_score = -self.isolation_forest.score_samples([current_magnitudes])[0]
            combined_score = 0.7 * fft_score + 0.3 * ml_score

            if self.anomaly_history:
                self.adaptive_threshold = np.mean(self.anomaly_history[-10:]) * self.adaptive_threshold_factor
            self.anomaly_history.append(combined_score)

            is_anomaly = combined_score > max(self.base_threshold, self.adaptive_threshold)
            explanation = self._explain_anomaly(combined_score, freqs, current_magnitudes) if is_anomaly else ""

            self.logger.log(
                "detect_anomaly", data, np.array([is_anomaly, combined_score]).tobytes(),
                {
                    "anomaly_score": combined_score,
                    "threshold": self.adaptive_threshold,
                    "is_anomaly": is_anomaly,
                    "explanation": explanation,
                    "execution_time_ns": time.time_ns() - start_time
                }
            )
            return is_anomaly, combined_score, explanation
        except Exception as e:
            self.logger.log(
                "detect_anomaly", data, b"",
                {"error": str(e), "execution_time_ns": time.time_ns() - start_time}, "ERROR", "ANOMALY_DETECTION_FAILED"
            )
            raise

    def _explain_anomaly(self, anomaly_score: float, freqs: np.ndarray, magnitudes: np.ndarray) -> str:
        try:
            dominant_idx = np.argmax(np.abs(magnitudes - self.baseline_frequencies))
            dominant_freq = freqs[dominant_idx]
            if anomaly_score > 2.0:
                return f"Severe anomaly: Unusual energy at {dominant_freq:.2f} Hz. Possible attack detected."
            elif anomaly_score > 1.0:
                return f"Moderate anomaly: Elevated activity at {dominant_freq:.2f} Hz. Potential security issue."
            return f"Mild anomaly: Slight deviation at {dominant_freq:.2f} Hz. Monitor for irregularities."
        except Exception as e:
            self.logger.log(
                "explain_anomaly", b"", b"",
                {"error": str(e)}, "ERROR", "ANOMALY_EXPLANATION_FAILED"
            )
            return "Anomaly explanation unavailable."