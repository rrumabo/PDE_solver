"""Minimal DiagnosticManager: track/save min/max/mean/mass/L2/L2 error."""

import csv
import logging
import numpy as np
import yaml
from typing import Iterable, Dict, List, Tuple, Optional
from scipy.integrate import simpson


class DiagnosticManager:
    """Tiny logger for common diagnostics ."""
    def __init__(
        self,
        dx: float,
        dy: Optional[float] = None,
        u_ref: Optional[np.ndarray] = None,
        track: Tuple[str, ...] = ("min", "max", "mean", "mass", "l2_error", "l2_norm"),
    ) -> None:
        if dx is None:
            raise ValueError("dx must be provided")
        self.dx = float(dx)
        self.dy = None if dy is None else float(dy)
        self.dimension = 1 if dy is None else 2
        self.u_ref = None if u_ref is None else np.asarray(u_ref)
        self.track = set(track)
        self.records: List[Dict[str, float]] = []
        self.logger = None

    # Logging to file (optional)
    def enable_logging(self, log_path: str) -> None:
        logging.basicConfig(filename=log_path, filemode="w", level=logging.INFO, format="%(asctime)s - %(message)s")
        self.logger = logging.getLogger(__name__)
        self.logger.info("DiagnosticManager logging initialized.")

    # Helpers for integrals
    def _mass(self, u: np.ndarray) -> float:
        # 1D: ∫ u dx ; 2D: ∬ u dxdy
        if self.dy is None:
            return float(simpson(np.real_if_close(u), dx=self.dx))
        if u.ndim == 2:
            # Rectangle rule (robust for periodic grids)
            return float(np.sum(np.real_if_close(u)) * self.dx * self.dy)
        return float(np.sum(np.real_if_close(u)) * self.dx * self.dy)

    def _l2(self, u: np.ndarray) -> float:
        # ||u||_2 = sqrt(∫ |u|^2)
        uu = np.abs(u) ** 2
        if self.dy is None:
            return float(np.sqrt(simpson(uu, dx=self.dx)))
        if u.ndim == 2:
            tmp = simpson(uu, dx=self.dy, axis=0)
            return float(np.sqrt(simpson(tmp, dx=self.dx)))
        return float(np.sqrt(np.sum(uu) * self.dx * self.dy))

    def _l2_error(self, u: np.ndarray) -> float:
        if self.u_ref is None:
            return float("nan")
        if u.shape != self.u_ref.shape:
            raise ValueError(f"Shape mismatch: u {u.shape} vs u_ref {self.u_ref.shape}")
        return self._l2(u - self.u_ref)

    # Record one step
    def track_step(self, u: np.ndarray, t: float, **extras: float) -> None:
        u = np.asarray(u)
        rec: Dict[str, float] = {"time": float(t), "dimension": self.dimension}
        if "min" in self.track:  rec["min"] = float(np.min(u))
        if "max" in self.track:  rec["max"] = float(np.max(u))
        if "mean" in self.track: rec["mean"] = float(np.real_if_close(np.mean(u)))
        if "mass" in self.track: rec["mass"] = self._mass(u)
        if "l2_error" in self.track and self.u_ref is not None: rec["l2_error"] = self._l2_error(u)
        if "l2_norm" in self.track: rec["l2_norm"] = self._l2(u)
        if "residual" in extras: rec["residual"] = float(extras["residual"])
        self.records.append(rec)
        if self.logger: self.logger.info(f"t={t}: {rec}")

    # Save all records to CSV
    def save_csv(self, path: str) -> None:
        if not self.records: return
        keys = list(self.records[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(self.records)

    # Save compact summary to YAML
    def save_yaml(self, path: str) -> None:
        if not self.records:
            return
        first, last = self.records[0], self.records[-1]
        out: Dict[str, float] = {"time_final": float(last.get("time", 0.0))}
        if "mass" in self.track and "mass" in first and "mass" in last:
            m0, m1 = float(first["mass"]), float(last["mass"])
            out["mass_initial"], out["mass_final"] = m0, m1
            out["rel_mass_drift"] = float((m1 - m0) / (abs(m0) + 1e-15))
        if "l2_norm" in self.track and "l2_norm" in first and "l2_norm" in last:
            out["l2_initial"], out["l2_final"] = float(first["l2_norm"]), float(last["l2_norm"])
        out["min_final"] = float(last.get("min", np.nan))
        out["max_final"] = float(last.get("max", np.nan))
        out["mean_final"] = float(last.get("mean", np.nan))
        with open(path, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False)