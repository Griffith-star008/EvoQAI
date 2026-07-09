"""
Experiment Logger for AQIP Research Pipeline.
Tracks configurations, seeds, metrics, and outputs for full reproducibility.
"""
import json
import os
import hashlib
import platform
from datetime import datetime


class ExperimentLogger:
    """Structured metadata tracker for research experiments."""

    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, "tracking", f"{experiment_name}_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.metadata = {
            "experiment_name": experiment_name,
            "run_id": self.run_id,
            "timestamp_start": datetime.now().isoformat(),
            "timestamp_end": None,
            "environment": self._capture_environment(),
            "config": {},
            "random_seeds": {},
            "metrics": {},
            "artifacts": [],
            "status": "RUNNING",
        }

    def _capture_environment(self) -> dict:
        """Capture the execution environment for reproducibility."""
        return {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

    def log_config(self, config: dict):
        """Log experiment configuration (hyperparameters, model settings, etc.)."""
        self.metadata["config"].update(config)
        print(f"[ExperimentLogger] Config logged: {list(config.keys())}")

    def log_seeds(self, seeds: dict):
        """Log random seeds for reproducibility."""
        self.metadata["random_seeds"].update(seeds)
        print(f"[ExperimentLogger] Seeds logged: {seeds}")

    def log_metric(self, name: str, value: float, step: int = None):
        """Log a single metric value, optionally with a step index."""
        if name not in self.metadata["metrics"]:
            self.metadata["metrics"][name] = []
        entry = {"value": value, "timestamp": datetime.now().isoformat()}
        if step is not None:
            entry["step"] = step
        self.metadata["metrics"][name].append(entry)

    def log_artifact(self, filepath: str, description: str = ""):
        """Register an artifact (model checkpoint, plot, report, etc.)."""
        checksum = self._compute_checksum(filepath) if os.path.exists(filepath) else "N/A"
        self.metadata["artifacts"].append({
            "path": filepath,
            "description": description,
            "checksum_sha256": checksum,
            "timestamp": datetime.now().isoformat(),
        })
        print(f"[ExperimentLogger] Artifact registered: {filepath}")

    def _compute_checksum(self, filepath: str) -> str:
        """Compute SHA-256 checksum for artifact integrity verification."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def finish(self, status: str = "COMPLETED"):
        """Finalize the experiment run and save metadata."""
        self.metadata["timestamp_end"] = datetime.now().isoformat()
        self.metadata["status"] = status

        metadata_path = os.path.join(self.run_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"[ExperimentLogger] Experiment '{self.experiment_name}' finished.")
        print(f"[ExperimentLogger] Metadata saved to: {metadata_path}")
        return metadata_path


if __name__ == "__main__":
    logger = ExperimentLogger("AQII_Convergence_Test")
    logger.log_config({
        "optimizer": "Self-Evolving DAG",
        "evolution_cycles": 100,
        "w_max": 20,
        "loss_weights": {"alpha": 0.3, "beta": 0.2, "gamma": 0.15},
    })
    logger.log_seeds({"numpy": 42, "torch": 42, "qiskit": 42})

    for step in range(10):
        logger.log_metric("aqii_score", 50.0 + step * 3.6, step=step)
        logger.log_metric("loss", 50.0 - step * 4.0, step=step)

    logger.finish()
