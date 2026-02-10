"""
SpectralFM experiment logging.
Dual backend: Weights & Biases (primary) + local JSONL (fallback).
"""
import json
import time
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentLogger:
    """Dual W&B + JSON logger for SpectralFM experiments."""

    def __init__(self, project: str = "SpectralFM",
                 run_name: Optional[str] = None,
                 config: Optional[dict] = None,
                 log_dir: str = "logs",
                 use_wandb: bool = True,
                 wandb_entity: Optional[str] = None,
                 tags: Optional[list] = None):

        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate run name if not provided
        if run_name is None:
            run_name = f"run_{int(time.time())}"
        self.run_name = run_name

        # JSON log file
        self.json_path = self.log_dir / f"{run_name}.jsonl"
        self.json_file = open(self.json_path, 'a')

        # W&B init
        self.wandb_run = None
        if self.use_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config,
                    entity=wandb_entity,
                    tags=tags,
                    reinit=True,
                )
                logger.info(f"W&B run: {self.wandb_run.url}")
            except Exception as e:
                logger.warning(f"W&B init failed: {e}. Falling back to JSON-only.")
                self.use_wandb = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to both backends."""
        # Add timestamp
        entry = {"_timestamp": time.time(), **metrics}
        if step is not None:
            entry["_step"] = step

        # JSON
        self.json_file.write(json.dumps(entry) + '\n')
        self.json_file.flush()

        # W&B
        if self.use_wandb and self.wandb_run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.debug(f"W&B log failed: {e}")

    def log_summary(self, metrics: Dict[str, Any]):
        """Log summary metrics (shown in W&B run overview)."""
        if self.use_wandb and self.wandb_run:
            for k, v in metrics.items():
                wandb.run.summary[k] = v

        # Also save to JSON
        entry = {"_type": "summary", "_timestamp": time.time(), **metrics}
        self.json_file.write(json.dumps(entry) + '\n')
        self.json_file.flush()

    def log_artifact(self, path: str, name: str, artifact_type: str = "model"):
        """Log a file as W&B artifact."""
        if self.use_wandb and self.wandb_run:
            try:
                artifact = wandb.Artifact(name, type=artifact_type)
                artifact.add_file(path)
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.debug(f"W&B artifact failed: {e}")

    def finish(self):
        """Close logging backends."""
        self.json_file.close()
        if self.use_wandb and self.wandb_run:
            wandb.finish()
        logger.info(f"Logs saved to {self.json_path}")
