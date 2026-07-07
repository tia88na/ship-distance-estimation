from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class RecordConfig:
    """Stores the selected record name and root dataset directory."""

    name: str
    root: str


@dataclass
class PathConfig:
    """Stores video, sensor CSV, and output directory paths."""

    rgb_video: str
    thermal_video: str
    sensor_csv: str
    output_dir: str


@dataclass
class CameraConfig:
    """Stores camera height and default RGB/thermal field-of-view values."""

    height_m: float
    rgb_fov_h_deg: float
    rgb_fov_v_deg: float
    thermal_fov_h_deg: float
    thermal_fov_v_deg: float


@dataclass
class ModelConfig:
    """Stores model-related configuration values."""

    yolo_path: str


@dataclass
class AppConfig:
    """Groups all project configuration sections loaded from YAML."""

    record: RecordConfig
    paths: PathConfig
    camera: CameraConfig
    model: ModelConfig

    @classmethod
    def from_yaml(cls: type[AppConfig], config_path: str | Path) -> AppConfig:
        """Load application configuration from a YAML file."""
        path = Path(config_path)

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return cls(
            record=RecordConfig(**data["record"]),
            paths=PathConfig(**data["paths"]),
            camera=CameraConfig(**data["camera"]),
            model=ModelConfig(**data["model"]),
        )
