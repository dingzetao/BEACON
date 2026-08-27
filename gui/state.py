"""Config state (editable) vs run state (job progress / results)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .defaults import (
    DEFAULT_CHECKPOINTS,
    DEFAULT_CORE_DIR,
    DEFAULT_OUTPUT_DIR,
)


@dataclass
class ConfigState:
    """User-editable inference settings (not mutated by the worker except via UI)."""

    # Prefer absolute paths on the compute server — do not upload multi-hundred-MB WSIs.
    input_mode: str = "file"  # "file" | "directory"
    input_path: str = ""
    output_dir: str = str(DEFAULT_OUTPUT_DIR)

    run_tumor_egfr: bool = True
    run_mac_ereg: bool = True
    weights_tumor_egfr: str = str(DEFAULT_CHECKPOINTS["Tumor_EGFR"])
    weights_mac_ereg: str = str(DEFAULT_CHECKPOINTS["Mac_EREG"])
    uni_weights_path: str = ""  # empty → TRIDENT default / env

    device: str = "auto"  # auto | cuda | cpu
    patch_size: int = 128
    stride: int = 128
    batch_size: int = 64
    k_neighbors: int = 6
    mask_min_fraction: float = 0.10
    make_thumbnail: bool = True

    def resolved_targets(self) -> list[tuple[str, Path]]:
        out: list[tuple[str, Path]] = []
        if self.run_tumor_egfr:
            out.append(("Tumor_EGFR", Path(self.weights_tumor_egfr)))
        if self.run_mac_ereg:
            out.append(("Mac_EREG", Path(self.weights_mac_ereg)))
        return out


@dataclass
class RunState:
    """Job runtime state; updated from a background thread (GUI polls)."""

    status: str = "idle"  # idle | running | done | error
    stage: str = ""
    progress: float = 0.0  # 0–1
    logs: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    run_dir: str = ""

    def reset(self) -> None:
        self.status = "idle"
        self.stage = ""
        self.progress = 0.0
        self.logs.clear()
        self.results.clear()
        self.error = ""
        self.run_dir = ""

    def log(self, msg: str) -> None:
        self.logs.append(msg)
        if len(self.logs) > 2000:
            self.logs = self.logs[-1500:]


# Shared singletons for the NiceGUI process
config = ConfigState()
# Default: directory of H&E images under gui/WSI_data/
config.input_mode = "directory"
config.input_path = str(DEFAULT_CORE_DIR)

run = RunState()
