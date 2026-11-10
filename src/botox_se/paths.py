"""Central filesystem paths for the project."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @property
    def data(self) -> Path:
        return self.repo_root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def reference(self) -> Path:
        return self.data / "reference"

    @property
    def config(self) -> Path:
        return self.repo_root / "config"

    def ensure_dirs(self) -> None:
        for p in (self.raw, self.interim, self.processed, self.reference):
            p.mkdir(parents=True, exist_ok=True)
