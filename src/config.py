"""Application configuration and filesystem paths."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppPaths:
    root: Path = Path("local_app_data")

    @property
    def uploads(self) -> Path:
        return self.root / "uploads"

    @property
    def charts(self) -> Path:
        return self.root / "charts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def create(self) -> None:
        for directory in (self.uploads, self.charts, self.reports):
            directory.mkdir(parents=True, exist_ok=True)


PATHS = AppPaths()
