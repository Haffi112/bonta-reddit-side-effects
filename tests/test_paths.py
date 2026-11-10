from pathlib import Path

from botox_se.paths import Paths


def test_paths_resolves_subdirs_relative_to_repo_root(tmp_path: Path) -> None:
    p = Paths(repo_root=tmp_path)
    assert p.raw == tmp_path / "data" / "raw"
    assert p.interim == tmp_path / "data" / "interim"
    assert p.processed == tmp_path / "data" / "processed"
    assert p.reference == tmp_path / "data" / "reference"


def test_paths_ensures_dirs_exist(tmp_path: Path) -> None:
    p = Paths(repo_root=tmp_path)
    p.ensure_dirs()
    assert p.raw.is_dir()
    assert p.interim.is_dir()
    assert p.processed.is_dir()
    assert p.reference.is_dir()
