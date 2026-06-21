"""Project context handed to binding ops and custom checks.

Centralises file access so every op shares cached loaders and the same
encoding convention (utf-8-sig first, latin-1 fallback — survives BOMs and
Nordic characters in survey exports).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


class Context:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self._json_cache: dict[str, object] = {}
        self._csv_cache: dict[str, pd.DataFrame] = {}

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.project_root / p

    def load_json(self, rel: str) -> dict:
        if rel not in self._json_cache:
            with open(self.path(rel), encoding="utf-8-sig") as f:
                self._json_cache[rel] = json.load(f)
        return self._json_cache[rel]

    def load_csv(self, rel: str, **kwargs) -> pd.DataFrame:
        if rel not in self._csv_cache:
            path = self.path(rel)
            try:
                df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1", low_memory=False, **kwargs)
            self._csv_cache[rel] = df
        return self._csv_cache[rel]


def load_plugin(ops_path: Path) -> None:
    """Import a per-project ops.py so its decorators populate the registries.

    The plugin's directory is temporarily prepended to sys.path so it can
    import sibling helper modules.
    """
    ops_path = Path(ops_path).resolve()
    if not ops_path.exists():
        raise FileNotFoundError(f"ops module not found: {ops_path}")
    spec = importlib.util.spec_from_file_location(f"claimpin_plugin_{ops_path.stem}", ops_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ops_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ops_path.parent))
