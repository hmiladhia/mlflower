from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .project import PARAM_DEFAULT_KEY, get_raw_entry_points


@dataclass
class EntryPoint:
    source: str | None = None
    entry: str = "main"
    parameters: dict[str] = field(default_factory=dict)
    workflow_parameters: dict[str] = field(default_factory=dict)
    depends_on: set[str] = field(default_factory=set)

    @property
    def defaults(self) -> dict[str, Any]:
        return {
            key: param[PARAM_DEFAULT_KEY]
            for key, param in self.parameters.items()
            if PARAM_DEFAULT_KEY in param
        }


def get_entry_points(path: str | Path) -> dict[str, EntryPoint]:
    entry_points = get_raw_entry_points(path)

    return {key: EntryPoint(**entry) for key, entry in entry_points.items()}
