from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ENTRY_POINTS_KEY = "entry_points"

PROJECT_KEY = "source"
ENTRY_KEY = "entry"
COMMAND_KEY = "command"
DEPENDS_ON_KEY = "depends_on"
PARAMS_KEY = "parameters"
PARAM_SOURCE_KEY = "parameter_source"

PARAM_TYPE_KEY = "type"
PARAM_DEFAULT_KEY = "default"

SOURCE_TYPE_KEY = "type"
SOURCE_ID_KEY = "id"
SOURCE_CONTENT_KEY = "key"

MLFLOWER_FILENAME = "MLFlower"
MLRPOJECT_FILENAME = "MLProject"


def get_raw_entry_points(
    path: str | Path, entry_key: str | None = None
) -> dict[str, Any]:
    project = load_project(path)
    steps = project.get(ENTRY_POINTS_KEY, {})

    if entry_key is not None:
        return _load_entry(steps[entry_key], entry_key, path)

    return _consolidate_dependency(
        {key: _load_entry(entry_point, key, path) for key, entry_point in steps.items()}
    )


def load_project(path: str | Path) -> dict[str, Any]:
    project_path = _get_file(path, MLFLOWER_FILENAME, MLRPOJECT_FILENAME)

    if project_path is None:
        return {}

    content = project_path.read_text()
    return yaml.load(content, yaml.SafeLoader)


def _load_entry(entry_point: dict[str], key: str, path: str) -> dict[str, Any]:
    source = entry_point.get(PROJECT_KEY, None)
    entry = entry_point.setdefault(ENTRY_KEY, key)
    entry_point.pop(COMMAND_KEY, None)
    entry_point[PARAMS_KEY] = _get_consolidate_params(entry_point)

    # Consolidate depend_on
    depends_on = entry_point.setdefault(DEPENDS_ON_KEY, set())
    if isinstance(depends_on, str):
        entry_point[DEPENDS_ON_KEY] = {depends_on}

    if source is None:
        entry_point[PROJECT_KEY] = path
        return entry_point

    if not Path(source).is_absolute():
        source = Path(path).joinpath(source).resolve()

    new_entry_point = get_raw_entry_points(source, entry)
    new_entry_point.setdefault(PARAMS_KEY, {}).update(entry_point.get(PARAMS_KEY, {}))
    new_entry_point.setdefault(PARAM_SOURCE_KEY, {}).update(
        entry_point.get(PARAM_SOURCE_KEY, {})
    )
    new_entry_point.setdefault(DEPENDS_ON_KEY, set()).update(
        entry_point.get(DEPENDS_ON_KEY, set())
    )

    new_entry_point[PROJECT_KEY] = source
    new_entry_point[ENTRY_KEY] = entry

    return new_entry_point


def _get_consolidate_params(
    entry_point: dict[str, str | dict[str]]
) -> dict[str, dict[str]]:
    entry_point_params = {}
    for param_name, param in entry_point.get(PARAMS_KEY, {}).items():
        if isinstance(param, dict):
            entry_point_params[param_name] = param
        else:
            entry_point_params[param_name] = {PARAM_TYPE_KEY: param}

    return entry_point_params


def _consolidate_dependency(entry_points: dict[str]) -> dict[str]:
    for entry_point in entry_points.values():
        depend_on = entry_point.setdefault(DEPENDS_ON_KEY, set())
        for param in entry_point.get(PARAM_SOURCE_KEY, {}).values():
            param_type = param.get(SOURCE_TYPE_KEY, "parameter")
            if param_type not in ("artifact", "parameter"):
                continue
            depend_on.add(param[SOURCE_ID_KEY])

    return entry_points


def _get_file(path: str | Path, *alternatives: str) -> Path:
    file_name, *alternatives = alternatives
    return next(
        (p for p in Path(path).iterdir() if p.name.upper() == file_name.upper()),
        _get_file(path, *alternatives) if alternatives else None,
    )
