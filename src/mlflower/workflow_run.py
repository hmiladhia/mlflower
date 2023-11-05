from __future__ import annotations

from typing import Any

import mlflow
from mlflow.entities import Run
from mlflow.projects import SubmittedRun

from .entry_point import EntryPoint
from .project import SOURCE_CONTENT_KEY, SOURCE_ID_KEY, SOURCE_TYPE_KEY


class OrchestrationError(Exception):
    pass


class WorkflowRun:
    def __init__(self, entry_point: EntryPoint, run: Run | None = None):
        self.entry_point = entry_point

        self._submitted_run: SubmittedRun | None = None
        self._run = run

    @property
    def run(self) -> Run:
        if self._run is not None:
            return self._run

        if self._submitted_run is None:
            raise OrchestrationError()

        self._run = mlflow.get_run(self._submitted_run.run_id)

        return self._run

    def wait_dependencies(self, runtime_context: dict[str, SubmittedRun]) -> bool:
        for dependency in self.entry_point.depends_on:
            if dependency not in runtime_context:
                continue

            run_dependency = runtime_context[dependency]

            if not run_dependency.wait():
                return False

            runtime_context.pop(dependency)

        return True

    def submit(
        self, w_runs: dict[str, WorkflowRun], args: dict | None = None
    ) -> SubmittedRun:
        if self._submitted_run is not None:
            raise OrchestrationError()

        self._submitted_run = mlflow.run(
            self.entry_point.source,
            self.entry_point.entry,
            parameters=self._resolve_params(w_runs),
            run_name=self.entry_point.entry,
            **(args or {}),
        )

        return self._submitted_run

    def _resolve_params(self, w_runs: dict[str, WorkflowRun]) -> dict[str, Any]:
        return {
            key: get_param(param, w_runs)
            for key, param in self.entry_point.parameter_source.items()
        }


def get_param(param: dict, w_runs: dict[str, WorkflowRun]) -> Any:
    source_type = param[SOURCE_TYPE_KEY]
    entry_point_id = param[SOURCE_ID_KEY]
    key = param[SOURCE_CONTENT_KEY]

    if source_type == "artifact":
        run = w_runs[entry_point_id].run
        return run.info.artifact_uri + "/" + key

    if source_type == "parameter":
        wrun = w_runs[entry_point_id]
        run_params = wrun.run.data.params

        if key in run_params:
            return run_params[key]

        return wrun.entry_point.defaults[key]

    raise ValueError(f"Unsupported source type: {source_type}")
