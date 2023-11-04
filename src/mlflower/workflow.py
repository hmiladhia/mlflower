from __future__ import annotations

import contextlib
from typing import Iterator

import mlflow
from mlflow.entities import Run, RunStatus
from mlflow.projects import SubmittedRun

from .entry_point import EntryPoint, get_entry_points
from .graph_utils import topological_sort
from .workflow_run import WorkflowRun


class Workflow(SubmittedRun):
    def __init__(
        self,
        entry_points: dict[str, EntryPoint],
        active_run: Run | None = None,
        root_entry_point: str | None = None,
    ):
        self._is_internal = active_run is None
        self.active_run = active_run if active_run else mlflow.start_run()
        root_entry_point = root_entry_point or self.active_run.data.tags.get(
            "mlflow.project.entryPoint", "root"
        )

        self.runtime_context: dict[str, SubmittedRun] = {}
        self.workflow_runs = {
            key: WorkflowRun(
                entry_point, run=self.active_run if key == root_entry_point else None
            )
            for key, entry_point in entry_points.items()
        }

        self._resolution_order = iter(
            topological_sort(entry_points, root=root_entry_point)
        )
        self._status = RunStatus.SCHEDULED

    @classmethod
    def from_project_uri(
        cls, project_uri, active_run=None, root_entry_point: str | None = None
    ):
        entry_points = get_entry_points(project_uri)

        return cls(entry_points, active_run, root_entry_point)

    def __iter__(self) -> Iterator[tuple[str, WorkflowRun]]:
        for key in self._resolution_order:
            yield key, self.workflow_runs[key]

    def run(self, **run_args) -> None:
        if self.get_status() != RunStatus.SCHEDULED:
            return

        run_args = get_run_args(self.active_run, run_args)

        self._status = RunStatus.RUNNING
        for key, wrun in self:
            if not wrun.wait_dependencies(self.runtime_context):
                return self.fail()

            self.runtime_context[key] = wrun.submit(self.workflow_runs, run_args)

        if not self.wait():
            return self.fail()

        return self._end_run(RunStatus.FINISHED)

    def wait(self) -> bool:
        for key in list(self.runtime_context):
            submitted_run = self.runtime_context.pop(key)
            if not submitted_run.wait():
                return False

        return True

    def cancel(self):
        return self._cleanup(RunStatus.KILLED)

    def fail(self):
        return self._cleanup(RunStatus.FAILED)

    def get_status(self):
        return self._status

    @property
    def run_id(self):
        return self.active_run.info.run_id

    def _cleanup(self, status: RunStatus) -> None:
        if RunStatus.is_terminated(self.get_status()):
            return

        for job_key in list(self.runtime_context.keys()):
            submitted_run = self.runtime_context.pop(job_key)
            with contextlib.suppress(AttributeError):
                # submitted_run.cancel doesn't work on Windows (mlflow 2.8.0)
                submitted_run.cancel()

        self._end_run(status)

    def _end_run(self, status: RunStatus) -> None:
        self._status = status

        if self._is_internal:
            mlflow.end_run(RunStatus.to_string(status))


def get_run_args(active_run, run_args):
    backend = run_args.pop("backend", None) or active_run.data.tags.get(
        "mlflow.project.backend", "local"
    )
    env_manager = run_args.pop("env_manager", None) or active_run.data.tags.get(
        "mlflow.project.env", None
    )

    return {
        "backend": backend,
        "env_manager": env_manager,
        "synchronous": run_args.pop("sequential", False),
        **run_args,
    }
