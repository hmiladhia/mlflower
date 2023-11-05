from __future__ import annotations

import os
import sys
from typing import Any

import click
import mlflow
from mlflow import ActiveRun, MlflowClient
from mlflow.entities import Param, RunStatus
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_EXPERIMENT_NAME

from mlflower.project import load_project

from .workflow import Workflow


@click.command()
@click.argument("uri", type=click.STRING, required=False, default=None)
@click.option(
    "--entry-point",
    "-e",
    metavar="NAME",
    default=None,
    help="MLFlower entry point within project. default: root",
)
@click.option(
    "--param-list",
    "-P",
    metavar="NAME=VALUE",
    multiple=True,
    help="A parameter for the run, of the form -P name=value. Provided parameters that "
    "are not in the list of parameters for an entry point will be passed to the "
    "corresponding entry point as command-line arguments in the form `--name value`",
)
@click.option(
    "--docker-args",
    "-A",
    metavar="NAME=VALUE",
    multiple=True,
    help="A `docker run` argument or flag, of the form -A name=value (e.g. -A gpus=all) "
    "or -A name (e.g. -A t). The argument will then be passed as "
    "`docker run --name value` or `docker run --name` respectively. ",
)
@click.option(
    "--experiment-name",
    envvar=MLFLOW_EXPERIMENT_NAME.name,
    help="Name of the experiment under which to launch the run. If not "
    "specified, 'experiment-id' option will be used to launch run.",
)
@click.option(
    "--experiment-id",
    envvar=MLFLOW_EXPERIMENT_ID.name,
    type=click.STRING,
    help="ID of the experiment under which to launch the run.",
)
@click.option(
    "--backend",
    "-b",
    metavar="BACKEND",
    default=None,
    help="Execution backend to use for run. Supported values: 'local', 'databricks', "
    "kubernetes (experimental)",
)
@click.option(
    "--backend-config",
    "-c",
    metavar="FILE",
    help="Path to JSON file (must end in '.json') or JSON string which will be passed "
    "as config to the backend.",
)
@click.option(
    "--env-manager",
    default=None,
    type=click.STRING,
    help="""
    If specified, create an environment for MLmodel using the specified
    environment manager. The following values are supported:

    \b
    - local: use the local environment
    - virtualenv: use virtualenv (and pyenv for Python version management)
    - conda: use conda

    If unspecified, default to virtualenv.
""",
)
@click.option(
    "--storage-dir",
    envvar="MLFLOW_TMP_DIR",
    help="Only valid when ``backend`` is local. "
    "MLflow downloads artifacts from distributed URIs passed to parameters of "
    "type 'path' to subdirectories of storage_dir.",
)
@click.option(
    "--run-name",
    metavar="RUN_NAME",
    help="The name to give the MLflow Run associated with the project execution. If not specified, "
    "the MLflow Run name is left unset.",
)
@click.option(
    "--sequential",
    is_flag=True,
    default=False,
    show_default=True,
    help="Whether to run the steps sequentially or in parallel (when possible)",
)
@click.option(
    "--build-image",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Only valid for Docker projects. If specified, build a new Docker image that's based on "
        "the image specified by the `image` field in the MLproject file, and contains files in the "
        "project directory."
    ),
)
def main(
    uri: str | None,
    entry_point: str | None,
    param_list: list[str] | None,
    docker_args: list[str] | None,
    experiment_name: str | None,
    experiment_id: str | None,
    backend: str | None,
    backend_config: str | None,
    env_manager: str | None,
    storage_dir: str | None,
    run_name: str | None,
    build_image: str | None,
    sequential: bool,
) -> None:

    project_uri = uri or os.getcwd()
    experiment_id = get_experiment_id(project_uri, experiment_id, experiment_name)

    param_dict = _to_dict(param_list)

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as active_run:
        update_params(active_run, param_dict)

        workflow = Workflow.from_project_uri(
            project_uri, active_run, root_entry_point=entry_point
        )

        workflow.run(
            {
                "docker_args": _to_dict(docker_args, True),
                "backend": backend,
                "backend_config": backend_config,
                "env_manager": env_manager,
                "storage_dir": storage_dir,
                "build_image": build_image,
                "sequential": sequential,
                "experiment_id": experiment_id,
            }
        )

        if workflow.get_status() in (RunStatus.FAILED, RunStatus.KILLED):
            sys.exit(1)


def update_params(active_run: ActiveRun, param_dict: dict[str, Any]) -> None:
    if not param_dict:
        return

    params = [Param(key, value) for key, value in param_dict.items()]
    active_run.data.params.update(param_dict)
    MlflowClient().log_batch(active_run.info.run_id, params=params)


def _to_dict(arguments: list[str], allow_flags: bool = False) -> dict[str]:
    user_dict = {}
    for arg in arguments:
        name, *values = arg.split("=", 1)

        if len(values) == 0 and allow_flags:
            value = True
        elif len(values) == 1:
            value = next(iter(values))
        else:
            raise ValueError

        user_dict[name] = value
    return user_dict


def get_experiment_id(
    project_uri: str,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> str:
    if experiment_id:
        return experiment_id

    experiment_name = experiment_name or load_project(project_uri).get("name")

    if experiment_name is None:
        return None

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return mlflow.create_experiment(experiment_name)

    return experiment.experiment_id


if __name__ == "__main__":
    main()
