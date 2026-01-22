import os

import typer

import wandb


def link_model(artifact_path: str, alias: str = "production") -> None:
    if not artifact_path:
        raise typer.BadParameter("artifact_path is required")

    # Parse "entity/project/artifact:version"
    try:
        entity, project, artifact_name_version = artifact_path.split("/", 2)
        artifact_name, _version = artifact_name_version.split(":", 1)
    except ValueError as e:
        raise typer.BadParameter(f"Unexpected artifact path format: {artifact_path}") from e

    api = wandb.Api()  # uses WANDB_API_KEY from env
    artifact = api.artifact(artifact_path)

    # Link into model registry under the same entity
    target_path = f"{entity}/model-registry/{artifact_name}"
    artifact.link(target_path=target_path, aliases=[alias])
    artifact.save()

    typer.echo(f"Linked {artifact_path} -> {target_path} with alias '{alias}'")


if __name__ == "__main__":
    typer.run(link_model)
