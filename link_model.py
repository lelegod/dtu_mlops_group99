# link_model.py
import argparse
import os
import wandb


def parse_target_path(full_name: str) -> str:
    """
    Accepts either:
      - wandb-registry-REGISTRY/COLLECTION:v0
      - ENTITY/wandb-registry-REGISTRY/COLLECTION:v0

    Returns:
      - wandb-registry-REGISTRY/COLLECTION
    """
    # Drop entity prefix if present
    if "/wandb-registry-" in full_name:
        full_name = full_name.split("/wandb-registry-", 1)[1]
        full_name = "wandb-registry-" + full_name

    # Remove :v0 / :latest / etc
    return full_name.split(":", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Full W&B artifact name, e.g. entity/wandb-registry-REG/COLLECTION:v0")
    parser.add_argument("--alias", default="production", help="Alias to apply (default: production)")
    args = parser.parse_args()

    api_key = os.getenv("WANDB_API_KEY")
    project = os.getenv("WANDB_PROJECT")

    if not api_key:
        raise RuntimeError("WANDB_API_KEY not set")
    if not project:
        raise RuntimeError("WANDB_PROJECT not set")

    api = wandb.Api(api_key=api_key)

    # Get the staged artifact version (MODEL_NAME points to it)
    artifact = api.artifact(args.model_name)

    # Link to the same registry/collection, but with a new alias (production)
    target_path = parse_target_path(args.model_name)
    artifact.link(target_path=target_path, aliases=[args.alias])

    print(f"Linked {args.model_name} to {target_path} with alias '{args.alias}'")


if __name__ == "__main__":
    main()
