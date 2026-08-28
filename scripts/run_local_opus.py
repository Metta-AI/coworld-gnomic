"""Run the standard Gnomic variant with Bedrock credentials in game and players.

The stock local Coworld runner intentionally grants ``--use-bedrock`` only to
player containers. Gnomic's Judge is part of the game container, so this small
developer harness injects the same temporary AWS session environment into both.
Credentials remain in process memory and Docker environment; they are never
written to the episode artifacts or printed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from coworld.certifier import build_manifest_episode_job_spec, load_coworld_package
from coworld.runner.runner import EpisodeArtifacts, run_coworld_episode


def bedrock_env(profile: str, region: str) -> dict[str, str]:
    result = subprocess.run(
        ["aws", "configure", "export-credentials", "--format", "process", "--profile", profile],
        check=True,
        capture_output=True,
        text=True,
    )
    exported = json.loads(result.stdout)
    env = {
        "AWS_ACCESS_KEY_ID": exported["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": exported["SecretAccessKey"],
        "AWS_REGION": region,
        "AWS_DEFAULT_REGION": region,
        "BEDROCK_MODEL": "us.anthropic.claude-opus-4-8",
        "USE_BEDROCK": "true",
    }
    if exported.get("SessionToken"):
        env["AWS_SESSION_TOKEN"] = exported["SessionToken"]
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("coworld_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-opus"))
    parser.add_argument("--profile", default="softmax")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Optional local smoke cap; the standard manifest remains 45 turns.",
    )
    parser.add_argument(
        "images",
        nargs="*",
        default=[
            "coworld-gnomic-ivan:latest",
            "coworld-gnomic-anton:latest",
            "coworld-gnomic-yura:latest",
        ],
    )
    args = parser.parse_args()
    if len(args.images) != 3:
        parser.error("provide either no image arguments or exactly three")
    if args.turns is not None and not 1 <= args.turns <= 45:
        parser.error("--turns must be between 1 and 45")

    credentials = bedrock_env(args.profile, args.region)
    package = load_coworld_package(args.manifest)
    job = build_manifest_episode_job_spec(
        package,
        variant_id="standard-3-opus",
        player_images=args.images,
        player_run=["python", "-m", "gnomic.players.llm"],
    )
    if args.turns is not None:
        job = job.model_copy(
            deep=True,
            update={"game_config": {**job.game_config, "turns_max": args.turns}},
        )
    manifest = job.manifest.model_copy(deep=True)
    runnable = manifest.game.runnable.model_copy(
        deep=True,
        update={"env": {**manifest.game.runnable.env, **credentials}},
    )
    manifest.game = manifest.game.model_copy(deep=True, update={"runnable": runnable})
    job = job.model_copy(deep=True, update={"manifest": manifest})

    artifacts = EpisodeArtifacts.create(args.output.resolve(), prefix="gnomic-opus-")
    run_coworld_episode(
        job,
        artifacts,
        timeout_seconds=6_600,
        verify_replay=True,
        container_prefix="gnomic-opus",
        secret_env=credentials,
    )
    print(f"Results: {artifacts.results_path}")
    print(f"Replay: {artifacts.replay_path}")
    print(f"Logs: {artifacts.logs_dir}")


if __name__ == "__main__":
    main()
