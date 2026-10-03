"""Run the standard variant through an explicitly configured local native sidecar.

The Elder keeps its manifest model. Learner model overrides apply only to players.
No AWS credentials or provider fallback are installed into game or player containers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from coworld.certifier import build_manifest_episode_job_spec, load_coworld_package
from coworld.runner.runner import EpisodeArtifacts, run_coworld_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("coworld_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-opus"))
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("COWORLD_LLM_ENDPOINT"),
        required="COWORLD_LLM_ENDPOINT" not in os.environ,
    )
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

    native_env = {"COWORLD_LLM_ENDPOINT": args.endpoint}
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
        update={"env": {**manifest.game.runnable.env, **native_env}},
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
        secret_env={
            **native_env,
            **{
                key: os.environ[key]
                for key in (
                    "COWORLD_LLM_MODEL",
                    "COWORLD_LLM_TEMPERATURE",
                    "COWORLD_LLM_TOP_P",
                )
                if key in os.environ
            },
        },
    )
    print(f"Results: {artifacts.results_path}")
    print(f"Replay: {artifacts.replay_path}")
    print(f"Logs: {artifacts.logs_dir}")


if __name__ == "__main__":
    main()
