"""Export whole authoritative Gnomic episodes with the ordinary Scribe player SDK."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path

from gnomic.players.scribe import ScribePolicy
from gnomic.players.sdk import InProcessTransport, PlayerSession
from gnomic.server.channel import InProcessChannel
from gnomic.server.config import GameConfig
from gnomic.server.episode import Episode
from gnomic.training import TrainingEpisode, write_private_episode

ROOT = Path(__file__).resolve().parents[1]


async def collect(seed: int, config_values: dict) -> TrainingEpisode:
    channels = [InProcessChannel(seat) for seat in range(3)]
    sessions = [
        PlayerSession(ScribePolicy(), InProcessTransport(channel))
        for channel in channels
    ]
    tasks = [asyncio.create_task(session.run()) for session in sessions]
    config = GameConfig.model_validate(
        {**config_values, "tokens": ["a", "b", "c"], "seed": seed}
    )
    episode = Episode(config, channels, seed=seed, teacher_seats=frozenset(range(3)))
    results, replay = await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": results["scores"]})
    await asyncio.gather(*tasks)
    assert all(not session.defaults for session in sessions)
    assert all(
        not event["action"]["default"]
        for event in replay["events"]
        if event["type"] == "action_made"
    )
    return episode.capture.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--episodes", type=int, default=10, help="Whole games per manifest variant"
    )
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument(
        "--turns-max",
        type=int,
        help="Explicit diagnostic cap; omit for the declared whole variant",
    )
    parser.add_argument("--judge-mode", choices=("native", "deterministic"))
    args = parser.parse_args()
    if (
        args.episodes < 10
        or args.first_seed < 0
        or (args.turns_max is not None and not 1 <= args.turns_max <= 45)
    ):
        raise ValueError(
            "Require ten games per variant, nonnegative seeds and a valid explicit cap"
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if dirty:
        raise ValueError("Teacher export requires clean committed source")
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    os.environ["COWORLD_SOURCE_REVISION"] = source
    os.environ["COWORLD_GAME_VERSION"] = (
        f"source-{source}-{args.judge_mode or 'declared'}"
        + (f"-cap{args.turns_max}" if args.turns_max is not None else "")
    )
    manifest = json.loads((ROOT / "coworld_manifest_template.json").read_text())
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    runs = []
    for variant in manifest["variants"]:
        config = {
            **variant["game_config"],
            "judge_mode": args.judge_mode or variant["game_config"]["judge_mode"],
            "players": [{"name": "scripted-scribe"} for _ in range(3)],
        }
        if args.turns_max is not None:
            config["turns_max"] = args.turns_max
        for seed in range(args.first_seed, args.first_seed + args.episodes):
            os.environ["COWORLD_EPISODE_ID"] = (
                f"gnomic-{variant['id']}-scribe-{seed}-{source[:12]}"
            )
            episode = asyncio.run(collect(seed, config))
            path = args.output / f"{variant['id']}-{seed}.jsonl"
            write_private_episode(path.as_uri(), episode)
            runs.append(
                {
                    "variant": variant["id"],
                    "seed": seed,
                    "seed_family": episode.episode.seed_family,
                    "path": path.name,
                    "status": episode.episode.status,
                    "decisions": len(episode.decisions),
                }
            )
    summary = {
        "source_revision": source,
        "game_version": os.environ["COWORLD_GAME_VERSION"],
        "teacher": "scripted-scribe",
        "judge": args.judge_mode,
        "runs": runs,
        "review_status": "unreviewed",
        "qualification": "Run Coworld SDK qualifier and Metta hosted importer against these private episodes",
    }
    with os.fdopen(
        os.open(
            args.output / "manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        ),
        "w",
    ) as output:
        output.write(json.dumps(summary, indent=2) + "\n")
    print(f"whole_games={len(runs)} decisions={sum(r['decisions'] for r in runs)}")


if __name__ == "__main__":
    main()
