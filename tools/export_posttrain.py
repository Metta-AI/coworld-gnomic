"""Export complete deterministic Gnomic games through the normal player SDK."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from gnomic.players.posttrain_prompt import messages_for
from gnomic.players.scribe import ScribePolicy
from gnomic.players.sdk import GameView, InProcessTransport, PlayerSession
from gnomic.server.channel import InProcessChannel
from gnomic.server.config import GameConfig
from gnomic.server.episode import Episode

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = {"introduce_request", "action_request", "action_repair_request", "proposal_request", "debate_request", "vote_request"}


class RecordingTransport(InProcessTransport):
    def __init__(self, channel: InProcessChannel, view: GameView) -> None:
        super().__init__(channel)
        self.view = view
        self.request: dict[str, Any] | None = None
        self.rows: list[dict[str, Any]] = []

    async def recv(self) -> dict:
        message = await super().recv()
        if message["type"] in DECISIONS:
            self.request = message
        return message

    async def send(self, message: dict) -> None:
        assert self.request is not None and message["rid"] == self.request["rid"]
        self.rows.append(
            {
                "rid": message["rid"],
                "seat": self.view.seat,
                "request": self.request,
                "prompt": messages_for(self.view),
                "reply": message,
            }
        )
        await super().send(message)
        self.request = None


async def collect(seed: int, config_values: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    channels = [InProcessChannel(seat) for seat in range(3)]
    sessions: list[PlayerSession] = []
    recorders: list[RecordingTransport] = []
    for channel in channels:
        view = GameView()
        recorder = RecordingTransport(channel, view)
        session = PlayerSession(ScribePolicy(), recorder)
        session.view = view
        sessions.append(session)
        recorders.append(recorder)
    tasks = [asyncio.create_task(session.run()) for session in sessions]
    config = GameConfig.model_validate({**config_values, "tokens": ["a", "b", "c"], "seed": seed})
    results, replay = await Episode(config, channels, seed=seed).run()
    for channel in channels:
        await channel.send({"type": "final", "scores": results["scores"]})
    await asyncio.gather(*tasks)
    assert all(not session.defaults for session in sessions)
    assert all(not event["action"]["default"] for event in replay["events"] if event["type"] == "action_made")
    rows = sorted((row for recorder in recorders for row in recorder.rows), key=lambda row: row["rid"])
    assert rows and [row["rid"] for row in rows] == list(range(1, len(rows) + 1))
    return rows, results


def write_private(path: Path, content: str) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as output:
        output.write(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--turns-max", type=int, default=45)
    args = parser.parse_args()
    if args.episodes < 10 or args.first_seed < 1 or not 1 <= args.turns_max <= 45:
        raise ValueError("Require at least ten episodes, a positive first seed, and 1-45 turns")
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = json.loads((ROOT / "coworld_manifest_template.json").read_text())
    config_values = {
        **manifest["certification"]["game_config"],
        "turns_max": args.turns_max,
        "players": [{"name": f"scribe-{seat}"} for seat in range(3)],
    }
    output_rows: dict[str, list[str]] = {"train": [], "validation": []}
    runs = []
    for seed in range(args.first_seed, args.first_seed + args.episodes):
        decisions, result = asyncio.run(collect(seed, config_values))
        split = "validation" if seed % 5 == 0 else "train"
        episode_id = f"gnomic-scribe-{seed}"
        for row in decisions:
            example = {
                "episode_id": episode_id,
                "seed": episode_id,
                "decision_id": row["rid"],
                "prompt": row["prompt"],
                "completion": [{"role": "assistant", "content": json.dumps(row["reply"])}],
                "game": "gnomic",
                "action_schema_revision": "gnomic-player-v1",
            }
            output_rows[split].append(json.dumps(example, ensure_ascii=False))
        runs.append({"seed": seed, "decisions": len(decisions), "scores": result["scores"], "turns": result["turns_played"]})
    if not all(output_rows.values()):
        raise ValueError("Both training and validation splits need complete episodes")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for split, rows in output_rows.items():
        write_private(args.output / f"{split}.jsonl", "\n".join(rows) + "\n")
    write_private(
        args.output / "manifest.json",
        json.dumps(
            {
                "schema_version": 1,
                "game": "gnomic",
                "action_schema_revision": "gnomic-player-v1",
                "source_revision": source_revision,
                "teacher": "scribe",
                "judge": "deterministic",
                "turns_max": args.turns_max,
                "train_examples": len(output_rows["train"]),
                "validation_examples": len(output_rows["validation"]),
                "runs": runs,
            },
            indent=2,
        ) + "\n",
    )
    print(f"train={len(output_rows['train'])} validation={len(output_rows['validation'])}")


if __name__ == "__main__":
    main()
