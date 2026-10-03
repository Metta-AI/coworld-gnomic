"""A complete normal-player episode exports usable post-training rows."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_export_splits_complete_games_and_preserves_player_requests(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "dataset"
    subprocess.run(
        [
            sys.executable,
            str(root / "tools/export_posttrain.py"),
            str(output),
            "--episodes",
            "10",
            "--turns-max",
            "3",
            "--judge-mode",
            "deterministic",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    episodes = [
        json.loads((output / run["path"]).read_text()) for run in manifest["runs"]
    ]
    assert len(episodes) == 20
    assert {run["variant"] for run in manifest["runs"]} == {
        "standard-3-opus",
        "qualifier-1-turn",
    }
    assert len({episode["episode"]["seed_family"] for episode in episodes}) == 10
    assert all(episode["episode"]["status"] == "completed" for episode in episodes)
    decisions = [row for episode in episodes for row in episode["decisions"]]
    assert {row["observation"]["view"]["request"]["type"] for row in decisions} == {
        "introduce_request",
        "action_request",
        "proposal_request",
        "debate_request",
        "vote_request",
    }
    teacher = [
        attempt
        for row in decisions
        for attempt in row["attempts"]
        if attempt["origin"] == "teacher"
    ]
    assert teacher
    assert all(attempt["accepted"] and attempt["response"] for attempt in teacher)
    assert all(
        attempt["parsed_action"] == row["executed_action"]
        for row in decisions
        for attempt in row["attempts"]
        if attempt["origin"] == "teacher"
    )
    assert all(
        row["source_revision"] == manifest["source_revision"] for row in decisions
    )
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())
