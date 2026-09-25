"""A complete normal-player episode exports usable post-training rows."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_export_splits_complete_games_and_preserves_player_requests(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "dataset"
    subprocess.run(
        [sys.executable, str(root / "tools/export_posttrain.py"), str(output), "--episodes", "10", "--turns-max", "3"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    splits = {
        split: [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        for split in ("train", "validation")
    }
    assert manifest["source_revision"]
    assert len(manifest["runs"]) == 10
    assert len(splits["train"]) == manifest["train_examples"]
    assert len(splits["validation"]) == manifest["validation_examples"]
    assert all(splits.values())
    assert not {row["seed"] for row in splits["train"]} & {row["seed"] for row in splits["validation"]}
    assert {json.loads(row["prompt"][1]["content"])["request"]["type"] for rows in splits.values() for row in rows} == {
        "introduce_request", "action_request", "proposal_request", "debate_request", "vote_request"
    }
    assert all(
        json.loads(row["completion"][0]["content"])["rid"] == row["decision_id"]
        for rows in splits.values() for row in rows
    )
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in output.iterdir())
