"""Whole-game wording checks through the real engine and three Scribe sessions.

The standard variant uses an explicitly diagnostic deterministic judge here.
This does not establish production Elder parity or model performance.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from gnomic.lifecycle import main_owned
from gnomic.prompts import BANK, PromptSelection
from tools.export_posttrain import collect


def main() -> None:
    manifest = json.loads(Path("coworld_manifest_template.json").read_text())
    wordings = (
        "canonical",
        "plain",
        "numbered",
        "questions",
        "reordered",
        "heldout-briefing",
        "heldout-reference",
        "mixed",
    )
    canonical = {}
    systems = {}
    games = 0
    for variant in manifest["variants"]:
        for seed in (7, 19):
            for wording in wordings:
                selection = PromptSelection(variant=wording, seed=4)
                config = {
                    **variant["game_config"],
                    "judge_mode": "deterministic",
                    "players": [{"name": "scripted-scribe"} for _ in range(3)],
                    "prompt_selection": selection.model_dump(),
                }
                os.environ.update(
                    COWORLD_SOURCE_REVISION="uncommitted-gnomic-validation",
                    COWORLD_GAME_VERSION="diagnostic-deterministic-prompts",
                    COWORLD_EPISODE_ID=f"prompt-{variant['id']}-{seed}-{wording}",
                )
                records = []
                episode = main_owned(collect(seed, config, records.append))
                assert episode is not None and len(records) == 1
                assert episode.episode.status == "completed"
                assert episode.episode.seed_family == f"gnomic-{seed}"
                assert (
                    episode.episode.outcome["environment_policy"]["mode"]
                    == "deterministic"
                )
                assert episode.episode.outcome["ownership_joined"]
                trace = []
                for decision in episode.decisions:
                    assert decision.action_status == "accepted"
                    metadata = decision.observation["prompt_template"]
                    assert (
                        metadata["bank"],
                        metadata["variant"],
                        metadata["seed"],
                    ) == (BANK, selection.selected, 4)
                    assert metadata["messages_sha256"] == [
                        hashlib.sha256(message["content"].encode()).hexdigest()
                        for message in decision.prompt
                    ]
                    (attempt,) = decision.attempts
                    assert attempt.origin == (
                        "unknown"
                        if decision.observation["view"]["request"]["type"]
                        == "introduce_request"
                        else "teacher"
                    )
                    assert attempt.accepted
                    assert (
                        attempt.prompt == decision.prompt
                        and attempt.parsed_action == decision.executed_action
                    )
                    assert "prompt_selection" not in decision.prompt[1]["content"]
                    assert "prompt_template" not in decision.prompt[1]["content"]
                    systems.setdefault(wording, set()).add(
                        decision.prompt[0]["content"]
                    )
                    trace.append(
                        (
                            decision.seat,
                            decision.observation["view"],
                            decision.prompt[1],
                            decision.executed_action,
                            decision.reward,
                            decision.terminal,
                        )
                    )
                key = variant["id"], seed
                actual = (
                    trace,
                    episode.episode.outcome["results"],
                    episode.episode.participant_outcomes,
                )
                if wording == "canonical":
                    canonical[key] = actual
                else:
                    assert actual == canonical[key]
                games += 1
                print(
                    variant["id"],
                    seed,
                    wording,
                    len(trace),
                    "decisions PASS",
                    flush=True,
                )
    assert all(len(values) == 1 for values in systems.values())
    assert len({next(iter(values)) for values in systems.values()}) == 7
    print(
        games,
        "whole games; invariant actions, observations, results, families and private hashes PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
