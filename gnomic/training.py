"""Private engine-owned decisions; public replay never reads this journal."""

from __future__ import annotations

import base64
import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .llm_transport import Attempt
from .players.posttrain_prompt import messages_for
from .players.sdk import GameView
from .protocol import parse_reply_text


class RequestWindow(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, extra="forbid")
    request: dict[str, JsonValue]
    observation: JsonValue
    prompt: list[dict[str, str]]
    deadline: float
    attempts: dict[str, Attempt] = Field(default_factory=dict)
    submitted: JsonValue = None
    executed: JsonValue = None
    fallback: bool = True
    closed: bool = False


class Decision(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, extra="forbid")
    schema_version: Literal["1"] = "1"
    event_type: Literal["decision"] = "decision"
    episode_id: str
    decision_id: str
    decision_index: int
    game: Literal["gnomic"] = "gnomic"
    game_version: str
    source_revision: str
    image_digest: str | None = None
    seat: str
    visibility: Literal["private"] = "private"
    observation: JsonValue
    prompt: JsonValue
    attempts: list[Attempt]
    selected_attempt_id: str | None
    executed_action: JsonValue
    action_status: Literal["accepted", "rejected", "fallback"]
    fallback_origin: str | None
    reward: float | None = None
    terminal: bool = False


class EpisodeRecord(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, extra="forbid")
    schema_version: Literal["1"] = "1"
    event_type: Literal["episode"] = "episode"
    episode_id: str
    seed_family: str
    game: Literal["gnomic"] = "gnomic"
    game_version: str
    source_revision: str
    image_digest: str | None = None
    status: Literal["completed", "failed", "truncated"]
    outcome: JsonValue
    participant_outcomes: JsonValue


class TrainingEpisode(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, extra="forbid")
    schema_version: Literal["1"] = "1"
    episode: EpisodeRecord
    decisions: list[Decision]


class Capture:
    def __init__(self, episode, *, teacher_seats: frozenset[int]) -> None:
        self.episode = episode
        self.teacher_seats = teacher_seats
        self.views = {seat: GameView(seat=seat) for seat in range(3)}
        self.windows: dict[tuple[int, int], RequestWindow] = {}
        self.sealed = False
        self.ingress_issues: list[str] = []

    def delivered(self, seat: int, message: dict) -> None:
        if self.sealed:
            raise RuntimeError("decision delivery after private seal")
        self.views[seat].fold(copy.deepcopy(message))
        if "rid" in message:
            key = (seat, message["rid"])
            if key not in self.windows:
                prompt = messages_for(self.views[seat])
                self.windows[key] = RequestWindow(
                    request=copy.deepcopy(message),
                    observation=json.loads(prompt[1]["content"]),
                    prompt=prompt,
                    deadline=time.monotonic() + message["timeout_s"],
                )

    def receive(self, seat: int, packet: dict) -> dict | None:
        if packet.get("type") != "private_attempt":
            return packet
        if self.sealed:
            return None
        key = (seat, packet["rid"])
        if key not in self.windows:
            self.ingress_issues.append("unknown-request")
            return None
        window = self.windows[key]
        attempt = Attempt.model_validate(packet["attempt"])
        attempt.accepted = False
        attempt.parsed_action = None
        if attempt.origin != "model":
            attempt.origin = "unknown"
        if attempt.attempt_id in window.attempts:
            started = window.attempts[attempt.attempt_id]
            if any(
                getattr(started, name) != getattr(attempt, name)
                for name in ("prompt", "request", "decoder", "policy")
            ):
                self.ingress_issues.append("started-request-mutated")
                return None
            if started.response_body_b64 is not None and (
                attempt.response_body_b64 is None
                or not base64.b64decode(
                    attempt.response_body_b64, validate=True
                ).startswith(base64.b64decode(started.response_body_b64, validate=True))
            ):
                self.ingress_issues.append("received-body-mutated")
                return None
            if started.latency_ms is not None:
                self.ingress_issues.append("completed-attempt-mutated")
                return None
        elif window.closed or time.monotonic() > window.deadline:
            self.ingress_issues.append("out-of-window-attempt")
            return None
        window.attempts[attempt.attempt_id] = attempt
        return None

    def consumed(
        self, seat: int, rid: int, raw: dict | None, action: dict, fallback: bool
    ) -> None:
        if self.sealed:
            raise RuntimeError("engine action after private seal")
        window = self.windows[(seat, rid)]
        window.submitted = copy.deepcopy(raw)
        window.executed = copy.deepcopy(action)
        window.fallback = fallback
        window.closed = True
        if (
            seat in self.teacher_seats
            and not fallback
            and window.request["type"] != "introduce_request"
        ):
            response = json.dumps({"rid": rid, **action}, ensure_ascii=False)
            attempt = Attempt(
                policy="scripted-scribe",
                origin="teacher",
                inference_mode="text_action",
                prompt=window.prompt,
                request=None,
                response=response,
                model=None,
                decoder=None,
                parsed_action=action,
                accepted=True,
                rejection_reason=None,
            )
            window.attempts[attempt.attempt_id] = attempt

    def finish(
        self,
        *,
        status: Literal["completed", "failed", "truncated"] = "completed",
        failure_kind: str | None = None,
        ownership_joined: bool | None = None,
    ) -> TrainingEpisode:
        from .server.episode import Episode

        self.sealed = True
        ep = self.episode
        decisions = []
        incomplete = False
        for (seat, rid), window in self.windows.items():
            attempts = [a.model_copy(deep=True) for a in window.attempts.values()]
            selected = None
            for attempt in attempts:
                incomplete |= attempt.origin == "model" and (
                    attempt.latency_ms is None
                    or attempt.response_complete is False
                    or attempt.response_reader_joined is False
                )
                if attempt.origin == "teacher":
                    selected = attempt
                    continue
                if attempt.origin == "model" and (
                    attempt.raw_response is None
                    or attempt.platform_call_id is None
                    or attempt.response_complete is not True
                    or attempt.response_reader_joined is not True
                ):
                    attempt.rejection_reason = "missing-native-response-provenance"
                    continue
                if attempt.response is None or attempt.prompt != window.prompt:
                    attempt.rejection_reason = (
                        "generation-failed-or-noncanonical-opponent-prompt"
                    )
                    continue
                parsed = parse_reply_text(attempt.response)
                if parsed.kind == "invalid":
                    attempt.rejection_reason = parsed.error
                    continue
                decoded = parsed.value
                assert decoded is not None
                if decoded.get("rid") != rid:
                    attempt.rejection_reason = "response-request-mismatch"
                    continue
                attempt.parsed_action = Episode.canonical_action(
                    window.request, seat, decoded
                )
                if attempt.parsed_action == window.executed and not window.fallback:
                    selected = attempt
                else:
                    attempt.rejection_reason = "parser-execution-mismatch"
            if selected is not None and not window.fallback:
                selected.accepted = True
                selected.rejection_reason = None
            if not attempts:
                attempt = Attempt(
                    policy=ep.config.players[seat].name,
                    origin="fallback" if window.fallback else "unknown",
                    prompt=window.prompt,
                    request=None,
                    model=None,
                    decoder=None,
                    parsed_action=window.executed,
                    accepted=not window.fallback,
                )
                attempts.append(attempt)
                selected = attempt
            action_status = (
                "fallback"
                if window.fallback
                else "accepted"
                if selected is not None
                else "rejected"
            )
            decisions.append(
                Decision(
                    episode_id=os.environ["COWORLD_EPISODE_ID"],
                    decision_id=str(rid),
                    decision_index=len(decisions),
                    game_version=os.environ["COWORLD_GAME_VERSION"],
                    source_revision=os.environ["COWORLD_SOURCE_REVISION"],
                    image_digest=os.environ.get("COWORLD_GAME_IMAGE_DIGEST"),
                    seat=str(seat),
                    observation={
                        "view": window.observation,
                        "submitted": window.submitted,
                    },
                    prompt=selected.prompt if selected is not None else window.prompt,
                    attempts=attempts,
                    selected_attempt_id=selected.attempt_id
                    if selected is not None
                    else None,
                    executed_action=window.executed,
                    action_status=action_status,
                    fallback_origin="engine-default" if window.fallback else None,
                )
            )
        scores = ep.results["scores"] if ep.results is not None else [None] * 3
        completion = "truncated" if status == "completed" and incomplete else status
        for seat in range(3):
            rows = [d for d in decisions if d.seat == str(seat)]
            if rows:
                rows[-1].reward = scores[seat]
                rows[-1].terminal = completion == "completed"
        generations = ep.judge.generations if hasattr(ep.judge, "generations") else []
        return TrainingEpisode(
            episode=EpisodeRecord(
                episode_id=os.environ["COWORLD_EPISODE_ID"],
                seed_family=f"gnomic-{ep.seed}",
                game_version=os.environ["COWORLD_GAME_VERSION"],
                source_revision=os.environ["COWORLD_SOURCE_REVISION"],
                image_digest=os.environ.get("COWORLD_GAME_IMAGE_DIGEST"),
                status=completion,
                participant_outcomes={str(s): {"score": scores[s]} for s in range(3)},
                outcome={
                    "results": ep.results,
                    "config": ep.config.model_dump(exclude={"tokens"}),
                    "failure_kind": failure_kind,
                    "ownership_joined": ownership_joined,
                    "ingress_issues": self.ingress_issues,
                    "received_header_pairs": [
                        {"seat": c.seat, "records": c.received_header_records()}
                        for c in ep.channels
                    ],
                    "environment_policy": {
                        "mode": ep.config.judge_mode,
                        "model": ep.config.judge_model,
                        "max_tokens": ep.judge.max_tokens
                        if ep.config.judge_mode == "native"
                        else None,
                        "thinking": {"type": "adaptive"}
                        if ep.config.judge_mode == "native"
                        else None,
                        "output_config": {"effort": "high"}
                        if ep.config.judge_mode == "native"
                        else None,
                        "system": ep.judge_systems,
                        "temperature": 1,
                        "top_p": 1,
                    },
                    "environment_generations": [
                        a.model_dump(mode="json") for a in generations
                    ],
                },
            ),
            decisions=decisions,
        )


def validate_capture_environment() -> None:
    for key in (
        "COWORLD_EPISODE_ID",
        "COWORLD_GAME_VERSION",
        "COWORLD_SOURCE_REVISION",
    ):
        if not os.environ[key]:
            raise ValueError(f"Private capture requires {key}")
    if (
        re.fullmatch(
            r"[a-f0-9]{40}|(?:sha256:)?[a-f0-9]{64}",
            os.environ["COWORLD_SOURCE_REVISION"],
        )
        is None
    ):
        raise ValueError("Private capture requires an immutable source revision")


def write_private_episode(uri: str, episode: TrainingEpisode) -> None:
    target = urlsplit(uri)
    if (
        target.scheme != "file"
        or target.netloc
        or target.query
        or target.fragment
        or not Path(target.path).is_absolute()
    ):
        raise ValueError("Private capture requires an absolute local file URI")
    path = Path(target.path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
    with os.fdopen(
        os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
    ) as output:
        output.write(episode.model_dump_json() + "\n")
    os.link(temporary, path)
    temporary.unlink()
