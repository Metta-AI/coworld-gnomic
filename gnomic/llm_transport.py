"""Native Messages transport and private evidence for each real started call."""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Attempt(BaseModel):
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, allow_inf_nan=False
    )

    attempt_id: str = Field(default_factory=lambda: str(uuid4()))
    platform_call_id: UUID | None = None
    policy: str
    origin: Literal["model", "teacher", "fallback", "human", "unknown"] = "model"
    inference_mode: Literal["text_action"] | None = None
    prompt: JsonValue
    request: JsonValue | None
    raw_response: str | None = None
    response_headers: dict[str, str] | None = None
    provider_request_id: str | None = None
    response: str | None = None
    model: str | None
    model_identity: str | None = None
    tokenizer_identity: str | None = None
    chat_template_sha256: str | None = None
    decoder: JsonValue | None
    prompt_token_ids: list[int] | None = None
    sampled_token_ids: list[int] | None = None
    behavior_logprobs: list[float] | None = None
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    parsed_action: JsonValue | None = None
    accepted: bool = False
    rejection_reason: str | None = (
        "generation did not produce an applied learner action"
    )


class ContentBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    text: str = ""


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class Sampling(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    behavior_log_probs: list[float] | None = None
    stop_reason: str | None = None

    @model_validator(mode="after")
    def actual_sampling(self) -> Sampling:
        for tokens in (self.prompt_token_ids, self.completion_token_ids):
            if tokens is not None and any(token < 0 for token in tokens):
                raise ValueError("Token identities must be nonnegative")
        if self.behavior_log_probs is not None:
            if self.completion_token_ids is None or len(
                self.completion_token_ids
            ) != len(self.behavior_log_probs):
                raise ValueError(
                    "Behavior probabilities require matching actual tokens"
                )
            if any(probability > 0 for probability in self.behavior_log_probs):
                raise ValueError("Log probabilities cannot be positive")
        return self


class Response(BaseModel):
    model_config = ConfigDict(extra="allow")
    content: list[ContentBlock]
    model: str
    usage: Usage
    stop_reason: str | None = None
    sampling_evidence: Sampling | None = None

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if block.type == "text")


class Decoder(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)


@dataclass
class LearnerWindow:
    seat: int
    policy: str
    deadline: float
    publish: Callable[[Attempt], None]
    attempts: list[Attempt] = field(default_factory=list)


current_window: ContextVar[LearnerWindow] = ContextVar("gnomic_learner_window")


class AttemptPublisher:
    """Bind thread progress to its original authenticated request and transport."""

    def __init__(
        self, rid: int, send: Callable[[dict], Coroutine[Any, Any, None]]
    ) -> None:
        self.rid = rid
        self.send = send
        self.loop = asyncio.get_running_loop()
        self.pending: set[asyncio.Task] = set()

    def __call__(self, attempt: Attempt) -> None:
        packet = {
            "type": "private_attempt",
            "rid": self.rid,
            "attempt": attempt.model_dump(mode="json"),
        }

        def schedule() -> None:
            task = self.loop.create_task(self.send(packet))
            self.pending.add(task)

        self.loop.call_soon_threadsafe(schedule)

    async def drain(self) -> None:
        await asyncio.sleep(0)
        while self.pending:
            pending = tuple(self.pending)
            await asyncio.gather(*pending)
            self.pending.difference_update(pending)


async def complete_native(
    body: dict,
    model: str,
    *,
    purpose: Literal["learner", "environment"],
    timeout: float,
    slot: int | None = None,
    generations: list[Attempt] | None = None,
) -> Response:
    window = current_window.get(None) if purpose == "learner" else None
    if purpose == "learner":
        model = os.environ.get("COWORLD_LLM_MODEL", model)
        slot = window.seat if window is not None else slot
        if slot is None or slot < 0:
            raise ValueError("Native learner requires the authenticated player slot")
        if window is not None:
            timeout = min(timeout, window.deadline - time.monotonic())
    if not math.isfinite(timeout) or timeout <= 0:
        raise TimeoutError("Native generation has no remaining decision time")
    temperature = (
        float(
            os.environ.get("COWORLD_LLM_TEMPERATURE", str(body.get("temperature", 0)))
        )
        if purpose == "learner"
        else body["temperature"]
    )
    top_p = (
        float(os.environ.get("COWORLD_LLM_TOP_P", "1"))
        if purpose == "learner"
        else body["top_p"]
    )
    decoder = Decoder(
        max_tokens=body["max_tokens"], temperature=temperature, top_p=top_p
    )
    payload = {**body, "model": model, **decoder.model_dump()}
    attempt = Attempt(
        policy=window.policy
        if window is not None
        else "frozen-elder"
        if purpose == "environment"
        else "native-learner",
        inference_mode="text_action" if purpose == "learner" else None,
        prompt=[{"role": "system", "content": body["system"]}, *body["messages"]],
        request=payload,
        model=model,
        decoder={
            **{
                key: value
                for key, value in payload.items()
                if key not in {"model", "system", "messages"}
            },
            "timeout_ms": timeout * 1000,
        },
    )
    if generations is not None:
        generations.append(attempt)
    if window is not None:
        window.attempts.append(attempt)
        window.publish(attempt)
    headers = {"anthropic-version": "2023-06-01"}
    if purpose == "learner":
        headers["X-Coworld-Player-Slot"] = str(slot)
    started = time.monotonic()
    try:
        async with (
            asyncio.timeout(timeout),
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream(
                "POST",
                os.environ["COWORLD_LLM_ENDPOINT"].rstrip("/") + "/v1/messages",
                json=payload,
                headers=headers,
            ) as response,
        ):
            captured_headers = dict(response.headers)
            attempt.response_headers = captured_headers
            if "x-softmax-llm-call-id" in captured_headers:
                attempt.platform_call_id = UUID(
                    captured_headers["x-softmax-llm-call-id"]
                )
            attempt.provider_request_id = captured_headers.get("request-id")
            attempt.model_identity = captured_headers.get("x-coworld-checkpoint-sha256")
            attempt.tokenizer_identity = captured_headers.get(
                "x-coworld-tokenizer-sha256"
            )
            attempt.chat_template_sha256 = captured_headers.get(
                "x-coworld-chat-template-sha256"
            )
            if window is not None:
                window.publish(attempt)
            attempt.raw_response = (await response.aread()).decode()
            response.raise_for_status()
            parsed = Response.model_validate_json(attempt.raw_response)
            attempt.model = parsed.model
            attempt.response = "".join(
                block.text for block in parsed.content if block.type == "text"
            )
            attempt.stop_reason = parsed.stop_reason
            attempt.input_tokens = parsed.usage.input_tokens
            attempt.output_tokens = parsed.usage.output_tokens
            if parsed.sampling_evidence is not None:
                attempt.prompt_token_ids = parsed.sampling_evidence.prompt_token_ids
                attempt.sampled_token_ids = (
                    parsed.sampling_evidence.completion_token_ids
                )
                attempt.behavior_logprobs = parsed.sampling_evidence.behavior_log_probs
                attempt.stop_reason = parsed.sampling_evidence.stop_reason
            return parsed
    finally:
        attempt.latency_ms = (time.monotonic() - started) * 1000
        if window is not None:
            window.publish(attempt)
