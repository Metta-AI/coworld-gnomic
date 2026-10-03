"""Native Messages transport and private evidence for each real started call."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import ssl
import sys
import time
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    model_validator,
)

from gnomic.lifecycle import OwnershipUnsettled, owned_task, settle, shutdown_deadline


class ReceivedHeaderPairs(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, extra="forbid")
    attempt_id: str
    pairs: list[tuple[str, str]]


class Attempt(BaseModel):
    model_config = ConfigDict(
        hide_input_in_errors=True,
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
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
    response_body_b64: str | None = None
    response_headers_b64: str | None = None
    response_complete: Annotated[bool, Field(strict=True)] | None = None
    response_reader_joined: Annotated[bool, Field(strict=True)] | None = None
    http_status: Annotated[int, Field(strict=True, ge=100, le=599)] | None = None
    _received_header_pairs: ReceivedHeaderPairs | None = PrivateAttr(default=None)
    provider_request_id: str | None = None
    response: str | None = None
    model: str | None
    model_identity: str | None = None
    tokenizer_identity: str | None = None
    chat_template_sha256: str | None = None
    decoder: JsonValue | None
    prompt_token_ids: list[Annotated[int, Field(strict=True, ge=0)]] | None = None
    sampled_token_ids: list[Annotated[int, Field(strict=True, ge=0)]] | None = None
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

    @model_validator(mode="after")
    def received_bytes_match(self) -> Attempt:
        if self.response_body_b64 is not None:
            received = base64.b64decode(self.response_body_b64, validate=True)
            if self.raw_response is not None and received != self.raw_response.encode(
                "utf-8"
            ):
                raise ValueError("received native bytes differ from decoded text")
        if self.response_headers_b64 is not None:
            base64.b64decode(self.response_headers_b64, validate=True)
        return self


class ContentBlock(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, extra="allow")
    type: str
    text: str = ""


class Usage(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, extra="allow")
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class Sampling(BaseModel):
    model_config = ConfigDict(
        hide_input_in_errors=True, extra="allow", allow_inf_nan=False
    )
    prompt_token_ids: list[Annotated[int, Field(strict=True, ge=0)]] | None = None
    completion_token_ids: list[Annotated[int, Field(strict=True, ge=0)]] | None = None
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
    model_config = ConfigDict(hide_input_in_errors=True, extra="allow")
    content: list[ContentBlock]
    model: str
    usage: Usage
    stop_reason: str | None = None
    sampling_evidence: Sampling | None = None

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if block.type == "text")


class Decoder(BaseModel):
    model_config = ConfigDict(
        hide_input_in_errors=True, extra="forbid", allow_inf_nan=False
    )
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)


@dataclass
class LearnerWindow:
    seat: int
    policy: str
    deadline: float
    publish: Callable[[Attempt], Coroutine[Any, Any, None]]
    attempts: list[Attempt] = field(default_factory=list)


current_window: ContextVar[LearnerWindow] = ContextVar("gnomic_learner_window")


class AttemptPublisher:
    """Await progress on its original authenticated request and transport."""

    def __init__(
        self, rid: int, send: Callable[[dict], Coroutine[Any, Any, None]]
    ) -> None:
        self.rid = rid
        self.send = send

    async def __call__(self, attempt: Attempt) -> None:
        await self.send(
            {
                "type": "private_attempt",
                "rid": self.rid,
                "attempt": attempt.model_dump(mode="json"),
                "received_headers": attempt._received_header_pairs.model_dump(
                    mode="json"
                )
                if attempt._received_header_pairs is not None
                else None,
            }
        )


async def complete_native(
    body: dict,
    model: str,
    *,
    purpose: Literal["learner", "environment"],
    timeout: float,
    slot: int | None = None,
    generations: list[Attempt] | None = None,
) -> Response:
    owner_deadline = shutdown_deadline.get()
    if owner_deadline is not None and owner_deadline[0] is not None:
        raise OwnershipUnsettled("native request issued after ownership stop")
    if purpose == "environment" and slot is not None:
        raise ValueError("environment request must not claim a player slot")
    window = current_window.get(None) if purpose == "learner" else None
    if purpose == "learner":
        model = os.environ.get("COWORLD_LLM_MODEL", model)
        slot = window.seat if window is not None else slot
        if not isinstance(slot, int) or isinstance(slot, bool) or slot < 0:
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
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "accept-encoding": "identity",
    }
    if purpose == "learner":
        headers["X-Coworld-Player-Slot"] = str(slot)
    on_attempt = window.publish if window is not None else None
    attempt.request = payload
    attempt.prompt = [{"role": "system", "content": body["system"]}, *body["messages"]]
    attempt.decoder = {
        key: value
        for key, value in payload.items()
        if key not in {"model", "system", "messages"}
    }
    attempt.decoder["timeout_ms"] = timeout * 1000
    started = time.monotonic()
    deadline = asyncio.get_running_loop().time() + timeout

    client = httpx.AsyncClient(
        timeout=None, verify=ssl.create_default_context(), trust_env=False
    )
    response: httpx.Response | None = None
    try:
        async with asyncio.timeout_at(deadline):
            if on_attempt is not None:
                await on_attempt(attempt)
            endpoint = os.environ["COWORLD_LLM_ENDPOINT"].rstrip("/")
            response = await client.send(
                client.build_request(
                    "POST",
                    endpoint + "/v1/messages",
                    headers=headers,
                    content=json.dumps(payload).encode(),
                ),
                stream=True,
            )
            attempt._received_header_pairs = ReceivedHeaderPairs(
                attempt_id=attempt.attempt_id,
                pairs=[
                    (name.decode("latin-1"), value.decode("latin-1"))
                    for name, value in response.headers.raw
                ],
            )
            attempt.http_status = response.status_code
            attempt.response_complete = False
            attempt.response_reader_joined = False
            attempt.response_body_b64 = ""
            attempt.raw_response = ""
            names = [name.decode("ascii").lower() for name, _ in response.headers.raw]
            controlled = {
                "x-softmax-llm-call-id",
                "request-id",
                "x-request-id",
                "x-coworld-checkpoint-sha256",
                "x-coworld-tokenizer-sha256",
                "x-coworld-chat-template-sha256",
            }
            if any(names.count(name) > 1 for name in controlled):
                raise ValueError("duplicate native identity header")
            if "request-id" in names and "x-request-id" in names:
                if response.headers["request-id"] != response.headers["x-request-id"]:
                    raise ValueError("conflicting native request identity aliases")
            attempt.response_headers = dict(response.headers)
            attempt.provider_request_id = response.headers.get(
                "request-id"
            ) or response.headers.get("x-request-id")
            attempt.platform_call_id = response.headers.get("X-Softmax-Llm-Call-Id")
            attempt.model_identity = response.headers.get("X-Coworld-Checkpoint-Sha256")
            attempt.tokenizer_identity = response.headers.get(
                "X-Coworld-Tokenizer-Sha256"
            )
            attempt.chat_template_sha256 = response.headers.get(
                "X-Coworld-Chat-Template-Sha256"
            )
            if on_attempt is not None:
                await on_attempt(attempt)
            if (
                response.headers.get("content-encoding", "identity").lower()
                != "identity"
            ):
                raise ValueError("native response must honor identity encoding")
            received = bytearray()
            async for chunk in response.aiter_raw():
                received.extend(chunk)
                attempt.raw_response = None
                attempt.response_body_b64 = base64.b64encode(received).decode("ascii")
                if len(received) > 4_000_000:
                    raise ValueError("native response exceeds private packet budget")
                text = received.decode("utf-8", errors="ignore")
                attempt.raw_response = (
                    text if text.encode("utf-8") == received else None
                )
                if on_attempt is not None:
                    await on_attempt(attempt)
            attempt.response_complete = True
            attempt.raw_response = None
            raw = received.decode("utf-8")
            attempt.raw_response = raw
            if response.status_code != 200:
                raise ValueError(
                    f"native Messages failed with status {response.status_code}"
                )

        parsed = Response.model_validate_json(raw)
        attempt.model = parsed.model
        attempt.stop_reason = parsed.stop_reason
        attempt.input_tokens = parsed.usage.input_tokens
        attempt.output_tokens = parsed.usage.output_tokens
        attempt.response = "".join(
            block.text for block in parsed.content if block.type == "text"
        )
        if parsed.sampling_evidence is not None:
            sampled = parsed.sampling_evidence
            attempt.prompt_token_ids = sampled.prompt_token_ids
            attempt.sampled_token_ids = sampled.completion_token_ids
            attempt.behavior_logprobs = sampled.behavior_log_probs
            attempt.stop_reason = sampled.stop_reason
        if parsed.stop_reason == "refusal":
            raise ValueError("model refused the request")
        return parsed
    finally:
        failure = sys.exception()
        if failure is not None:
            attempt.rejection_reason = type(failure).__name__
        cleanup_deadline = asyncio.get_running_loop().time() + 1
        inherited = shutdown_deadline.get()
        if inherited is not None and inherited[0] is not None:
            cleanup_deadline = min(cleanup_deadline, inherited[0])

        async def close_transport() -> None:
            try:
                if response is not None:
                    await response.aclose()
            finally:
                await client.aclose()

        cleanup = owned_task(close_transport())
        joined = await settle({cleanup}, cleanup_deadline, cancel=False)
        if not joined:
            await settle({cleanup}, cleanup_deadline, cancel=True)
        elif not cleanup.cancelled():
            cleanup.result()
        if response is not None:
            attempt.response_reader_joined = joined
        attempt.latency_ms = (time.monotonic() - started) * 1000
        progress_joined = True
        if on_attempt is not None:
            progress = owned_task(on_attempt(attempt.model_copy(deep=True)))
            progress_joined = await settle({progress}, cleanup_deadline, cancel=False)
            if not progress_joined:
                await settle({progress}, cleanup_deadline, cancel=True)
            elif not progress.cancelled():
                progress.result()

        if not joined or not progress_joined:
            raise OwnershipUnsettled(
                "native transport or evidence writer did not settle"
            )
