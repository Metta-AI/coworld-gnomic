"""Episode-stable player wording, independent of the game's random stream."""

from __future__ import annotations

import hashlib
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

TRAINING = ("canonical", "plain", "numbered", "questions", "reordered")
VARIANTS = (*TRAINING, "heldout-briefing", "heldout-reference", "mixed")
BANK = "gnomic-player-rules-v2"


class PromptSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    variant: Literal[
        "canonical",
        "plain",
        "numbered",
        "questions",
        "reordered",
        "heldout-briefing",
        "heldout-reference",
        "mixed",
    ] = "canonical"
    seed: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def explicit_mixed_seed(self) -> PromptSelection:
        if self.variant == "mixed" and "seed" not in self.model_fields_set:
            raise ValueError("mixed wording requires an explicit prompt seed")
        return self

    @property
    def selected(self) -> str:
        return (
            TRAINING[self.seed % len(TRAINING)]
            if self.variant == "mixed"
            else self.variant
        )

    def introduction(self, canonical: str, identity: str) -> str:
        return {
            "canonical": canonical,
            "plain": f"Your role in this game: {identity}.",
            "numbered": f"1. Game role: {identity}.",
            "questions": f"What is your role in this game? {identity}.",
            "reordered": f"In this game, {identity} is your role.",
            "heldout-briefing": f"Take the following game role: {identity}.",
            "heldout-reference": f"Player role reference: {identity}.",
        }[self.selected]

    def metadata(self, prompt: list[dict[str, str]]) -> dict:
        return {
            "bank": BANK,
            "variant": self.selected,
            "selection": self.variant,
            "seed": self.seed,
            "messages_sha256": [
                hashlib.sha256(message["content"].encode()).hexdigest()
                for message in prompt
            ],
        }

    def rules(
        self, canonical: str, paragraphs: tuple[str, ...], heldout: tuple[str, ...]
    ) -> str:
        if self.selected == "canonical":
            return canonical
        if self.selected in ("heldout-briefing", "heldout-reference"):
            paragraphs = heldout
        if self.selected in ("reordered", "heldout-reference"):
            paragraphs = tuple(reversed(paragraphs))
        if self.selected == "numbered":
            return "\n".join(
                f"{index}. {paragraph}" for index, paragraph in enumerate(paragraphs, 1)
            )
        if self.selected == "questions":
            return "What instructions govern this decision?\n" + "\n\n".join(paragraphs)
        return "\n\n".join(paragraphs)
