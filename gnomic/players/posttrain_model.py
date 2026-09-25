"""Run a Metta-trained Gnomic adapter through the ordinary player SDK."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .posttrain_prompt import messages_for
from .sdk import GameView, Policy, main_for


class ModelPolicy(Policy):
    def __init__(self, generate: Callable[[list[dict[str, str]], float], str]) -> None:
        self.generate = generate

    def _reply(self, view: GameView) -> dict[str, Any]:
        timeout = float(15 if view.request["type"] == "introduce_request" else view.request["timeout_s"])
        deadline = time.monotonic() + timeout - min(1.0, timeout / 4)
        reply = json.loads(self.generate(messages_for(view), deadline))
        if not isinstance(reply, dict) or reply["rid"] != view.request["rid"]:
            raise ValueError("Model reply has the wrong request id")
        return reply

    def introduce(self, view: GameView) -> str:
        name = self._reply(view)["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model gnome name must be nonempty text")
        return name

    def action(self, view: GameView) -> str:
        action = self._reply(view)["action"]
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Model action must be nonempty text")
        return action

    def propose(self, view: GameView) -> dict:
        proposal = self._reply(view)["proposal"]
        if not isinstance(proposal, dict):
            raise ValueError("Model proposal must be an object")
        return proposal

    def debate(self, view: GameView) -> dict:
        reply = self._reply(view)
        if not isinstance(reply["text"], str) or reply["vote_intent"] not in {
            "aye",
            "nay",
        }:
            raise ValueError("Model debate reply has invalid fields")
        return {"text": reply["text"], "vote_intent": reply["vote_intent"]}

    def vote(self, view: GameView) -> str:
        vote = self._reply(view)["vote"]
        if vote not in {"aye", "nay"}:
            raise ValueError("Model vote must be aye or nay")
        return vote


class TransformersGenerator:
    def __init__(
        self, adapter: Path, device: str, max_input_tokens: int, max_new_tokens: int
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        manifest = json.loads((adapter / "training_manifest.json").read_text())
        base = manifest["model"]
        revision = manifest["revision"]
        if Path(base).is_dir():
            digest = hashlib.sha256()
            for path in sorted(
                path for path in Path(base).rglob("*") if path.is_file()
            ):
                digest.update(str(path.relative_to(base)).encode())
                digest.update(hashlib.sha256(path.read_bytes()).digest())
            if revision != f"local-sha256:{digest.hexdigest()}":
                raise ValueError("Base model differs from the training manifest")
            self.tokenizer = AutoTokenizer.from_pretrained(base)
            network = AutoModelForCausalLM.from_pretrained(base, device_map=device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
            network = AutoModelForCausalLM.from_pretrained(
                base, revision=revision, device_map=device
            )
            if network.config._commit_hash != revision:
                raise ValueError(
                    "Base model revision differs from the training manifest"
                )
        self.network = PeftModel.from_pretrained(network, adapter)
        self.network.eval()
        self.lock = threading.Lock()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.torch = torch

    def __call__(self, messages: list[dict[str, str]], deadline: float) -> str:
        from transformers import MaxTimeCriteria, StoppingCriteriaList

        with self.lock:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            ).to(self.network.device)
            length = inputs["input_ids"].shape[-1]
            if length > self.max_input_tokens:
                raise ValueError(
                    f"Player prompt has {length} tokens, above {self.max_input_tokens}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("No model decision time remains")
            with self.torch.inference_mode():
                generated = self.network.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList(
                        [MaxTimeCriteria(max_time=remaining)]
                    ),
                )
            return self.tokenizer.decode(
                generated[0, length:], skip_special_tokens=True
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    if args.max_input_tokens < 64 or args.max_new_tokens < 1:
        raise ValueError("Model token limits must be positive")
    generator = TransformersGenerator(
        args.adapter, args.device, args.max_input_tokens, args.max_new_tokens
    )
    main_for(lambda: ModelPolicy(generator))


if __name__ == "__main__":
    main()
