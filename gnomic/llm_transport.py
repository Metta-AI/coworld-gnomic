"""Native Coworld Messages requests shared by the judge and player clients."""

import json
import os
import urllib.request


def complete_native(body: dict, model: str, *, slot: int | None = None) -> dict:
    payload = {"model": model, "max_tokens": body["max_tokens"],
               "system": body["system"], "messages": body["messages"]}
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if slot is not None:
        headers["X-Coworld-Player-Slot"] = str(slot)
    endpoint = os.environ["COWORLD_LLM_ENDPOINT"].rstrip("/")
    request = urllib.request.Request(endpoint + "/v1/messages", method="POST",
                                     headers=headers, data=json.dumps(payload).encode())
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())
