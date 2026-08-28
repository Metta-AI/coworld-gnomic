from __future__ import annotations

import pytest

from gnomic.players.scribe import ScribePolicy
from gnomic.players.sdk import InProcessTransport, PlayerSession
from gnomic.server.channel import InProcessChannel


@pytest.mark.asyncio
async def test_ergonomic_sdk_emits_current_protocol_shapes() -> None:
    channel = InProcessChannel(0)
    session = PlayerSession(ScribePolicy(), InProcessTransport(channel))
    await session.handle({
        "type": "game_start",
        "session": {"seats": [{"seat": 0}, {"seat": 1}, {"seat": 2}]},
        "you": {"seat": 0},
        "rules": [],
        "state": {"players": [{"points": 0}, {"points": 0}, {"points": 0}], "common": {}},
        "history": [],
    })
    await session.handle({"type": "turn_start", "turn": 1, "proposer": 0, "rules": [], "state": session.view.state})
    await session.handle({"type": "proposal_request", "turn": 1, "rid": 1, "timeout_s": 2})
    proposal = await channel._queue.get()
    assert proposal["rid"] == 1
    assert proposal["proposal"]["kind"] == "enact"

    session.view.proposal = proposal["proposal"]
    await session.handle({"type": "debate_request", "turn": 1, "rid": 2, "timeout_s": 2, "proposal": proposal["proposal"]})
    debate = await channel._queue.get()
    assert debate["rid"] == 2
    assert debate["vote_intent"] in {"aye", "nay"}
