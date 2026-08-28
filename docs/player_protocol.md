# Player protocol

Connect JSON-over-WebSocket to the URL in `COWORLD_PLAYER_WS_URL`. Every request
has a unique `rid`; echo it in the response. Unknown server messages should be
ignored. Late, malformed, or missing actions receive deterministic defaults.

Server state messages:

- `game_start`: session/seats, your seat, host constraints, complete rulebook,
  public state, and completed history.
- `turn_start`: turn, proposer, current vote threshold, complete rules and state.
  In that state, `proposer_cursor` points to the next regular entry after the
  current proposer. `proposer_order` may omit or repeat seats.
- `action_made`, `action_ruling`, `proposal_made`, `debate_made`, `vote_reveal`,
  `judge_ruling`: authoritative
  public deltas. `judge_ruling` also includes complete post-ruling rules/state.
- `game_over`: winners, league scores, in-game points, and reason.
- `final`: terminal process signal.

Decision requests and replies:

```json
{"type":"action_request","turn":1,"rid":1,"timeout_s":600}
{"rid":1,"action":"I spend one key to open the northern gate."}
```

The action is free-form natural language but must be authorized by rules already
active at the start of the turn. Use `"pass"` when no meaningful move is legal.
If the Judge rejects it, only that player receives one repair request:

```json
{"type":"action_repair_request","turn":1,"rid":2,"original_action":{"text":"..."},"rejection_reason":"...","timeout_s":600}
{"rid":2,"action":"I spend my one public key to open gate north."}
```

```json
{"type":"proposal_request","turn":1,"rid":3,"timeout_s":600}
{"rid":3,"proposal":{"kind":"enact","text":"...","rationale":"..."}}
```

For amendment use `kind:"amend"`, `rule_id`, and the complete replacement text.
For repeal use `kind:"repeal"` and `rule_id`.
For a protected rule, first use `kind:"transmute"` and `rule_id`; if that
unanimous measure passes, a later proposal may amend or repeal the now-mutable rule.

```json
{"type":"debate_request","turn":1,"rid":2,"proposal":{},"proposer":0,"timeout_s":360}
{"rid":2,"text":"Public argument","vote_intent":"aye"}
```

Both non-proposers receive debate requests at once and do not see each other's
statement until both windows close.

```json
{"type":"vote_request","turn":1,"rid":4,"proposal":{},"debates":[],"timeout_s":360}
{"rid":4,"vote":"nay","reason":"Private reason"}
```

Votes are hidden until all three replies/defaults are collected. A malformed or
missing vote is NAY.
