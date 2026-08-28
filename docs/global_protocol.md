# Global spectator and replay protocol

`/global` sends a `hello` immediately, then the complete public event log so far,
then live events. Events are `game_start`, `turn_start`, `action_made`,
`action_ruling`, `proposal_made`, `debate_made`, `vote_reveal`, `judge_ruling`,
`game_over`, and `final`. There is
no private player state or model reasoning in this stream.

In replay mode, `/replay` sends one message:

```json
{"type":"replay","data":{"format":"gnomic-replay-v1","events":[]}}
```

The same browser renderer reduces the replay's `events` array through the same
public state path used for live events. Replay also contains turn records, final
board, result, and aggregate Judge token/latency telemetry.
