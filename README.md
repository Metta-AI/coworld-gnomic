# Gnomic CoWorld

Gnomic is the Gnome Moot of Heartleaf: a three-player self-amending-rules game
for the Softmax Universe. It is a full reskin of the
[Nomic coworld](https://github.com/Metta-AI/coworld-nomic) into the world of
[Heartleaf](https://github.com/Metta-AI/coworld-heartleaf), the cozy gnome
garden-dinner game. Three gnome elders — Ivan of House One, Anton of House Two,
and Yura of House Three — act inside a game while rewriting its Gnome Law: each
active gnome states a natural-language move, proposes a change to the Law,
debates, votes, and receives binding rulings from a separate Claude Opus 4.7
Elder. An invalid move gets one repair attempt. Gnome Law can change actions,
voting, turn order, scoring, victory, and arbitrary public JSON state; a small
host shell keeps the protocol safe and ends the episode after at most 45 turns.

Game mechanics are identical to Nomic. The reskin covers the cast, the muse
words, the replay viewer (Heartleaf's PICO-8 palette and Moot language), and
every player- and spectator-facing surface.

## MVP rules

- Exactly three seats. Initial proposer order is 0, 1, 2. An amended order is
  an exact non-empty sequence of at most twelve seats: it may omit or repeat seats
  and begins next turn at its first listed seat.
- A proposal needs all three AYE votes through the first two complete circuits,
  then two AYEs. Transmuting a protected rule initially always needs unanimity.
- An adopted proposal gives its proposer 3 points; a rejected or vetoed proposal
  costs 1 point. Votes do not directly pay supporters.
- After every proposal ruling, seeded Fate rolls 1d6 and awards it to a uniformly
  random seat. The recorded host entropy can support amended random mechanics.
- The initial point victory is 100 and is checked only after each complete
  three-turn circuit. At the 45-turn cap, all players tied for most points co-win.
- Five procedural rules begin protected. A protected rule needs its own unanimous
  transmutation proposal before a later proposal may amend or repeal it.
- Each game exposes two deterministic, nonbinding muse words to encourage a fresh
  direction without hard-coding a large starting economy.
- Only seat count, phase skeleton, proposal-only rulebook mutation, declarative
  execution/safety bounds, and the hard turn cap are host constraints.
- The Elder may veto only an invalid, incoherent, impossible, or host-unsafe
  proposal—not one it considers strategically bad.

The game image includes a live spectator and replay renderer. The bundled
`baseline` is deterministic and makes no model calls. League players are three
separately uploaded Opus policies playing Ivan, Anton, and Yura. Player calls
use adaptive high-effort reasoning, a 20,000-token advisory task budget, and a
32,768-token hard ceiling. Non-proposers reconsider their vote after seeing
both debate statements, and contradictory/immediate-loss decisions are rejected
and repaired. Their shared scaffold treats each gnome as both competitor and
co-designer, favoring reusable mechanics and meaningful choices over seat gifts
and threshold churn.
Production decision windows are 600 seconds so a high-effort call may finish
without the host replacing it with a deterministic default.
The manifest also sets the hosted episode deadline to 100 minutes; the platform
otherwise applies a 20-minute Kubernetes deadline, which is too short for a
45-turn high-reasoning game.
The production Elder uses the same reasoning controls so its structured ruling
is not crowded out by hidden reasoning. Harmless model-added annotations are
discarded before strict validation; required fields and all executable rule/state
operations remain schema-checked and dry-run atomically by the engine.

The hosted commissioner runs daily Competition rounds with three seats, win-only
round scoring, and a seven-day EWMA leaderboard. A submitted policy first plays
one cheap, one-turn self-play qualifier using the deterministic judge; any policy
that completes it is promoted immediately. This intake division exercises the
hosted image without paying for a full production game.

The manifest sets `players_per_user` to three so one owner can keep Ivan,
Anton, and Yura active in the league at the same time.

The commissioner selects the qualifier through its own manifest variant rather
than an episode-level config override. That is deliberate: the deployed platform
wire contract resolves league episode configuration from `variant_id`.

## Local development

```bash
uv sync --extra dev
uv run pytest -q
docker compose build gnomic
```

Hydrate and certify the manifest using the current `coworld` CLI from the Metta
repository:

```bash
uv run coworld build compose.yaml coworld_manifest_template.json 0.1.0 coworld_manifest.json
uv run coworld certify coworld_manifest.json --timeout-seconds 300
```

Releases run through `.github/workflows/coworld-release.yml`
(`gh workflow run coworld-release.yml -f version=X.Y.Z`): build → certify →
upload the three persona policies from `tools/ci/policies.json` → upload the
coworld. The repo secret `SOFTMAX_TOKEN` is propagated by coworld-builder's
`propagate-secrets.yml`.
