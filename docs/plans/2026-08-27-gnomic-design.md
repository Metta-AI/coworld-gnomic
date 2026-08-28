# Gnomic — design note (2026-08-27)

## What it is

Gnomic is the Nomic coworld reskinned into Heartleaf. Three gnome elders hold
the Gnome Moot and rewrite Gnome Law while playing under it. Mechanics,
protocol, engine, judge, server, and tests are Nomic's, unchanged; the reskin
covers every named and visible surface.

## Pins

- **Starter**: `Metta-AI/coworld-nomic` @ `e92488d` (0.2.3, the release with
  the Parley-style scrubber and the `/global` drain fix). Copied with fresh
  history; this repo is the definitive Gnomic tree.
- **Theme source**: `Metta-AI/coworld-heartleaf` — the cast (house gnomes
  Ivan, Anton, Yura), "Gnome Law" (Heartleaf's real tournament-rules name),
  and the PICO-8 palette (`data/pallete.png`).
- **Cast**: seat 0 Ivan of House One (institution-builder), seat 1 Anton of
  House Two (dinner-host, favor-keeper), seat 2 Yura of House Three
  (tinkerer, loophole-lover). The Judge is themed as **the Elder**; the wire
  role name stays `judge` and all protocol event names are unchanged.
- **Wire compatibility**: identical protocol and results schema
  (`constitution_victory` termination kept); replay format tag is
  `gnomic-replay-v1`.
- **Viewer**: same chrome and scrubber as Nomic 0.2.3; palette re-derived
  from Heartleaf's PICO-8 colors (night `#0e101f`, ink `#fff1e8`, heart-pink
  accent `#ff77a8`, seats amber/sky/leaf); Assembly→Moot, Constitution→Gnome
  Law, delegates→gnomes in every label and narration line.
- **Muse pairs** rethemed to Heartleaf (soup, lanterns, mushrooms, burrows,
  harvest, gardens, badgers, curfew — one abstract partner word each).
- **Policies**: `tools/ci/policies.json` uploads `gnomic-{ivan,anton,yura}-opus-4-8`
  (game image, `python -m gnomic.players.llm`, `GNOMIC_PERSONA` env) plus
  `gnomic-baseline` for fillers. Same Opus 4.8 scaffold as Nomic.
- **Release**: `.github/workflows/coworld-release.yml` (Nomic's, slug swapped):
  build → certify (container-backed replay liveness) → upload policies →
  upload coworld. Judge uses platform Bedrock creds; no coworld secret.
- **League**: platform ladder like Nomic's — Competition division, daily
  rounds, win scoring, seven-day EWMA, three persona champions
  (`players_per_user: 3`), baseline fillers, small-field budget.

## Non-goals

Any mechanics change. If a rule, window, threshold, or judge behavior differs
from Nomic 0.2.3, that is a bug in the reskin.
