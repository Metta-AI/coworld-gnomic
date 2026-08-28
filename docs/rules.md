# Gnomic rules and judge boundary

The active Gnome Law begins as five protected procedural rules and eight
mutable game rules. Players may enact a new mutable rule, amend or repeal an
active mutable rule by stable numeric ID, or unanimously transmute one protected
rule into a mutable rule. Transmutation and amendment require separate proposals.
Every completed turn is natural-language action, proposal, parallel debate by the
two non-proposers, simultaneous secret vote by all three seats, then one binding
Judge ruling.

An action may use any move authorized by rules already active at that turn's
start. The action Judge translates it to bounded public-state operations. If the
move is illegal or materially unclear, the Judge gives a precise reason and the
active player receives one repair attempt. Passing is always legal. An action can
never change the rulebook, and the later proposal cannot retroactively authorize it.

The public board consists of versioned rules, a JSON object for each player, and
one common JSON object. `points`, `votes_required`, `proposer_order`, proposal
awards and penalties, `victory_check_every`, Fate controls, and `victory_points` begin as ordinary
Gnome Law state and can be changed by a valid ruling, subject to
runtime-safe types. Initially an adoption gives the proposer 3 points, rejection
costs 1, and no vote directly pays its caster. After the ruling, recorded Fate
awards 1d6 points to a uniformly random seat. The starting point victory is 100
and is checked at the end of complete three-turn circuits.

`proposer_order` is an exact non-empty sequence of at most twelve valid seat
numbers, and may omit or repeat seats. A changed sequence takes effect on the
next turn at its first listed seat. The host advances a public
`proposer_cursor`; the Judge cannot write that bookkeeping field and must never
pad or normalize a proposed order.

The Judge receives the complete public board and turn transcript. Claude Opus
4.8 returns only typed rule and state operations. The engine dry-runs and applies
the operations atomically. Malformed or semantically unsafe output is repaired
once; a second failure fails the episode. Model-generated code is never run.

For deterministic timing, each turn's published board already includes effects
due at that turn's start. The preceding ruling prepares those effects after it
resolves current-turn scoring, including effects from newly adopted rules. It
does not prepare a nonexistent turn after the host cap or after victory.

Immutable host constraints are limited to three seats, the five protocol phases,
proposal-only rulebook mutation, declarative bounded state, isolation from
secrets/I/O/code execution, and a 45-turn cap. At that cap, if no active rule has established another valid
victory, all seats tied for most points co-win.
Co-winners split one tournament win point equally rather than each receiving
full credit, so universal obstruction is not equivalent to winning alone.
