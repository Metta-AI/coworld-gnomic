"""Gnomic commissioner adapter for the deployed Coworld wire contract.

The generic ruleset-strategy commissioner can attach a game_config override to
an episode request, but the currently deployed platform protocol intentionally
resolves episode configuration from variant_id. Select Gnomic's explicit cheap
qualifier variant here and self-seat only the round's scheduled candidate.
"""

from __future__ import annotations

from uuid import UUID

from commissioners.common.commissioners import get_commissioner
from commissioners.common.protocol import EpisodeRequest, RoundStart, ScheduleEpisodes
from commissioners.common.ruleset_strategy.entrants import select_rule
from commissioners.common.ruleset_strategy.round_start import RoundStartView
from commissioners.common.server import create_app

QUALIFIER_DIVISION = "Qualifiers"
QUALIFIER_VARIANT_ID = "qualifier-1-turn"
RulesetStrategyCommissioner = type(get_commissioner("config_driven"))


class GnomicCommissioner(RulesetStrategyCommissioner):
    def schedule_episodes_for_round_start(self, round_start: RoundStart) -> ScheduleEpisodes:
        config = self._config()
        view = RoundStartView(round_start, config)
        if view.current_division.name != QUALIFIER_DIVISION:
            return super().schedule_episodes_for_round_start(round_start)

        rule = select_rule(config, view.current_division, view.memberships)
        entries = view.entries(rule)
        configured = {
            UUID(str(policy_id))
            for policy_id in (view.round_config.get("entrant_policy_version_ids") or [])
        }
        if configured:
            entries = [entry for entry in entries if entry.policy_version_id in configured]
        if not entries:
            raise ValueError("Gnomic qualifier round has no scheduled candidate")

        variant = next(
            (candidate for candidate in round_start.variants if candidate.id == QUALIFIER_VARIANT_ID),
            None,
        )
        if variant is None:
            raise ValueError(f"Gnomic Coworld is missing variant {QUALIFIER_VARIANT_ID!r}")
        num_agents = variant.game_config.get("num_agents")
        if not isinstance(num_agents, int) or num_agents != 3:
            raise ValueError("Gnomic qualifier variant must declare num_agents=3")

        return ScheduleEpisodes(
            episodes=[
                EpisodeRequest(
                    request_id=str(index),
                    variant_id=QUALIFIER_VARIANT_ID,
                    policy_version_ids=[entry.policy_version_id] * num_agents,
                    tags={"pool_id": str(view.pool(rule).id), "gnomic_phase": "qualifier"},
                )
                for index, entry in enumerate(entries)
            ]
        )


app = create_app(GnomicCommissioner())
