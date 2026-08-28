from pathlib import Path


HTML = Path("gnomic/server/viewer/index.html").read_text()


def test_viewer_has_live_and_replay_surfaces() -> None:
    assert "GNOMIC MOOT" in HTML
    assert "new WebSocket" in HTML
    assert "isReplay" in HTML
    assert "proposal_made" in HTML
    assert "action_made" in HTML
    assert "action_ruling" in HTML
    assert "debate_made" in HTML
    assert "vote_reveal" in HTML
    assert "judge_ruling" in HTML
    assert 'id="scrub"' in HTML
    assert 'id="chamberTab"' in HTML
    assert 'id="lawTab"' in HTML


def test_viewer_has_accessible_replay_controls() -> None:
    for control in ("back", "play", "forward", "narration", "scrub"):
        assert f'id="{control}"' in HTML
    assert 'aria-label="Replay controls"' in HTML
    assert "aria-pressed" in HTML
    assert "prefers-reduced-motion: reduce" in HTML


def test_viewer_narration_follows_spoken_event_duration() -> None:
    assert "SpeechSynthesisUtterance" in HTML
    assert "utterance.onend = finish" in HTML
    assert "await narrateCurrentEvent()" in HTML
    assert "narrationEnabled ? 700 : 5200" in HTML
    assert "function narrationParts" in HTML
    assert "function splitSpeech" in HTML
    assert 'aria-label="Enable voice narration"' in HTML


def test_viewer_narration_prioritizes_player_speech_and_natural_voices() -> None:
    assert "proposes. ${proposal.text" in HTML
    assert "responds. ${statement.text}" in HTML
    assert 'The proposal ${state.passed ? "passes" : "fails"}' in HTML
    assert "has the floor. ${state.required} aye votes" not in HTML
    assert "The ballot is revealed" not in HTML
    assert "function voiceQuality" in HTML
    assert "Bad News|Bahh|Bells" in HTML
    assert "Natural|Enhanced|Premium|Neural|Online" in HTML
    assert "utterance.pitch = 1" in HTML
    assert "utterance.volume = 1" in HTML


def test_viewer_keeps_secondary_detail_collapsed() -> None:
    assert 'class="technical"' in HTML
    assert "Why the proposer says this should pass" in HTML
    assert "Inspect exact state changes" in HTML
    assert "JSON.stringify(changes, null, 2)" in HTML


def test_viewer_has_focused_event_views_and_constitution_history() -> None:
    for renderer in ("renderAction", "renderActionRuling", "renderProposal", "renderDebate", "renderVote", "renderRuling", "renderFinale"):
        assert f"function {renderer}" in HTML
    assert "earlier ${history.length === 1" in HTML
    assert "window.__GNOMIC_VIEWER__" in HTML


def test_viewer_uses_gnomic_visual_language_without_kickers() -> None:
    for generic_kicker in (
        "beat-kicker",
        "measure-label",
        "Softmax Universe",
        "The living text",
        "Proposed text",
        "✦",
    ):
        assert generic_kicker not in HTML
    assert '<div class="seal" aria-hidden="true">❧</div>' in HTML
    assert "--seat-1: #4fa8e8" in HTML
    assert "--seat-2: #62d97e" in HTML
