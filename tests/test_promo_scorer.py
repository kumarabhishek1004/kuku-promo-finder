"""Tests for the four promo scoring dimensions."""

import json
from pathlib import Path

from src.promo_scorer import load_candidates, score_candidate, score_candidates, weighted_overall


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "outputs" / "promo_candidates.json"


def _cue(cue_type: str, start_ms: int, end_ms: int, text: str, **extra) -> dict:
    minutes, remainder = divmod(start_ms // 10, 6_000)
    seconds, fraction = divmod(remainder, 100)
    end_minutes, end_remainder = divmod(end_ms // 10, 6_000)
    end_seconds, end_fraction = divmod(end_remainder, 100)
    cue = {
        "type": cue_type,
        "text": text,
        "start": f"{minutes:02d}:{seconds:02d}.{fraction:02d}",
        "end": f"{end_minutes:02d}:{end_seconds:02d}.{end_fraction:02d}",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "line": extra.pop("line", 1),
    }
    if cue_type == "DIALOGUE":
        cue["speaker"] = extra.pop("speaker", "JULIANA")
        cue["parentheticals"] = extra.pop("parentheticals", ["Firm"])
        cue["raw_label"] = extra.pop("raw_label", "JULIANA (Firm)")
    cue.update(extra)
    return cue


def _candidate(dialogue=None, action=None, camera=None, start_ms=0, end_ms=20_000) -> dict:
    return {
        "candidate_id": "ep01-c0001",
        "episode_number": 1,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "start": "00:00.00",
        "end": "00:20.00",
        "dialogue": dialogue or [],
        "action": action or [],
        "camera": camera or [],
        "sound": [],
    }


def _scores_in_range(scored: dict) -> None:
    scores = scored["scores"]
    for name in ("hook", "cliffhanger", "standalone_clarity", "emotional_intensity"):
        assert 0 <= scores[name] <= 10
    assert 0 <= scores["overall"] <= 100
    assert scores["overall"] == weighted_overall(
        scores["hook"],
        scores["cliffhanger"],
        scores["standalone_clarity"],
        scores["emotional_intensity"],
    )


def test_weighted_score_boundaries_and_weights():
    assert weighted_overall(0, 0, 0, 0) == 0
    assert weighted_overall(10, 10, 10, 10) == 100
    assert weighted_overall(10, 0, 0, 0) == 35
    assert weighted_overall(0, 10, 0, 0) == 30
    assert weighted_overall(0, 0, 10, 0) == 20
    assert weighted_overall(0, 0, 0, 10) == 15
    assert weighted_overall(7, 6, 5, 8) == 64.5
    assert weighted_overall(11, -2, 0, 0) == 35


def test_hook_uses_only_the_first_three_seconds():
    early = _cue(
        "DIALOGUE",
        0,
        2_000,
        '"I will kill you!"',
        parentheticals=["Furious"],
        raw_label="JULIANA (Furious)",
    )
    late = _cue("DIALOGUE", 10_000, 13_000, '"I will kill you!"', parentheticals=["Furious"])
    opening = score_candidate(_candidate([early], end_ms=20_000))
    delayed = score_candidate(_candidate([late], start_ms=0, end_ms=20_000))
    assert opening["scores"]["hook"] >= 7
    assert delayed["scores"]["hook"] == 0
    assert "opening_has_no_spine_cue" in delayed["review_flags"]
    assert "kill" in json.dumps(opening["evidence"]["hook"])
    _scores_in_range(opening)


def test_cliffhanger_prefers_an_unresolved_ending():
    question = _cue("DIALOGUE", 17_500, 20_000, '"Who killed him?"', parentheticals=["Horrified"])
    farewell = _cue("DIALOGUE", 17_500, 20_000, '"Thank you."', parentheticals=["Calm"])
    padding = _cue("DIALOGUE", 0, 3_000, '"Stay here with me."')
    open_ending = score_candidate(_candidate([padding, question], end_ms=20_000))
    closed_ending = score_candidate(_candidate([padding, farewell], end_ms=20_000))
    assert open_ending["scores"]["cliffhanger"] >= 6
    assert closed_ending["scores"]["cliffhanger"] == 0
    assert "unresolved question" in open_ending["explanations"]["cliffhanger"]
    assert "Who killed him?" in json.dumps(open_ending["evidence"]["cliffhanger"])


def test_standalone_clarity_rewards_context_and_penalizes_fragments():
    visible = _cue(
        "ACTION",
        0,
        4_000,
        "Juliana stands in the cathedral aisle in her mother's white gown and looks up at the altar.",
    )
    statement = _cue("DIALOGUE", 4_000, 8_000, '"I will not marry you tomorrow morning."')
    reply = _cue("DIALOGUE", 8_000, 12_000, '"Then leave this house before sunset."', speaker="GERALD")
    clear = score_candidate(_candidate([statement, reply], [visible], end_ms=20_000))
    fragments = score_candidate(
        _candidate(
            [
                _cue("DIALOGUE", 0, 1_000, '"Why?"'),
                _cue("DIALOGUE", 17_000, 18_000, '"No."'),
            ],
            end_ms=20_000,
        )
    )
    assert clear["scores"]["standalone_clarity"] >= 7
    assert fragments["scores"]["standalone_clarity"] <= 4
    assert "Base clarity is 4" in clear["explanations"]["standalone_clarity"]


def test_emotional_intensity_follows_delivery_and_conflict_words():
    fierce = _cue("DIALOGUE", 0, 3_000, '"I will kill you."', parentheticals=["Furious"])
    mild = _cue("DIALOGUE", 0, 3_000, '"Pass the bread."', parentheticals=["Calm"])
    heated = score_candidate(_candidate([fierce], end_ms=20_000))
    quiet = score_candidate(_candidate([mild], end_ms=20_000))
    assert heated["scores"]["emotional_intensity"] >= 7
    assert quiet["scores"]["emotional_intensity"] == 0
    assert heated["scores"]["emotional_intensity"] > quiet["scores"]["emotional_intensity"]
    assert "keyword:kill" in json.dumps(heated["evidence"]["emotional_intensity"])


def test_missing_dialogue_and_missing_action_are_safe():
    action = _cue("ACTION", 0, 4_000, "She slaps him across the face and stumbles back from the altar.")
    action_only = score_candidate(_candidate([], [action], end_ms=20_000))
    assert "no_dialogue" in action_only["review_flags"]
    assert action_only["scores"]["hook"] >= 4
    assert action_only["scores"]["emotional_intensity"] >= 5
    _scores_in_range(action_only)

    line = _cue("DIALOGUE", 0, 3_000, '"I will kill you."', parentheticals=["Furious"])
    dialogue_only = score_candidate(_candidate([line], [], end_ms=20_000))
    assert "no_action" in dialogue_only["review_flags"]
    assert dialogue_only["scores"]["hook"] >= 7
    _scores_in_range(dialogue_only)


def test_empty_and_invalid_candidates_score_zero():
    empty = score_candidate({})
    assert empty["scores"] == {
        "hook": 0,
        "cliffhanger": 0,
        "standalone_clarity": 0,
        "emotional_intensity": 0,
        "overall": 0.0,
    }
    assert empty["needs_human_review"] is True
    invalid = score_candidate(None)
    assert invalid["scores"]["overall"] == 0
    assert "invalid_candidate" in invalid["review_flags"]
    payload = score_candidates(None)
    assert payload["candidates"] == []
    assert payload["summary"]["errors"]
    kept = score_candidates({"candidates": [{}, None]})
    assert len(kept["candidates"]) == 2


def test_scoring_is_reproducible_and_does_not_mutate_input():
    original = _candidate(
        [_cue("DIALOGUE", 0, 2_500, '"I will kill you!"', parentheticals=["Furious"])],
        [_cue("ACTION", 17_000, 20_000, "She reaches for the chapel door.")],
        end_ms=20_000,
    )
    snapshot = json.dumps(original)
    first = score_candidate(original)
    second = score_candidate(original)
    assert json.dumps(original) == snapshot
    assert "scores" not in original
    assert json.dumps(first) == json.dumps(second)


def test_saved_candidates_are_scored_without_being_removed():
    payload = load_candidates(CANDIDATES)
    result = score_candidates(payload)
    assert result["summary"]["errors"] == []
    assert len(result["candidates"]) == len(payload["candidates"])
    assert result["summary"]["candidate_count"] == len(payload["candidates"])
    for source, scored in zip(payload["candidates"], result["candidates"]):
        assert scored["candidate_id"] == source["candidate_id"]
        assert scored["episode_number"] == source["episode_number"]
        assert scored["start_ms"] == source["start_ms"]
        assert scored["end_ms"] == source["end_ms"]
        _scores_in_range(scored)
        assert set(scored["explanations"]) == {
            "hook",
            "cliffhanger",
            "standalone_clarity",
            "emotional_intensity",
            "overall",
        }
        assert set(scored["evidence"]) == {
            "hook",
            "cliffhanger",
            "standalone_clarity",
            "emotional_intensity",
        }
