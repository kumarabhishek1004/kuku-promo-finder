"""Tests for promo quality validation."""

import json

from src.validator import validate_candidates


def _line(text: str, start: int, end: int, speaker: str = "JULIANA") -> dict:
    return {
        "type": "DIALOGUE",
        "speaker": speaker,
        "text": text,
        "start_ms": start,
        "end_ms": end,
        "parentheticals": ["Firm"],
    }


def _candidate(candidate_id: str, episode: int, dialogue: list[dict], start: int, end: int, scenes=None) -> dict:
    return {
        "candidate_id": candidate_id,
        "episode_number": episode,
        "start_ms": start,
        "end_ms": end,
        "start": "00:00.00",
        "end": "00:20.00",
        "scenes": scenes or [{"scene_number": 1, "scene_heading": "INT. ROOM - DAY"}],
        "dialogue": dialogue,
        "action": [],
        "scores": {
            "hook": 4,
            "cliffhanger": 4,
            "standalone_clarity": 6,
            "emotional_intensity": 6,
            "overall": 48.0,
        },
    }


def test_identical_clips_are_flagged_and_kept():
    line = [_line('"I will kill you."', 0, 4_000)]
    first = _candidate("a", 3, line, 0, 20_000)
    second = _candidate("b", 3, line, 0, 20_000)
    result = validate_candidates({"candidates": [first, second]})
    assert len(result["candidates"]) == 2
    assert result["duplicate_pairs"][0]["reasons"] == ["iou_at_least_0.50", "shorter_clip_overlap_at_least_0.60", "text_similarity_with_partial_overlap", "text_similarity_with_shared_scene"]
    assert result["duplicate_pairs"][0]["discarded"] is False
    assert result["candidates"][0]["validation"]["duplicate_warnings"][0]["other_candidate_id"] == "b"


def test_partial_overlap_below_half_iou_is_still_flagged():
    long_clip = _candidate(
        "long",
        28,
        [_line('"She looks exactly like a poisoner."', 12_700, 15_000)],
        12_700,
        64_800,
        [{"scene_number": 1}],
    )
    short_clip = _candidate(
        "short",
        28,
        [_line('"The maid found the poison."', 45_600, 48_000)],
        45_600,
        66_800,
        [{"scene_number": 1}],
    )
    result = validate_candidates({"candidates": [long_clip, short_clip]})
    pair = result["duplicate_pairs"][0]
    assert pair["iou"] < 0.50
    assert pair["shorter_overlap"] >= 0.60
    assert "shorter_clip_overlap_at_least_0.60" in pair["reasons"]
    assert "iou_at_least_0.50" not in pair["reasons"]
    assert pair["human_review"] is True


def test_mid_sentence_ending_and_unresolved_question():
    cut_off = _candidate("cut", 2, [_line('"You abandoned a woman who trusted you..."', 16_500, 19_000)], 0, 19_000)
    question = _candidate("ask", 35, [_line('"A traitor?"', 17_000, 20_000, speaker="MOTHER")], 0, 20_000)
    cut_result = validate_candidates({"candidates": [cut_off]})["candidates"][0]
    ask_result = validate_candidates({"candidates": [question]})["candidates"][0]
    assert "ending_mid_sentence" in cut_result["validation"]["flags"]
    assert cut_result["validation"]["ending"]["ends_mid_sentence"] is True
    assert ask_result["validation"]["ending"]["unresolved_question_threat_reveal_or_confrontation"] is True
    assert "ending_resolves_or_explains" not in ask_result["validation"]["flags"]
    assert "ending_mid_sentence" not in ask_result["validation"]["flags"]


def test_weak_ending_and_stronger_following_line():
    weak = _candidate(
        "dose",
        25,
        [
            _line('"What did you find?"', 0, 2_000, speaker="ALEXANDER"),
            _line(
                '"Someone has been giving it to her again and again. Small amounts."',
                16_000,
                20_000,
            ),
        ],
        0,
        20_000,
    )
    parsed = {
        "episodes": [
            {
                "episode_number": 25,
                "scenes": [
                    {
                        "scene_number": 1,
                        "cues": [
                            {
                                "type": "DIALOGUE",
                                "text": '"Then call it witchcraft."',
                                "start_ms": 20_400,
                                "end_ms": 22_000,
                                "parentheticals": ["Sinister"],
                                "speaker": "ELDER",
                            }
                        ],
                    }
                ],
            }
        ]
    }
    result = validate_candidates({"candidates": [weak]}, parsed)["candidates"][0]
    assert "ending_resolves_or_explains" in result["validation"]["flags"]
    assert "stronger_beat_after_cut" in result["validation"]["flags"]
    assert result["validation"]["ending"]["following_line"]["text"] == '"Then call it witchcraft."'
    assert result["end_ms"] == 20_000


def test_missing_context_is_flagged_without_claiming_understanding():
    vague = _candidate(
        "vague",
        36,
        [
            _line('"Then where is she?"', 0, 2_000, speaker="ALEXANDER"),
            _line('"They arrested her!"', 3_000, 6_000, speaker="MOTHER"),
            _line('"But she saved you again!"', 8_000, 11_000, speaker="ALEXANDER"),
        ],
        0,
        12_000,
    )
    result = validate_candidates({"candidates": [vague]})["candidates"][0]
    clarity = result["validation"]["clarity"]
    assert "unexplained_pronouns" in result["validation"]["flags"]
    assert "previous_event_reference" in result["validation"]["flags"]
    assert "emotionally compelling" in clarity["limitation"]
    assert result["validation"]["human_review"]["recommended"] is True


def test_empty_input_and_original_records_stay_intact():
    assert validate_candidates(None)["candidates"] == []
    assert validate_candidates(None)["summary"]["errors"]
    assert validate_candidates({})["summary"]["errors"]
    assert validate_candidates({"candidates": []})["summary"]["candidate_count"] == 0
    original = _candidate("same", 1, [_line('"I changed my mind."', 0, 2_000, speaker="GERALD")], 0, 20_000)
    snapshot = json.dumps(original)
    validated = validate_candidates({"candidates": [original]})["candidates"][0]
    assert json.dumps(original) == snapshot
    assert "validation" not in original
    assert validated["scores"] == original["scores"]
    assert validated["start_ms"] == original["start_ms"]
    assert validated["end_ms"] == original["end_ms"]
