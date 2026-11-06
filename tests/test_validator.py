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
    assert result["validation"]["priority"] == "medium"
    assert result["validation"]["inspection"]["in_queue"] is False
    codes = {reason["code"]: reason["status"] for reason in result["validation"]["priority_reasons"]}
    assert codes["unexplained_pronouns"] == "informational"
    assert codes["hard_callback"] == "heuristic"


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
    assert len(validate_candidates({"candidates": []})["inspection_queue"]["items"]) == 0


def _parsed_after(episode: int, text: str, start: int = 20_200, parentheticals=None) -> dict:
    return {
        "episodes": [
            {
                "episode_number": episode,
                "scenes": [
                    {
                        "scene_number": 1,
                        "cues": [
                            {
                                "type": "DIALOGUE",
                                "text": text,
                                "start_ms": start,
                                "end_ms": start + 1_500,
                                "parentheticals": parentheticals or [],
                                "speaker": "CATHERINE",
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _codes(candidate: dict) -> dict:
    return {reason["code"]: reason for reason in candidate["validation"]["priority_reasons"]}


def test_priority_levels_and_named_ellipsis_are_not_failures():
    father = _candidate(
        "father",
        1,
        [_line('"I changed my mind."', 0, 2_000), _line('"Father..."', 18_000, 20_000)],
        0,
        20_000,
    )
    alexander = _candidate(
        "alexander",
        42,
        [_line('"And yet, here I stand."', 0, 2_000, speaker="ALEXANDER"), _line('"Alexander..."', 18_000, 20_000, speaker="MOTHER")],
        0,
        20_000,
    )
    pronouns = _candidate(
        "pronouns",
        33,
        [
            _line('"She drew something around him."', 0, 3_000, speaker="LORD"),
            _line('"She placed herbs around him."', 4_000, 8_000, speaker="MAID"),
            _line('"She was praying!"', 9_000, 12_000, speaker="MAID"),
        ],
        0,
        12_000,
    )
    thin = _candidate("thin", 4, [_line('"No"', 0, 800)], 0, 20_000)
    father_result = validate_candidates({"candidates": [father]}, _parsed_after(1, '"Tell them this is a mistake."'))["candidates"][0]
    alexander_result = validate_candidates(
        {"candidates": [alexander]},
        _parsed_after(42, '"Alexander halts his march as his mother reaches him."'),
    )["candidates"][0]
    pronoun_result = validate_candidates({"candidates": [pronouns]})["candidates"][0]
    thin_result = validate_candidates({"candidates": [thin]})["candidates"][0]
    assert father_result["validation"]["priority"] == "low"
    assert _codes(father_result)["short_address_ellipsis"]["status"] == "informational"
    assert father_result["validation"]["inspection"]["in_queue"] is False
    assert alexander_result["validation"]["priority"] == "low"
    assert alexander_result["validation"]["inspection"]["in_queue"] is False
    assert pronoun_result["validation"]["priority"] == "low"
    assert _codes(pronoun_result)["unexplained_pronouns"]["status"] == "informational"
    assert thin_result["validation"]["priority"] == "medium"
    assert _codes(thin_result)["thin_opening"]["status"] == "heuristic"


def test_sentence_continuation_and_question_with_stronger_line_enter_the_queue():
    split = _candidate("split", 2, [_line('"You abandoned a woman who trusted you..."', 16_000, 19_000)], 0, 19_000)
    question = _candidate("question", 30, [_line('"What?"', 17_000, 20_000)], 0, 20_000)
    split_result = validate_candidates({"candidates": [split]}, _parsed_after(2, '"...because another woman promised you greater beauty."', 19_200))
    question_result = validate_candidates(
        {"candidates": [question]},
        _parsed_after(30, '"Burn the ledger. Burn every page."', 20_200, ["Sinister"]),
    )
    split_candidate = split_result["candidates"][0]
    question_candidate = question_result["candidates"][0]
    assert split_candidate["validation"]["priority"] == "high"
    assert _codes(split_candidate)["sentence_split"]["status"] == "confirmed"
    assert split_result["inspection_queue"]["inspection_count"] == 1
    assert question_candidate["validation"]["priority"] == "high"
    assert _codes(question_candidate)["question_then_stronger_line"]["status"] == "heuristic"
    assert question_result["inspection_queue"]["items"][0]["candidate_id"] == "question"


def test_family_grouping_is_deterministic_and_dedupes_same_endings():
    shared = [_line('"How convenient that the only person who can prove the poison exists..."', 16_000, 19_000)]
    other = [_line('"Where did a gutter rat like you find a silver locket?"', 21_000, 22_900)]
    seed = _candidate("seed", 28, [_line('"The maid found the poison."', 1_000, 4_000)], 0, 21_000)
    seed["scores"]["overall"] = 93.0
    early = _candidate("early", 28, shared, 12_000, 20_000)
    early["scores"]["overall"] = 86.0
    copy = _candidate("copy", 28, shared, 12_000, 20_000)
    copy["scores"]["overall"] = 70.0
    alternate = _candidate("alternate", 28, other, 14_000, 23_000)
    alternate["scores"]["overall"] = 60.0
    outside = _candidate("outside", 9, [_line('"A separate scene."', 0, 4_000)], 0, 20_000)
    outside["scores"]["overall"] = 99.0
    payload = {"candidates": [copy, seed, alternate, early, outside]}
    parsed = {
        "episodes": [
            {
                "episode_number": 28,
                "scenes": [
                    {
                        "scene_number": 1,
                        "cues": [
                            {"type": "DIALOGUE", "text": '"...is the maid who claims to have discovered it."', "start_ms": 20_200, "end_ms": 22_000, "parentheticals": [], "speaker": "CATHERINE"},
                            {"type": "DIALOGUE", "text": '"Give it back!"', "start_ms": 23_200, "end_ms": 24_000, "parentheticals": ["Sinister"], "speaker": "RUPERT"},
                        ],
                    }
                ],
            }
        ]
    }
    first = validate_candidates(payload, parsed)
    second = validate_candidates(payload, parsed)
    assert [item["validation"]["family"]["family_id"] for item in first["candidates"]] == [
        item["validation"]["family"]["family_id"] for item in second["candidates"]
    ]
    family_ids = {item["candidate_id"]: item["validation"]["family"]["family_id"] for item in first["candidates"]}
    assert family_ids["seed"] == family_ids["early"] == family_ids["copy"] == family_ids["alternate"]
    assert family_ids["outside"] != family_ids["seed"]
    queued = [item["candidate_id"] for item in first["inspection_queue"]["items"]]
    assert queued == ["early", "alternate"]
    roles = {item["candidate_id"]: item["role"] for item in first["inspection_queue"]["items"]}
    assert roles["early"] == "representative"
    assert roles["alternate"] == "alternative"
    assert first["inspection_queue"]["inspection_count"] == 2
    copy_row = next(item for item in first["candidates"] if item["candidate_id"] == "copy")
    assert copy_row["validation"]["inspection"]["in_queue"] is False
    assert copy_row["validation"]["inspection"]["covered_by"] == "early"
    assert len(first["candidates"]) == 5
    assert copy_row["scores"]["overall"] == 70.0
    assert copy_row["end_ms"] == 20_000


def test_queue_size_follows_the_input_rather_than_a_fixed_number():
    clips = []
    for episode in (1, 2, 3):
        clip = _candidate(f"ep{episode}", episode, [_line('"You trusted me..."', 16_000, 19_000)], 0, 19_000)
        clip["scores"]["overall"] = 80.0 - episode
        clips.append(clip)
    parsed = {
        "episodes": [
            {
                "episode_number": episode,
                "scenes": [
                    {
                        "scene_number": 1,
                        "cues": [
                            {
                                "type": "DIALOGUE",
                                "text": '"...and then you left."',
                                "start_ms": 19_200,
                                "end_ms": 21_000,
                                "parentheticals": [],
                                "speaker": "JULIANA",
                            }
                        ],
                    }
                ],
            }
            for episode in (1, 2, 3)
        ]
    }
    result = validate_candidates({"candidates": clips}, parsed)
    assert result["summary"]["family_count"] == 3
    assert result["summary"]["high_priority_family_count"] == 3
    assert result["inspection_queue"]["inspection_count"] == len(clips)
    assert result["summary"]["inspection_queue_count"] == 3
