"""Tests for dramatic-moment detection and promo candidate windows."""

from pathlib import Path

from src.event_detector import score_cue
from src.promo_generator import (
    MAX_DURATION_MS,
    MIN_DURATION_MS,
    NEAR_DUPLICATE_IOU,
    dedupe_candidates,
    generate_candidates,
    load_parsed,
)


ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "outputs" / "parsed_episodes.json"


def _stamp(milliseconds: int) -> str:
    centiseconds = milliseconds // 10
    minutes, remainder = divmod(centiseconds, 6_000)
    seconds, fraction = divmod(remainder, 100)
    return f"{minutes:02d}:{seconds:02d}.{fraction:02d}"


def _cue(cue_type: str, start_ms: int, end_ms: int, text: str, line: int, **extra) -> dict:
    cue = {
        "type": cue_type,
        "raw_label": extra.pop("raw_label", cue_type if cue_type != "DIALOGUE" else "JULIANA (Firm)"),
        "text": text,
        "start": _stamp(start_ms),
        "end": _stamp(end_ms),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "line": line,
    }
    if cue_type == "DIALOGUE":
        cue.setdefault("speaker", extra.pop("speaker", "JULIANA"))
        cue.setdefault("parentheticals", extra.pop("parentheticals", ["Firm"]))
        cue.setdefault("markers", extra.pop("markers", []))
    cue.update(extra)
    return cue


def _scene(number: int, start_ms: int, end_ms: int, cues: list[dict], heading: str = "INT. ROOM - DAY") -> dict:
    return {
        "scene_number": number,
        "scene_heading": heading,
        "scene_start": _stamp(start_ms),
        "scene_end": _stamp(end_ms),
        "scene_start_ms": start_ms,
        "scene_end_ms": end_ms,
        "cues": cues,
        "unparsed_lines": [],
    }


def _episode(scenes: list[dict], number: int = 1) -> dict:
    return {
        "episode_number": number,
        "catalog_id": "100",
        "filename_index": number,
        "filename": f"episode-{number:03d}-Episode 100.txt",
        "logline": "Logline.",
        "continuity_notes": "Notes.",
        "scenes": scenes,
        "errors": [],
        "warnings": [],
        "unparsed_lines": [],
    }


def _parsed(*episodes: dict) -> dict:
    return {"episodes": list(episodes)}


def _iou(left: dict, right: dict) -> float:
    intersection = min(left["end_ms"], right["end_ms"]) - max(left["start_ms"], right["start_ms"])
    if intersection <= 0:
        return 0.0
    union = (left["end_ms"] - left["start_ms"]) + (right["end_ms"] - right["start_ms"]) - intersection
    return intersection / union


def test_duration_stays_inside_20_to_60_seconds():
    cues = []
    line = 1
    for index in range(18):
        start = index * 5_000
        text = "I will kill you." if index == 8 else "We should keep walking."
        cues.append(_cue("DIALOGUE", start, start + 4_000, text, line))
        line += 1
    result = generate_candidates(_parsed(_episode([_scene(1, 0, 90_000, cues)])))
    assert result["candidates"]
    for candidate in result["candidates"]:
        assert MIN_DURATION_MS <= candidate["duration_ms"] <= MAX_DURATION_MS
        assert candidate["duration_ms"] == candidate["end_ms"] - candidate["start_ms"]
        assert candidate["duration_seconds"] == candidate["duration_ms"] / 1000
        assert 20 <= candidate["duration_seconds"] <= 60


def test_window_uses_dialogue_and_action_timestamps_not_camera():
    cues = [
        _cue("CAMERA", 0, 30_000, "Wide shot of the whole argument.", 1),
        _cue("DIALOGUE", 1_000, 4_000, "Stay back.", 2),
        _cue("ACTION", 4_000, 8_000, "She steps away.", 3),
        _cue("DIALOGUE", 8_000, 12_000, "I will kill you.", 4, raw_label="CATHERINE (Furious)", speaker="CATHERINE", parentheticals=["Furious"]),
        _cue("DIALOGUE", 12_000, 18_000, "You would not dare.", 5),
        _cue("DIALOGUE", 18_000, 24_000, "Watch me.", 6),
        _cue("SOUND FX", 0, 40_000, "A storm that lasts longer than the scene.", 7),
    ]
    result = generate_candidates(_parsed(_episode([_scene(1, 0, 40_000, cues)])))
    assert result["candidates"]
    for candidate in result["candidates"]:
        spine = candidate["dialogue"] + candidate["action"]
        spine.sort(key=lambda cue: (cue["start_ms"], cue["end_ms"]))
        assert candidate["start"] == spine[0]["start"]
        assert candidate["end"] == spine[-1]["end"]
        assert candidate["start_ms"] == spine[0]["start_ms"]
        assert candidate["end_ms"] == spine[-1]["end_ms"]
        assert candidate["start_ms"] != 0 or spine[0]["type"] != "CAMERA"
        assert all(cue["type"] != "CAMERA" for cue in spine)


def test_candidate_can_cross_scene_boundaries():
    scene_one = [
        _cue("DIALOGUE", 0, 4_000, "Good morning.", 1),
        _cue("DIALOGUE", 4_000, 9_000, "We should talk.", 2),
        _cue("DIALOGUE", 9_000, 14_000, "You look pale.", 3),
        _cue("DIALOGUE", 14_000, 18_000, "I will kill you.", 4, raw_label="GERALD (Vicious)", speaker="GERALD", parentheticals=["Vicious"]),
    ]
    scene_two = [
        _cue("DIALOGUE", 18_000, 24_000, "You would not.", 5),
        _cue("DIALOGUE", 24_000, 30_000, "Watch me.", 6),
        _cue("DIALOGUE", 30_000, 36_000, "Please stop.", 7),
        _cue("DIALOGUE", 36_000, 42_000, "It is too late.", 8),
    ]
    result = generate_candidates(
        _parsed(
            _episode(
                [
                    _scene(1, 0, 18_000, scene_one, "INT. HALL - DAY"),
                    _scene(2, 18_000, 42_000, scene_two, "EXT. YARD - CONTINUOUS"),
                ]
            )
        )
    )
    crossing = [candidate for candidate in result["candidates"] if len(candidate["scenes"]) >= 2]
    assert crossing
    scenes = {scene["scene_number"] for scene in crossing[0]["scenes"]}
    assert scenes == {1, 2}
    assert crossing[0]["start_ms"] < 18_000 < crossing[0]["end_ms"]


def test_dramatic_dialogue_and_action_anchor_the_window():
    cues = [
        _cue("DIALOGUE", 0, 5_000, "Pass the bread.", 1),
        _cue("DIALOGUE", 5_000, 10_000, "The weather is mild.", 2),
        _cue("ACTION", 10_000, 16_000, "She slaps him across the face.", 3),
        _cue("DIALOGUE", 16_000, 22_000, "Who told you the secret?", 4, raw_label="JULIANA (Shocked)", speaker="JULIANA", parentheticals=["Shocked"]),
        _cue("DIALOGUE", 22_000, 28_000, "My lady—", 5),
        _cue("DIALOGUE", 28_000, 34_000, "Say it.", 6),
    ]
    scored = score_cue(cues[2])
    assert "physical_action" in scored["categories"]
    assert scored["score"] >= 3
    question = score_cue(cues[3])
    assert "suspenseful_question" in question["categories"]
    assert "reveal" in question["categories"]
    interrupted = score_cue(cues[4])
    assert "interrupted_dialogue" in interrupted["categories"]

    result = generate_candidates(_parsed(_episode([_scene(1, 0, 34_000, cues)])))
    texts = {
        cue["text"]
        for candidate in result["candidates"]
        for cue in candidate["dialogue"] + candidate["action"]
    }
    assert "She slaps him across the face." in texts
    assert "Who told you the secret?" in texts
    triggers = {candidate["trigger"]["text"] for candidate in result["candidates"]}
    assert "Who told you the secret?" in triggers
    assert "My lady—" in triggers
    for candidate in result["candidates"]:
        included = [cue["text"] for cue in candidate["dialogue"] + candidate["action"]]
        assert candidate["trigger"]["text"] in included
        assert candidate["reason"]
        assert candidate["trigger"]["signals"]

    action_only = [
        _cue("DIALOGUE", 0, 6_000, "Pass the bread.", 1),
        _cue("DIALOGUE", 6_000, 12_000, "The weather is mild.", 2),
        _cue("ACTION", 12_000, 18_000, "She slaps him across the face.", 3),
        _cue("DIALOGUE", 18_000, 24_000, "Say that again.", 4),
        _cue("DIALOGUE", 24_000, 30_000, "I will not.", 5),
    ]
    action_result = generate_candidates(_parsed(_episode([_scene(1, 0, 30_000, action_only)])))
    assert {candidate["trigger"]["text"] for candidate in action_result["candidates"]} == {
        "She slaps him across the face."
    }
    assert action_result["candidates"][0]["trigger"]["cue_type"] == "ACTION"


def test_duplicate_and_near_duplicate_windows_are_removed():
    cues = []
    for index in range(8):
        start = index * 5_000
        cues.append(_cue("DIALOGUE", start, start + 4_500, "I will kill you.", index + 1))
    result = generate_candidates(_parsed(_episode([_scene(1, 0, 40_000, cues)])))
    assert result["summary"]["candidates_before_dedupe"] > result["summary"]["candidate_count"]
    candidates = result["candidates"]
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            assert _iou(left, right) < NEAR_DUPLICATE_IOU

    duplicate = {
        "episode_number": 3,
        "start_ms": 0,
        "end_ms": 30_000,
        "duration_ms": 30_000,
        "trigger": {"score": 3, "start_ms": 0, "end_ms": 2_000},
    }
    stronger = dict(duplicate)
    stronger["trigger"] = {"score": 9, "start_ms": 10_000, "end_ms": 12_000}
    assert len(dedupe_candidates([duplicate, stronger, dict(duplicate)])) == 1


def test_empty_and_invalid_input_produce_no_candidates():
    assert generate_candidates(None)["candidates"] == []
    assert generate_candidates(None)["summary"]["errors"]
    assert generate_candidates({})["candidates"] == []
    assert generate_candidates({"episodes": None})["candidates"] == []
    assert generate_candidates({"episodes": "nope"})["candidates"] == []
    assert generate_candidates({"episodes": [None, "bad"]})["candidates"] == []

    too_short = _episode(
        [
            _scene(
                1,
                0,
                12_000,
                [_cue("DIALOGUE", 0, 4_000, "I will kill you.", 1), _cue("DIALOGUE", 4_000, 8_000, "Run.", 2)],
            )
        ]
    )
    assert generate_candidates(_parsed(too_short))["candidates"] == []

    broken = _episode(
        [
            _scene(
                1,
                0,
                30_000,
                [
                    {
                        "type": "DIALOGUE",
                        "text": "I will kill you.",
                        "start": "00:00.00",
                        "end": None,
                        "start_ms": None,
                        "end_ms": None,
                        "speaker": "JULIANA",
                        "parentheticals": ["Furious"],
                        "line": 1,
                    }
                ],
            )
        ]
    )
    assert generate_candidates(_parsed(broken))["candidates"] == []

    gapped = _episode(
        [
            _scene(
                1,
                0,
                80_000,
                [
                    _cue("DIALOGUE", 0, 5_000, "I will kill you.", 1),
                    _cue("DIALOGUE", 70_000, 75_000, "I will kill you.", 2),
                ],
            )
        ]
    )
    assert generate_candidates(_parsed(gapped))["candidates"] == []


def test_camera_and_sound_are_attached_only_when_they_overlap():
    cues = [
        _cue("CAMERA", 0, 2_000, "Opening shot before the argument.", 1),
        _cue("DIALOGUE", 0, 4_000, "Good morning.", 2),
        _cue("CAMERA", 10_000, 16_000, "Close-up on the threat.", 3),
        _cue("SOUND FX", 12_000, 15_000, "A blade leaves its sheath.", 4),
        _cue("DIALOGUE", 4_000, 9_000, "We should talk.", 5),
        _cue("DIALOGUE", 9_000, 14_000, "You look pale.", 6),
        _cue("DIALOGUE", 14_000, 18_000, "I will kill you.", 7),
        _cue("DIALOGUE", 18_000, 24_000, "You would not.", 8),
        _cue("DIALOGUE", 24_000, 30_000, "Watch me.", 9),
        _cue("SOUND FX", 50_000, 55_000, "Later thunder.", 10),
        _cue("CAMERA", 50_000, 55_000, "A shot after the window.", 11),
    ]
    result = generate_candidates(_parsed(_episode([_scene(1, 0, 55_000, cues)])))
    assert any(
        "Close-up on the threat." in {cue["text"] for cue in candidate["camera"]}
        and "A blade leaves its sheath." in {cue["text"] for cue in candidate["sound"]}
        for candidate in result["candidates"]
    )
    for candidate in result["candidates"]:
        camera_text = {cue["text"] for cue in candidate["camera"]}
        sound_text = {cue["text"] for cue in candidate["sound"]}
        assert "A shot after the window." not in camera_text
        assert "Later thunder." not in sound_text
        if "Opening shot before the argument." in camera_text:
            assert candidate["start_ms"] < 2_000
        assert candidate["start_ms"] == candidate["dialogue"][0]["start_ms"]
        for cue in candidate["camera"] + candidate["sound"]:
            assert cue["start_ms"] < candidate["end_ms"]
            assert cue["end_ms"] > candidate["start_ms"]


def test_source_cues_are_not_mutated():
    cue = _cue("DIALOGUE", 0, 25_000, "I will kill you.", 1)
    original = set(cue.keys())
    generate_candidates(_parsed(_episode([_scene(1, 0, 25_000, [cue])])))
    assert set(cue.keys()) == original


def test_real_parsed_episodes_follow_window_rules():
    parsed = load_parsed(PARSED)
    result = generate_candidates(parsed)
    summary = result["summary"]
    assert summary["errors"] == []
    assert summary["candidate_count"] > 0
    assert summary["duplicates_removed"] >= 0
    assert summary["candidate_count"] <= summary["candidates_before_dedupe"]
    seen = set()
    for candidate in result["candidates"]:
        assert MIN_DURATION_MS <= candidate["duration_ms"] <= MAX_DURATION_MS
        assert candidate["duration_ms"] == candidate["end_ms"] - candidate["start_ms"]
        assert candidate["start"]
        assert candidate["end"]
        assert candidate["candidate_id"].startswith(f"ep{candidate['episode_number']:02d}-")
        assert candidate["candidate_id"] not in seen
        seen.add(candidate["candidate_id"])
        spine = sorted(candidate["dialogue"] + candidate["action"], key=lambda cue: (cue["start_ms"], cue["end_ms"], cue["line"]))
        assert spine[0]["start_ms"] == candidate["start_ms"]
        assert spine[-1]["end_ms"] == candidate["end_ms"]
        assert candidate["trigger"]["text"] in {cue["text"] for cue in spine}
        for cue in candidate["camera"] + candidate["sound"]:
            assert cue["start_ms"] < candidate["end_ms"]
            assert cue["end_ms"] > candidate["start_ms"]
        assert candidate["scenes"]
    for left_index, left in enumerate(result["candidates"]):
        for right in result["candidates"][left_index + 1 :]:
            if left["episode_number"] == right["episode_number"]:
                assert _iou(left, right) < NEAR_DUPLICATE_IOU
