"""Tests for deterministic promo ranking and shortlist diversity."""

import json
from pathlib import Path

from src.ranking import rank_candidates


ROOT = Path(__file__).resolve().parents[1]
SCORED = ROOT / "outputs" / "scored_candidates.json"


def _candidate(
    candidate_id: str,
    episode: int,
    overall: float,
    text: str,
    start_ms: int = 0,
    end_ms: int = 20_000,
    hook: int = 5,
    cliffhanger: int = 5,
    clarity: int = 5,
    intensity: int = 5,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "episode_number": episode,
        "start": "00:00.00",
        "end": "00:20.00",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_seconds": (end_ms - start_ms) / 1000,
        "dialogue": [{"type": "DIALOGUE", "text": text, "start_ms": start_ms, "end_ms": end_ms}],
        "action": [],
        "scores": {
            "hook": hook,
            "cliffhanger": cliffhanger,
            "standalone_clarity": clarity,
            "emotional_intensity": intensity,
            "overall": overall,
        },
    }


def test_candidates_sort_by_overall_score_and_keep_dimension_scores():
    payload = {
        "candidates": [
            _candidate("low", 1, 30.0, "A quiet line.", hook=1, cliffhanger=2, clarity=3, intensity=4),
            _candidate("high", 2, 90.0, "I will kill you.", hook=10, cliffhanger=9, clarity=8, intensity=7),
            _candidate("mid", 3, 60.0, "Who did this?", hook=6, cliffhanger=6, clarity=6, intensity=6),
        ]
    }
    result = rank_candidates(payload, top_n=3, max_per_episode=1)
    order = [row["candidate_id"] for row in result["ranked_candidates"]]
    assert order == ["high", "mid", "low"]
    assert [row["score_rank"] for row in result["ranked_candidates"]] == [1, 2, 3]
    best = result["recommendations"][0]
    assert best["scores"] == {
        "hook": 10,
        "cliffhanger": 9,
        "standalone_clarity": 8,
        "emotional_intensity": 7,
        "overall": 90.0,
    }
    assert "Overall 90.0" in best["ranking_explanation"]
    assert "hook 10" in best["ranking_explanation"]


def test_near_duplicate_segments_are_excluded_and_recorded():
    original = _candidate("keep", 4, 88.0, "I will kill you in the cathedral.", start_ms=0, end_ms=30_000)
    overlap = _candidate("drop-time", 4, 87.0, "A different spoken line entirely.", start_ms=2_000, end_ms=28_000)
    same_words = _candidate(
        "drop-text",
        4,
        86.0,
        "I will kill you in the cathedral.",
        start_ms=40_000,
        end_ms=60_000,
    )
    other = _candidate("other", 5, 70.0, "The guards open the gate.", start_ms=0, end_ms=20_000)
    result = rank_candidates(
        {"candidates": [same_words, other, overlap, original]},
        top_n=4,
        max_per_episode=4,
    )
    recommended = [row["candidate_id"] for row in result["recommendations"]]
    assert recommended == ["keep", "other"]
    reasons = {row["candidate_id"]: row["exclusion_reason"] for row in result["excluded_from_shortlist"]}
    assert reasons["drop-time"] == "near_duplicate_time"
    assert reasons["drop-text"] == "near_duplicate_text"
    assert all(row["similar_to"] == "keep" for row in result["excluded_from_shortlist"])
    assert len(result["ranked_candidates"]) == 4


def test_same_timestamps_in_different_episodes_are_not_time_duplicates():
    first = _candidate("ep1", 1, 80.0, "Juliana faces the altar.", start_ms=0, end_ms=25_000)
    second = _candidate("ep2", 2, 79.0, "Alexander enters the hall.", start_ms=0, end_ms=25_000)
    result = rank_candidates({"candidates": [second, first]}, top_n=2, max_per_episode=1)
    assert [row["candidate_id"] for row in result["recommendations"]] == ["ep1", "ep2"]
    assert result["excluded_from_shortlist"] == []


def test_episode_diversity_cap():
    rows = [
        _candidate("a1", 1, 95.0, "First cathedral threat."),
        _candidate("a2", 1, 94.0, "Second barn confession.", start_ms=40_000, end_ms=60_000),
        _candidate("a3", 1, 93.0, "Third courtroom reveal.", start_ms=80_000, end_ms=100_000),
        _candidate("b1", 2, 80.0, "The spy names the duke."),
        _candidate("c1", 3, 70.0, "The wedding doors close."),
    ]
    result = rank_candidates({"candidates": rows}, top_n=3, max_per_episode=1)
    assert [row["candidate_id"] for row in result["recommendations"]] == ["a1", "b1", "c1"]
    assert result["summary"]["episodes_in_shortlist"] == [1, 2, 3]
    capped = [row for row in result["excluded_from_shortlist"] if row["exclusion_reason"] == "episode_cap"]
    assert {row["candidate_id"] for row in capped} == {"a2", "a3"}


def test_top_n_leaves_lower_scores_in_the_full_list():
    rows = [_candidate(f"c{index}", index, float(index), f"Line number {index}.") for index in range(1, 6)]
    result = rank_candidates({"candidates": rows}, top_n=2, max_per_episode=1)
    assert [row["candidate_id"] for row in result["recommendations"]] == ["c5", "c4"]
    assert len(result["ranked_candidates"]) == 5
    assert result["summary"]["below_shortlist_count"] == 3
    assert result["summary"]["candidate_count"] == 5


def test_empty_inputs_and_original_records_are_unchanged():
    empty = rank_candidates(None)
    assert empty["recommendations"] == []
    assert empty["ranked_candidates"] == []
    assert empty["summary"]["errors"]
    assert rank_candidates({})["recommendations"] == []
    assert rank_candidates({"candidates": None})["summary"]["errors"]
    assert rank_candidates({"candidates": []})["summary"]["candidate_count"] == 0

    original = _candidate("same", 1, 50.0, "I will kill you.")
    snapshot = json.dumps(original)
    rank_candidates({"candidates": [original]}, top_n=1, max_per_episode=1)
    assert json.dumps(original) == snapshot
    assert "score_rank" not in original
    assert original["scores"]["overall"] == 50.0


def test_ranking_is_deterministic():
    rows = [
        _candidate("b", 2, 40.0, "Second line of dialogue."),
        _candidate("a", 1, 40.0, "First line of dialogue."),
        _candidate("c", 1, 40.0, "Later distinct dialogue.", start_ms=50_000, end_ms=70_000),
    ]
    first = rank_candidates({"candidates": rows}, top_n=2, max_per_episode=2)
    second = rank_candidates({"candidates": list(reversed(rows))}, top_n=2, max_per_episode=2)
    assert json.dumps(first["recommendations"]) == json.dumps(second["recommendations"])
    assert [row["candidate_id"] for row in first["ranked_candidates"]] == ["a", "c", "b"]


def test_real_scored_file_shortlist_is_diverse_and_complete():
    payload = json.loads(SCORED.read_text(encoding="utf-8"))
    result = rank_candidates(payload, top_n=10, max_per_episode=2)
    assert result["summary"]["errors"] == []
    assert len(result["ranked_candidates"]) == len(payload["candidates"])
    assert len(result["recommendations"]) == 10
    overalls = [row["scores"]["overall"] for row in result["recommendations"]]
    assert overalls == sorted(overalls, reverse=True)
    counts: dict[int, int] = {}
    for row in result["recommendations"]:
        counts[row["episode_number"]] = counts.get(row["episode_number"], 0) + 1
        assert set(row["scores"]) == {
            "hook",
            "cliffhanger",
            "standalone_clarity",
            "emotional_intensity",
            "overall",
        }
        assert row["ranking_explanation"]
    assert all(count <= 2 for count in counts.values())
    assert len(counts) >= 5
    assert result["summary"]["excluded_count"] >= 1
    identities = {row["candidate_id"] for row in result["ranked_candidates"]}
    assert identities == {row["candidate_id"] for row in payload["candidates"]}
