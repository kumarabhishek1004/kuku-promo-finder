"""Tests for promo loading, filters, and reviewer decision persistence."""

import json

import pytest

from src.frontend_data import (
    DECISION_PLACEHOLDER,
    dashboard_counts,
    decision_widget_index,
    episode_distribution,
    filter_candidates,
    group_review_queue,
    load_candidates,
    load_decisions,
    read_json,
    recommended_promos,
    save_decision,
    save_is_enabled,
    score_distribution,
)


def _candidate(candidate_id, episode, overall, priority="low", status="below_shortlist", rank=None, family="ep01-f001"):
    return {
        "candidate_id": candidate_id,
        "episode_number": episode,
        "start": "00:00.00",
        "end": "00:20.00",
        "start_ms": 0,
        "end_ms": 20000,
        "duration_seconds": 20.0,
        "shortlist_status": status,
        "shortlist_rank": rank,
        "scores": {"overall": overall, "hook": 1, "cliffhanger": 1, "standalone_clarity": 1, "emotional_intensity": 1},
        "validation": {
            "priority": priority,
            "family": {"family_id": family},
            "inspection": {"in_queue": priority == "high"},
        },
        "dialogue": [{"speaker": "JULIANA", "text": '"Hello."', "start": "00:01.00", "end": "00:02.00"}],
        "action": [],
    }


def test_missing_and_malformed_json(tmp_path):
    missing, error = read_json(tmp_path / "absent.json")
    assert missing is None
    assert "not found" in error.lower()
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    payload, bad_error = read_json(bad)
    assert payload is None
    assert "not valid JSON" in bad_error
    rows, load_error = load_candidates(bad)
    assert rows == []
    assert load_error


def test_candidate_shapes_and_filters(tmp_path):
    payload = {
        "ranked_candidates": [
            _candidate("ep01-c0001", 1, 90, "high", "recommended", 1, "ep01-f001"),
            _candidate("ep02-c0004", 2, 40, "low", "below_shortlist", None, "ep02-f001"),
            _candidate("ep01-c0008", 1, 55, "medium", "excluded", None, "ep01-f002"),
        ]
    }
    path = tmp_path / "ranked.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows, error = load_candidates(path)
    assert error is None
    assert [row["candidate_id"] for row in recommended_promos(rows)] == ["ep01-c0001"]
    low = filter_candidates(rows, priority="low")
    assert [row["candidate_id"] for row in low] == ["ep02-c0004"]
    matched = filter_candidates(
        rows,
        text="c0008",
        episode=1,
        priority="medium",
        min_score=50,
        max_score=60,
        family="ep01-f002",
        ranking_status="excluded",
    )
    assert [row["candidate_id"] for row in matched] == ["ep01-c0008"]
    assert filter_candidates(rows, text="missing") == []
    counts = dashboard_counts(rows, {"items": [{"candidate_id": "ep01-c0001"}]})
    assert counts == {"total": 3, "high": 1, "medium": 1, "low": 1, "families": 3, "inspection_queue": 1}
    assert episode_distribution(rows) == [
        {"episode": 1, "candidates": 2},
        {"episode": 2, "candidates": 1},
    ]
    assert score_distribution(rows)[4]["candidates"] == 1
    assert score_distribution(rows)[5]["candidates"] == 1
    assert score_distribution(rows)[9]["candidates"] == 1


def test_review_queue_groups_representatives_before_alternatives():
    queue = {
        "items": [
            {"candidate_id": "alt", "family_id": "ep02-f001", "family_seed_id": "seed", "role": "alternative", "queue_position": 2},
            {"candidate_id": "rep", "family_id": "ep02-f001", "family_seed_id": "seed", "role": "representative", "queue_position": 1},
            {"candidate_id": "other", "family_id": "ep09-f001", "family_seed_id": "other", "role": "representative", "queue_position": 3},
        ]
    }
    grouped = group_review_queue(queue)
    assert [family["family_id"] for family in grouped] == ["ep02-f001", "ep09-f001"]
    assert [item["candidate_id"] for item in grouped[0]["items"]] == ["rep", "alt"]


def test_decisions_persist_and_do_not_touch_generated_files(tmp_path):
    decisions = tmp_path / "reviewer_decisions.json"
    generated = tmp_path / "validated_promos.json"
    generated.write_text('{"candidates": []}\n', encoding="utf-8")
    original = generated.read_text(encoding="utf-8")
    loaded, error = load_decisions(decisions)
    assert error is None
    assert loaded["decisions"] == {}
    saved = save_decision(decisions, "ep01-c0001", "HOLD", "Check the ending.", protected_paths=(generated,), clock=lambda: "2026-11-06T12:00:00+00:00")
    assert saved["decision"] == "HOLD"
    again, again_error = load_decisions(decisions)
    assert again_error is None
    assert again["decisions"]["ep01-c0001"]["comment"] == "Check the ending."
    save_decision(decisions, "ep01-c0001", "APPROVED", "", protected_paths=(generated,), clock=lambda: "2026-11-06T12:05:00+00:00")
    reloaded, _ = load_decisions(decisions)
    assert reloaded["decisions"]["ep01-c0001"]["decision"] == "APPROVED"
    assert reloaded["decisions"]["ep01-c0001"]["comment"] == ""
    assert generated.read_text(encoding="utf-8") == original
    with pytest.raises(ValueError):
        save_decision(generated, "ep01-c0001", "REJECTED", protected_paths=(generated,))
    with pytest.raises(ValueError):
        save_decision(decisions, "ep01-c0001", "MAYBE", protected_paths=())
    assert generated.read_text(encoding="utf-8") == original
    assert json.loads(decisions.read_text(encoding="utf-8"))["decisions"]["ep01-c0001"]["decision"] == "APPROVED"


def test_new_review_starts_with_no_decision_selected():
    assert decision_widget_index(None) is None
    assert decision_widget_index({}) is None
    assert decision_widget_index({"decision": DECISION_PLACEHOLDER}) is None
    assert save_is_enabled(None) is False
    assert save_is_enabled(DECISION_PLACEHOLDER) is False
    assert save_is_enabled("APPROVED") is True


def test_saved_decision_is_shown_and_reloaded(tmp_path):
    path = tmp_path / "reviewer_decisions.json"
    generated = tmp_path / "validated_promos.json"
    generated.write_text('{"candidates": []}\n', encoding="utf-8")
    before = generated.read_text(encoding="utf-8")
    assert decision_widget_index(None) is None
    saved = save_decision(
        path,
        "ep42-c0010",
        "REJECTED",
        "Cut is fine.",
        protected_paths=(generated,),
        clock=lambda: "2026-11-06T15:00:00+00:00",
    )
    loaded, error = load_decisions(path)
    record = loaded["decisions"]["ep42-c0010"]
    assert error is None
    assert record == saved
    assert decision_widget_index(record) == 1
    assert save_is_enabled(record["decision"]) is True
    assert generated.read_text(encoding="utf-8") == before


def test_priority_does_not_create_a_decision():
    rows = [_candidate("ep33-c0020", 33, 79, "low"), _candidate("ep36-c0038", 36, 83, "high", "recommended", 3)]
    counts = dashboard_counts(rows, {"items": []})
    assert counts["high"] == 1
    assert counts["low"] == 1
    assert all("decision" not in row for row in rows)
