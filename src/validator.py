"""Rule-based checks for scored and ranked promo candidates.

These checks flag overlap, cut points, and missing context. They do not change
scores, timestamps, or the ranking order, and they do not decide whether a
promo is emotionally compelling.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.event_detector import score_cue


HOOK_WINDOW_MS = 3_000
ENDING_WINDOW_MS = 3_000
FOLLOWUP_WINDOW_MS = 3_000
IOU_REVIEW = 0.50
SHORTER_OVERLAP_REVIEW = 0.60
TEXT_SIMILARITY_REVIEW = 0.60
TEXT_WITH_PARTIAL_OVERLAP = 0.40

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_PRONOUN_RE = re.compile(r"\b(?:he|she|her|him|his|hers|they|them|their|it)\b", re.IGNORECASE)
_PRIOR_RE = re.compile(
    r"\b(?:again|as before|as i said|last time|like before|remember|that night|you know)\b",
    re.IGNORECASE,
)
_OPENING_CONTINUATION = ("then ", "but ", "so ", "again ")
_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_NOT_NAMES = {
    "And",
    "But",
    "Her",
    "His",
    "How",
    "She",
    "The",
    "Then",
    "They",
    "Them",
    "That",
    "This",
    "What",
    "When",
    "Where",
    "Who",
    "Why",
    "You",
    "Your",
}
_METHOD_NOTE = (
    "Lexical and timestamp rules only. They do not establish that a promo is emotionally compelling."
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_candidates(payload: dict | list | None, parsed: dict | None = None) -> dict:
    """Validate every candidate. Input records are not modified or removed."""
    rows, error = _rows(payload)
    spine = _episode_spine(parsed)
    prepared = [_copy_candidate(row) if isinstance(row, dict) else _invalid_candidate() for row in rows]
    pairs = _duplicate_pairs(prepared)
    _attach_duplicate_warnings(prepared, pairs)
    validated = []
    for candidate in prepared:
        following = _following_cue(candidate, spine)
        validation = _validate_one(candidate, following)
        candidate.pop("_tokens", None)
        candidate["validation"] = validation
        validated.append(candidate)
    return {
        "candidates": validated,
        "duplicate_pairs": pairs,
        "summary": _summary(validated, pairs, error),
    }


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Write validation results for the ranked promo list."""
    root = repo_root()
    argument_parser = argparse.ArgumentParser(description="Validate ranked promo candidates.")
    argument_parser.add_argument("--ranked", default=str(root / "outputs" / "ranked_promos.json"))
    argument_parser.add_argument("--parsed", default=str(root / "outputs" / "parsed_episodes.json"))
    argument_parser.add_argument("--output", default=str(root / "outputs" / "validated_promos.json"))
    argument_parser.add_argument("--summary", default=str(root / "outputs" / "validation_summary.json"))
    args = argument_parser.parse_args(argv)
    ranked_path = Path(args.ranked)
    if not ranked_path.is_file():
        print(f"Ranked promo file not found: {ranked_path}")
        return 1
    parsed = load_json(Path(args.parsed)) if Path(args.parsed).is_file() else None
    result = validate_candidates(load_json(ranked_path), parsed)
    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = result["summary"]
    print(
        f"Validated {summary['candidate_count']} candidates. "
        f"Duplicate pairs {summary['duplicate_pair_count']}. "
        f"Human review {summary['human_review_count']}."
    )
    print(f"Wrote {output}")
    print(f"Wrote {summary_path}")
    return 0


def _validate_one(candidate: dict, following: dict | None) -> dict:
    opening = _opening_assessment(candidate)
    ending = _ending_assessment(candidate, following)
    clarity = _clarity_assessment(candidate)
    flags = []
    flags.extend(opening["flags"])
    flags.extend(ending["flags"])
    flags.extend(clarity["flags"])
    if _overlaps_equal_or_stronger(candidate):
        flags.append("duplicate_substantial_overlap")
    review_reasons = list(flags)
    return {
        "flags": flags,
        "duplicate_warnings": list(candidate.get("duplicate_warnings") or []),
        "opening": opening,
        "ending": ending,
        "clarity": clarity,
        "human_review": {
            "recommended": bool(review_reasons),
            "reasons": review_reasons,
            "note": _METHOD_NOTE,
        },
        "scores_unchanged": True,
        "timestamps_unchanged": True,
    }


def _opening_assessment(candidate: dict) -> dict:
    start = candidate.get("start_ms")
    cues = _window_cues(candidate, start, None if not isinstance(start, int) else start + HOOK_WINDOW_MS)
    flags = []
    evidence = [_evidence(cue, "Cue overlaps the first 3 seconds.") for cue in cues]
    if not isinstance(start, int):
        flags.append("opening_timestamps_missing")
        return _assessment(flags, evidence, {"has_meaningful_dialogue_or_action": False})
    meaningful = [cue for cue in cues if _meaningful(cue)]
    if not meaningful:
        flags.append("opening_no_meaningful_cue")
    spoken = _spoken(_first_text(cues))
    mid_sentence = _starts_mid_sentence(spoken)
    depends_on_prior = spoken.lower().startswith(_OPENING_CONTINUATION)
    if mid_sentence:
        flags.append("opening_mid_sentence")
    if depends_on_prior:
        flags.append("opening_depends_on_prior_event")
    situation = _situation_label(cues)
    if situation == "unclear" and meaningful:
        flags.append("opening_situation_unclear")
    return _assessment(
        flags,
        evidence,
        {
            "has_meaningful_dialogue_or_action": bool(meaningful),
            "starts_mid_sentence": mid_sentence,
            "depends_on_unseen_event": depends_on_prior,
            "situation_indicator": situation,
        },
    )


def _ending_assessment(candidate: dict, following: dict | None) -> dict:
    end = candidate.get("end_ms")
    cues = _window_cues(candidate, None if not isinstance(end, int) else end - ENDING_WINDOW_MS, end)
    flags = []
    evidence = [_evidence(cue, "Cue overlaps the final 3 seconds.") for cue in cues]
    if not isinstance(end, int):
        flags.append("ending_timestamps_missing")
        return _assessment(flags, evidence, {})
    last_dialogue = _last_dialogue(cues)
    spoken = _spoken(last_dialogue.get("text") if last_dialogue else "")
    mid_sentence = _ends_mid_sentence(spoken)
    unresolved = spoken.rstrip().endswith("?") or _has_signal(last_dialogue, {"suspenseful_question", "interrupted_dialogue"})
    threat_or_reveal = _has_signal(last_dialogue, {"threat", "reveal", "physical_action"})
    explains = _explains_rather_than_withholds(spoken)
    if mid_sentence:
        flags.append("ending_mid_sentence")
    if explains and not unresolved and not mid_sentence:
        flags.append("ending_resolves_or_explains")
    if not cues:
        flags.append("ending_no_dialogue_or_action")
    follow = _followup_record(following, last_dialogue)
    follow_exclamation = bool(follow.get("present") and _spoken(follow.get("text")).rstrip().endswith("!"))
    if follow["present"] and (follow["heuristic_stronger"] or (explains and follow_exclamation)):
        flags.append("stronger_beat_after_cut")
        evidence.append(follow["evidence"])
    return _assessment(
        flags,
        evidence,
        {
            "final_text": spoken or None,
            "ends_mid_sentence": mid_sentence,
            "unresolved_question_threat_reveal_or_confrontation": bool(unresolved or threat_or_reveal),
            "explains_rather_than_withholds": explains and not unresolved and not mid_sentence,
            "following_line": follow,
        },
    )


def _clarity_assessment(candidate: dict) -> dict:
    dialogue = _typed(candidate, "DIALOGUE")
    spoken_lines = [_spoken(cue.get("text")) for cue in dialogue if _spoken(cue.get("text"))]
    blob = " ".join(spoken_lines)
    pronouns = _PRONOUN_RE.findall(blob)
    names = []
    for line in spoken_lines:
        remainder = line.split(None, 1)[1] if " " in line else ""
        names.extend(name for name in _NAME_RE.findall(remainder) if name not in _NOT_NAMES)
    prior_hits = []
    for line in spoken_lines:
        prior_hits.extend(match.group(0).lower() for match in _PRIOR_RE.finditer(line))
    flags = []
    unexplained = len(pronouns) >= 4 and len(set(names)) <= 1
    if unexplained:
        flags.append("unexplained_pronouns")
    if prior_hits:
        flags.append("previous_event_reference")
    return _assessment(
        flags,
        [{"text": line, "note": "Spoken line counted for clarity."} for line in spoken_lines[:6]],
        {
            "pronoun_count": len(pronouns),
            "distinct_capitalized_names": sorted(set(names)),
            "previous_event_phrases": sorted(set(prior_hits)),
            "relies_on_unexplained_reference": bool(flags),
        },
    )


def _assessment(flags: list[str], evidence: list[dict], details: dict) -> dict:
    return {
        "method": "lexical_and_timestamp_rules",
        "limitation": _METHOD_NOTE,
        "flags": flags,
        "evidence": evidence,
        **details,
    }


def _duplicate_pairs(candidates: list[dict]) -> list[dict]:
    grouped: dict[object, list[dict]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.get("episode_number"), []).append(candidate)
    pairs = []
    for episode, group in grouped.items():
        ordered = sorted(group, key=lambda item: (item.get("candidate_id") or "", item.get("start_ms") or 0))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                pair = _pair_if_overlap(episode, left, right)
                if pair is not None:
                    pairs.append(pair)
    return pairs


def _pair_if_overlap(episode: object, left: dict, right: dict) -> dict | None:
    iou, shorter_ratio, intersection = _overlap_metrics(left, right)
    similarity = _jaccard(left.get("_tokens") or set(), right.get("_tokens") or set())
    shared_scenes = sorted(set(_scene_numbers(left)) & set(_scene_numbers(right)))
    reasons = []
    if iou >= IOU_REVIEW:
        reasons.append("iou_at_least_0.50")
    if shorter_ratio >= SHORTER_OVERLAP_REVIEW:
        reasons.append("shorter_clip_overlap_at_least_0.60")
    if similarity >= TEXT_SIMILARITY_REVIEW and shorter_ratio >= TEXT_WITH_PARTIAL_OVERLAP:
        reasons.append("text_similarity_with_partial_overlap")
    if similarity >= 0.80 and shared_scenes:
        reasons.append("text_similarity_with_shared_scene")
    if not reasons:
        return None
    return {
        "episode_number": episode,
        "candidate_ids": [left.get("candidate_id"), right.get("candidate_id")],
        "shared_scenes": shared_scenes,
        "iou": round(iou, 4),
        "shorter_overlap": round(shorter_ratio, 4),
        "intersection_ms": intersection,
        "text_similarity": round(similarity, 4),
        "reasons": reasons,
        "human_review": True,
        "discarded": False,
    }


def _attach_duplicate_warnings(candidates: list[dict], pairs: list[dict]) -> None:
    by_id = {candidate.get("candidate_id"): candidate for candidate in candidates}
    for pair in pairs:
        for candidate_id in pair["candidate_ids"]:
            candidate = by_id.get(candidate_id)
            if candidate is None:
                continue
            other = [item for item in pair["candidate_ids"] if item != candidate_id]
            other_id = other[0] if other else None
            other_candidate = by_id.get(other_id) or {}
            other_scores = other_candidate.get("scores") if isinstance(other_candidate.get("scores"), dict) else {}
            candidate.setdefault("duplicate_warnings", []).append(
                {
                    "other_candidate_id": other_id,
                    "other_overall": other_scores.get("overall"),
                    "iou": pair["iou"],
                    "shorter_overlap": pair["shorter_overlap"],
                    "text_similarity": pair["text_similarity"],
                    "shared_scenes": pair["shared_scenes"],
                    "reasons": list(pair["reasons"]),
                    "discarded": False,
                }
            )


def _overlaps_equal_or_stronger(candidate: dict) -> bool:
    own = (candidate.get("scores") or {}).get("overall") if isinstance(candidate.get("scores"), dict) else None
    for warning in candidate.get("duplicate_warnings") or []:
        other = warning.get("other_overall")
        if own is None or other is None or other >= own:
            return True
    return False


def _overlap_metrics(left: dict, right: dict) -> tuple[float, float, int]:
    values = (left.get("start_ms"), left.get("end_ms"), right.get("start_ms"), right.get("end_ms"))
    if not all(isinstance(value, int) for value in values):
        return 0.0, 0.0, 0
    start_left, end_left, start_right, end_right = values
    intersection = min(end_left, end_right) - max(start_left, start_right)
    if intersection <= 0:
        return 0.0, 0.0, 0
    union = (end_left - start_left) + (end_right - start_right) - intersection
    shorter = min(end_left - start_left, end_right - start_right)
    iou = intersection / union if union > 0 else 0.0
    shorter_ratio = intersection / shorter if shorter > 0 else 0.0
    return iou, shorter_ratio, intersection


def _episode_spine(parsed: dict | None) -> dict[object, list[dict]]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("episodes"), list):
        return {}
    spine: dict[object, list[dict]] = {}
    for episode in parsed["episodes"]:
        if not isinstance(episode, dict):
            continue
        cues = []
        for scene in episode.get("scenes") or []:
            if not isinstance(scene, dict):
                continue
            for cue in scene.get("cues") or []:
                if not isinstance(cue, dict) or cue.get("type") not in {"DIALOGUE", "ACTION"}:
                    continue
                if not isinstance(cue.get("start_ms"), int) or not isinstance(cue.get("end_ms"), int):
                    continue
                copied = dict(cue)
                copied["scene_number"] = scene.get("scene_number")
                cues.append(copied)
        cues.sort(key=lambda cue: (cue["start_ms"], cue.get("line") or 0))
        spine[episode.get("episode_number")] = cues
    return spine


def _following_cue(candidate: dict, spine: dict[object, list[dict]]) -> dict | None:
    end = candidate.get("end_ms")
    if not isinstance(end, int):
        return None
    included = {
        (cue.get("start_ms"), _spoken(cue.get("text")))
        for cue in _typed(candidate, "DIALOGUE") + _typed(candidate, "ACTION")
    }
    for cue in spine.get(candidate.get("episode_number"), []):
        if cue["start_ms"] < end or cue["start_ms"] > end + FOLLOWUP_WINDOW_MS:
            continue
        if (cue.get("start_ms"), _spoken(cue.get("text"))) in included:
            continue
        return cue
    return None


def _followup_record(following: dict | None, last_dialogue: dict | None) -> dict:
    if following is None:
        return {"present": False, "heuristic_stronger": False, "evidence": None}
    follow_score = score_cue(following)["score"]
    last_score = score_cue(last_dialogue)["score"] if last_dialogue else 0
    stronger = follow_score > last_score or (
        _ends_mid_sentence(_spoken(last_dialogue.get("text") if last_dialogue else ""))
        and _starts_mid_sentence(_spoken(following.get("text")))
    )
    return {
        "present": True,
        "heuristic_stronger": stronger,
        "start": following.get("start"),
        "end": following.get("end"),
        "start_ms": following.get("start_ms"),
        "text": following.get("text"),
        "lexical_score": follow_score,
        "evidence": _evidence(following, "Next dialogue or action starts within 3 seconds after the cut."),
    }


def _window_cues(candidate: dict, start: int | None, end: int | None) -> list[dict]:
    cues = []
    for cue in _typed(candidate, "DIALOGUE") + _typed(candidate, "ACTION"):
        cue_start = cue.get("start_ms")
        cue_end = cue.get("end_ms")
        if not isinstance(cue_start, int) or not isinstance(cue_end, int):
            continue
        if start is not None and cue_end <= start:
            continue
        if end is not None and cue_start >= end:
            continue
        cues.append(cue)
    cues.sort(key=lambda cue: (cue.get("start_ms") or 0, cue.get("line") or 0))
    return cues


def _last_dialogue(cues: list[dict]) -> dict | None:
    dialogue = [cue for cue in cues if cue.get("type") == "DIALOGUE"]
    if not dialogue:
        return None
    return max(dialogue, key=lambda cue: (cue.get("end_ms") or 0, cue.get("line") or 0))


def _meaningful(cue: dict) -> bool:
    words = _spoken(cue.get("text")).split()
    if cue.get("type") == "ACTION":
        return len(_spoken(cue.get("text"))) >= 30
    return len(words) >= 2


def _situation_label(cues: list[dict]) -> str:
    if any(_has_signal(cue, {"threat", "reveal", "physical_action", "intense_delivery"}) for cue in cues):
        return "conflict_language"
    if any(_spoken(cue.get("text")).rstrip().endswith("?") for cue in cues if cue.get("type") == "DIALOGUE"):
        return "question"
    if any(cue.get("type") == "ACTION" and _meaningful(cue) for cue in cues):
        return "visible_action"
    if any(_meaningful(cue) for cue in cues):
        return "spoken_exchange"
    return "unclear"


def _explains_rather_than_withholds(spoken: str) -> bool:
    lowered = spoken.lower()
    if not lowered or lowered.rstrip().endswith("?") or _ends_mid_sentence(spoken):
        return False
    phrases = ("again and again", "small amounts", "enough to", "the truth", "i told them")
    return any(phrase in lowered for phrase in phrases)


def _has_signal(cue: dict | None, categories: set[str]) -> bool:
    if not isinstance(cue, dict):
        return False
    return bool(categories.intersection(score_cue(cue).get("categories") or []))


def _starts_mid_sentence(spoken: str) -> bool:
    text = spoken.lstrip()
    if not text:
        return False
    if text.startswith("...") or text.startswith("\u2026") or text.startswith("—") or text.startswith("\u2014"):
        return True
    return text[:1].islower()


def _ends_mid_sentence(spoken: str) -> bool:
    text = spoken.rstrip()
    return text.endswith("...") or text.endswith("\u2026") or text.endswith("—") or text.endswith("\u2014")


def _first_text(cues: list[dict]) -> str:
    for cue in cues:
        if cue.get("type") == "DIALOGUE" and _spoken(cue.get("text")):
            return cue.get("text") or ""
    return cues[0].get("text") if cues else ""


def _typed(candidate: dict, cue_type: str) -> list[dict]:
    value = candidate.get("dialogue" if cue_type == "DIALOGUE" else "action")
    if not isinstance(value, list):
        return []
    return [cue for cue in value if isinstance(cue, dict)]


def _scene_numbers(candidate: dict) -> list[int]:
    scenes = candidate.get("scenes")
    if not isinstance(scenes, list):
        return []
    return [scene.get("scene_number") for scene in scenes if isinstance(scene, dict) and isinstance(scene.get("scene_number"), int)]


def _tokens(candidate: dict) -> set[str]:
    parts = []
    for key in ("dialogue", "action"):
        value = candidate.get(key)
        if not isinstance(value, list):
            continue
        for cue in value:
            if isinstance(cue, dict) and isinstance(cue.get("text"), str):
                parts.append(cue["text"])
    return set(_TOKEN_RE.findall(" ".join(parts).lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _spoken(text: object) -> str:
    if not isinstance(text, str):
        return ""
    spoken = text.strip()
    if len(spoken) >= 2 and spoken[0] == '"' and spoken[-1] == '"':
        return spoken[1:-1]
    return spoken


def _evidence(cue: dict, note: str) -> dict:
    return {
        "text": cue.get("text"),
        "type": cue.get("type"),
        "start": cue.get("start"),
        "end": cue.get("end"),
        "start_ms": cue.get("start_ms"),
        "end_ms": cue.get("end_ms"),
        "note": note,
    }


def _copy_candidate(row: dict) -> dict:
    candidate = dict(row)
    scores = row.get("scores")
    if isinstance(scores, dict):
        candidate["scores"] = dict(scores)
    candidate["_tokens"] = _tokens(candidate)
    return candidate


def _invalid_candidate() -> dict:
    return {
        "candidate_id": None,
        "scores": {},
        "duplicate_warnings": [],
        "validation_input_error": "Candidate was not a dictionary.",
        "_tokens": set(),
    }


def _rows(payload: dict | list | None) -> tuple[list, str | None]:
    if payload is None:
        return [], "Input is empty."
    if isinstance(payload, list):
        return payload, None
    if not isinstance(payload, dict):
        return [], "Input must be a dictionary or a list."
    rows = payload.get("ranked_candidates")
    if not isinstance(rows, list):
        rows = payload.get("candidates")
    if rows is None:
        return [], "Input has no candidates."
    if not isinstance(rows, list):
        return [], "Candidates must be a list."
    return rows, None


def _summary(candidates: list[dict], pairs: list[dict], error: str | None) -> dict:
    flag_counts: dict[str, int] = {}
    review = []
    for candidate in candidates:
        validation = candidate.get("validation") or {}
        for flag in validation.get("flags") or []:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        human = validation.get("human_review") or {}
        if human.get("recommended"):
            review.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "episode_number": candidate.get("episode_number"),
                    "start": candidate.get("start"),
                    "end": candidate.get("end"),
                    "shortlist_rank": candidate.get("shortlist_rank"),
                    "reasons": human.get("reasons") or [],
                }
            )
    shortlist = [item for item in review if item.get("shortlist_rank")]
    shortlist_ids = {item["candidate_id"] for item in candidates if item.get("shortlist_rank")}
    return {
        "candidate_count": len(candidates),
        "duplicate_pair_count": len(pairs),
        "human_review_count": len(review),
        "flag_counts": flag_counts,
        "shortlist_duplicate_pairs": [
            pair
            for pair in pairs
            if set(pair["candidate_ids"]).issubset(shortlist_ids)
        ],
        "shortlist_review": sorted(shortlist, key=lambda item: item.get("shortlist_rank") or 0),
        "thresholds": {
            "iou_review": IOU_REVIEW,
            "shorter_overlap_review": SHORTER_OVERLAP_REVIEW,
            "text_similarity_review": TEXT_SIMILARITY_REVIEW,
            "followup_window_ms": FOLLOWUP_WINDOW_MS,
        },
        "limitation": _METHOD_NOTE,
        "errors": [error] if error else [],
    }


if __name__ == "__main__":
    raise SystemExit(main())
