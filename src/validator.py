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
    families = _assign_families(validated)
    queue = _build_inspection_queue(families)
    return {
        "candidates": validated,
        "duplicate_pairs": pairs,
        "inspection_queue": queue,
        "summary": _summary(validated, pairs, families, queue, error),
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
    argument_parser.add_argument("--queue", default=str(root / "outputs" / "human_review_queue.json"))
    args = argument_parser.parse_args(argv)
    ranked_path = Path(args.ranked)
    if not ranked_path.is_file():
        print(f"Ranked promo file not found: {ranked_path}")
        return 1
    parsed = load_json(Path(args.parsed)) if Path(args.parsed).is_file() else None
    result = validate_candidates(load_json(ranked_path), parsed)
    output = Path(args.output)
    summary_path = Path(args.summary)
    queue_path = Path(args.queue)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    queue_path.write_text(
        json.dumps(result["inspection_queue"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = result["summary"]
    print(
        f"Validated {summary['candidate_count']} candidates. "
        f"Families {summary['family_count']}. "
        f"Inspection queue {summary['inspection_queue_count']}."
    )
    print(f"Wrote {output}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {queue_path}")
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
    reasons = _priority_reasons(candidate, opening, ending, clarity)
    priority = _highest_priority(reasons)
    return {
        "flags": flags,
        "priority": priority,
        "priority_reasons": reasons,
        "duplicate_warnings": list(candidate.get("duplicate_warnings") or []),
        "opening": opening,
        "ending": ending,
        "clarity": clarity,
        "family": None,
        "inspection": {
            "in_queue": False,
            "role": None,
            "covered_by": None,
            "same_ending_candidate_ids": [],
        },
        "human_review": {
            "recommended": False,
            "priority": priority,
            "advisory": priority == "medium",
            "reasons": [reason["code"] for reason in reasons if reason["level"] == priority],
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
        "last_lexical_score": last_score,
        "lexical_score_difference": follow_score - last_score,
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


_LEVEL_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
_HARD_CALLBACKS = {"again", "last time", "as before", "as i said", "like before", "that night"}
_RHETORICAL_CALLBACKS = {"you know", "remember"}


def _reason(code: str, level: str, status: str, text: str) -> dict:
    return {"code": code, "level": level, "status": status, "review_reason": text}


def _priority_reasons(candidate: dict, opening: dict, ending: dict, clarity: dict) -> list[dict]:
    """Separate confirmed cut defects from heuristic warnings and informational notes."""
    reasons = []
    final = ending.get("final_text") or ""
    follow = ending.get("following_line") or {}
    next_line = _spoken(follow.get("text") or "")
    continued = bool(follow.get("present") and _starts_mid_sentence(next_line))
    split = _ends_mid_sentence(final) and continued
    words = _core_words(final)
    if split:
        reasons.append(
            _reason(
                "sentence_split",
                "high",
                "confirmed",
                "The final line is cut off and the next screenplay line continues it.",
            )
        )
    if "ending_resolves_or_explains" in ending.get("flags", []):
        reasons.append(
            _reason(
                "ending_explains",
                "high",
                "heuristic",
                "Phrase match suggests the ending states the answer. This is a warning, not proof the promo fails.",
            )
        )
    if follow.get("heuristic_stronger") and final.rstrip().endswith("?") and not split:
        reasons.append(
            _reason(
                "question_then_stronger_line",
                "high",
                "heuristic",
                "The ending is a question and the next cue has a higher keyword score. This is a warning, not proof the cut is wrong.",
            )
        )
    difference = follow.get("lexical_score_difference")
    if (
        follow.get("present")
        and not split
        and not final.rstrip().endswith("?")
        and isinstance(difference, int)
        and difference >= 5
    ):
        reasons.append(
            _reason(
                "later_line_much_hotter",
                "medium",
                "heuristic",
                "The next cue scores at least 5 keyword points higher. This is a warning, not a confirmed defect.",
            )
        )
    if "opening_no_meaningful_cue" in opening.get("flags", []):
        reasons.append(
            _reason(
                "thin_opening",
                "medium",
                "heuristic",
                "The first 3 seconds lack a line of at least two words or an action of at least 30 characters.",
            )
        )
    if "opening_mid_sentence" in opening.get("flags", []):
        reasons.append(
            _reason(
                "opening_ellipsis",
                "medium",
                "heuristic",
                "The first spoken line starts with an ellipsis. Judge the rest of the clip before treating this as a bad opening.",
            )
        )
    if _ends_mid_sentence(final) and follow.get("present") and not continued and len(words) > 2:
        reasons.append(
            _reason(
                "long_ellipsis_new_sentence",
                "medium",
                "heuristic",
                "A longer ending ellipsis is followed by a new sentence. This may be a pause rather than a broken sentence.",
            )
        )
    phrases = set(clarity.get("previous_event_phrases") or [])
    if phrases & _HARD_CALLBACKS:
        reasons.append(
            _reason(
                "hard_callback",
                "medium",
                "heuristic",
                "The clip uses a callback such as 'again' or 'last time'. Keyword presence is not story understanding.",
            )
        )
    if phrases & _RHETORICAL_CALLBACKS:
        reasons.append(
            _reason(
                "rhetorical_callback",
                "low",
                "informational",
                "'You know' or 'remember' appears. This is informational and does not require review.",
            )
        )
    if "unexplained_pronouns" in clarity.get("flags", []):
        reasons.append(
            _reason(
                "unexplained_pronouns",
                "low",
                "informational",
                "Pronouns are common in this clip. The count does not show that the scene is unclear.",
            )
        )
    if _ends_mid_sentence(final) and 0 < len(words) <= 2:
        reasons.append(
            _reason(
                "short_address_ellipsis",
                "low",
                "informational",
                "The ending is a one- or two-word address or exclamation with an ellipsis, such as a name.",
            )
        )
    if "opening_depends_on_prior_event" in opening.get("flags", []):
        reasons.append(
            _reason(
                "opening_conjunction",
                "low",
                "informational",
                "The opening starts with Then, But, So, or Again. That is not a failure when the rest of the clip can be read.",
            )
        )
    if candidate.get("duplicate_warnings"):
        reasons.append(
            _reason(
                "duplicate_overlap",
                "low",
                "informational",
                "This clip overlaps another window in the same episode. It stays in its promo family and is not discarded.",
            )
        )
    return reasons


def _highest_priority(reasons: list[dict]) -> str:
    if not reasons:
        return "none"
    return max(reasons, key=lambda reason: _LEVEL_RANK[reason["level"]])["level"]


def _core_words(spoken: str) -> list[str]:
    text = spoken.replace("...", " ").replace("\u2026", " ").replace("\u2014", " ").replace("—", " ")
    return [word for word in text.split() if word]


def _assign_families(candidates: list[dict]) -> list[list[dict]]:
    """Greedy seeds. A clip joins the first same-episode seed it substantially overlaps."""
    ordered = sorted(candidates, key=_family_sort_key)
    families: list[list[dict]] = []
    for candidate in ordered:
        placed = False
        for group in families:
            seed = group[0]
            if seed.get("episode_number") != candidate.get("episode_number"):
                continue
            iou, shorter_ratio, _intersection = _overlap_metrics(seed, candidate)
            if iou >= IOU_REVIEW or shorter_ratio >= SHORTER_OVERLAP_REVIEW:
                group.append(candidate)
                placed = True
                break
        if not placed:
            families.append([candidate])
    counters: dict[object, int] = {}
    for group in families:
        episode = group[0].get("episode_number")
        counters[episode] = counters.get(episode, 0) + 1
        family_id = _family_id(episode, counters[episode])
        seed_id = group[0].get("candidate_id")
        for member in group:
            member["validation"]["family"] = {
                "family_id": family_id,
                "seed_candidate_id": seed_id,
                "is_seed": member.get("candidate_id") == seed_id,
                "member_count": len(group),
                "high_representative_id": None,
            }
    return families


def _family_sort_key(candidate: dict) -> tuple:
    overall = (candidate.get("scores") or {}).get("overall") if isinstance(candidate.get("scores"), dict) else None
    score = overall if isinstance(overall, (int, float)) else -1
    episode = candidate.get("episode_number")
    episode_key = episode if isinstance(episode, int) else 10**9
    start = candidate.get("start_ms")
    start_key = start if isinstance(start, int) else 10**12
    return (-score, episode_key, start_key, candidate.get("candidate_id") or "")


def _family_id(episode: object, index: int) -> str:
    if isinstance(episode, int):
        return f"ep{episode:02d}-f{index:03d}"
    return f"ep-{episode}-f{index:03d}"


def _build_inspection_queue(families: list[list[dict]]) -> dict:
    """One inspection per distinct high-priority ending inside each family."""
    items = []
    covered = []
    for group in families:
        high_members = [member for member in group if member["validation"]["priority"] == "high"]
        if not high_members:
            continue
        buckets: dict[str, list[dict]] = {}
        for member in high_members:
            buckets.setdefault(_final_text(member), []).append(member)
        representative = max(high_members, key=_inspection_sort_key)
        representative_final = _final_text(representative)
        family = representative["validation"]["family"]
        for member in group:
            member["validation"]["family"]["high_representative_id"] = representative.get("candidate_id")
        for final, members in buckets.items():
            selected = max(members, key=_inspection_sort_key)
            others = [member.get("candidate_id") for member in members if member is not selected]
            role = "representative" if final == representative_final else "alternative"
            inspection = selected["validation"]["inspection"]
            inspection["in_queue"] = True
            inspection["role"] = role
            inspection["same_ending_candidate_ids"] = others
            selected["validation"]["human_review"]["recommended"] = True
            item = {
                "candidate_id": selected.get("candidate_id"),
                "episode_number": selected.get("episode_number"),
                "family_id": family["family_id"],
                "family_seed_id": family["seed_candidate_id"],
                "role": role,
                "priority": "high",
                "reasons": [reason for reason in selected["validation"]["priority_reasons"] if reason["level"] == "high"],
                "final_text": final,
                "start": selected.get("start"),
                "end": selected.get("end"),
                "start_ms": selected.get("start_ms"),
                "end_ms": selected.get("end_ms"),
                "scores": dict(selected.get("scores") or {}),
                "same_ending_candidate_ids": others,
            }
            items.append(item)
            for member in members:
                if member is selected:
                    continue
                member["validation"]["inspection"]["covered_by"] = selected.get("candidate_id")
                member["validation"]["inspection"]["same_ending_candidate_ids"] = [
                    selected.get("candidate_id")
                ]
                covered.append(
                    {
                        "candidate_id": member.get("candidate_id"),
                        "covered_by": selected.get("candidate_id"),
                        "family_id": family["family_id"],
                        "final_text": final,
                    }
                )
    items.sort(key=lambda item: (item["episode_number"] if isinstance(item["episode_number"], int) else 10**9, item["family_id"], 0 if item["role"] == "representative" else 1, item["candidate_id"] or ""))
    for position, item in enumerate(items, start=1):
        item["queue_position"] = position
    representatives = sum(1 for item in items if item["role"] == "representative")
    return {
        "inspection_count": len(items),
        "representative_count": representatives,
        "alternative_count": len(items) - representatives,
        "same_ending_not_separately_queued": covered,
        "items": items,
        "note": "Queue size is calculated from the input. Candidates that share a final line stay recorded and are not deleted.",
    }


def _inspection_sort_key(candidate: dict) -> tuple:
    overall = (candidate.get("scores") or {}).get("overall") if isinstance(candidate.get("scores"), dict) else None
    score = overall if isinstance(overall, (int, float)) else -1
    start = candidate.get("start_ms")
    start_key = start if isinstance(start, int) else 10**12
    return (score, -start_key, candidate.get("candidate_id") or "")


def _final_text(candidate: dict) -> str:
    return candidate.get("validation", {}).get("ending", {}).get("final_text") or ""


def _summary(candidates: list[dict], pairs: list[dict], families: list[list[dict]], queue: dict, error: str | None) -> dict:
    flag_counts: dict[str, int] = {}
    priority_counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
    high_families = 0
    for group in families:
        if any(member["validation"]["priority"] == "high" for member in group):
            high_families += 1
    shortlist = []
    for candidate in candidates:
        validation = candidate.get("validation") or {}
        priority_counts[validation.get("priority") or "none"] += 1
        for flag in validation.get("flags") or []:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        if candidate.get("shortlist_rank"):
            shortlist.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "episode_number": candidate.get("episode_number"),
                    "shortlist_rank": candidate.get("shortlist_rank"),
                    "priority": validation.get("priority"),
                    "in_inspection_queue": (validation.get("inspection") or {}).get("in_queue"),
                    "family_id": (validation.get("family") or {}).get("family_id"),
                    "reasons": (validation.get("human_review") or {}).get("reasons") or [],
                }
            )
    shortlist_ids = {item["candidate_id"] for item in shortlist}
    return {
        "candidate_count": len(candidates),
        "duplicate_pair_count": len(pairs),
        "priority_counts": priority_counts,
        "family_count": len(families),
        "high_priority_family_count": high_families,
        "inspection_queue_count": queue["inspection_count"],
        "representative_count": queue["representative_count"],
        "alternative_cut_count": queue["alternative_count"],
        "same_ending_not_separately_queued_count": len(queue["same_ending_not_separately_queued"]),
        "human_review_count": queue["inspection_count"],
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
            "hotter_line_score_difference": 5,
        },
        "limitation": _METHOD_NOTE,
        "errors": [error] if error else [],
    }


if __name__ == "__main__":
    raise SystemExit(main())
