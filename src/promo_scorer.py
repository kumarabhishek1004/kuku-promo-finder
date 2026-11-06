"""Score promo candidates on four lexical dimensions.

The scores use the cue text, parentheticals, and timestamps already stored on
each candidate. They do not infer plot events that are not written in the
window, and they do not drop or rank candidates.
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


HOOK_WEIGHT = 0.35
CLIFFHANGER_WEIGHT = 0.30
CLARITY_WEIGHT = 0.20
INTENSITY_WEIGHT = 0.15
HOOK_WINDOW_MS = 3_000
ENDING_WINDOW_MS = 3_000

_CALLBACKS = (
    "again",
    "as before",
    "as i said",
    "last time",
    "like before",
    "remember",
    "still",
    "that night",
    "the proof",
    "you know",
)
_RESOLUTIONS = (
    "goodbye",
    "good bye",
    "thank you",
    "i love you",
    "we begin again",
    "it is done",
    "it is over",
)
_INCOMPLETE_ACTION = (
    "about to",
    "begins",
    "before she",
    "before he",
    "reaches for",
    "starts to",
    "turns toward",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def weighted_overall(hook: float, cliffhanger: float, clarity: float, intensity: float) -> float:
    """Map four 0–10 scores to 0–100 using the milestone weights.

    Hook is 35%, cliffhanger 30%, standalone clarity 20%, and emotional
    intensity 15%. Values outside 0–10 are clamped before the sum.
    """
    hook = _clamp_dimension(hook)
    cliffhanger = _clamp_dimension(cliffhanger)
    clarity = _clamp_dimension(clarity)
    intensity = _clamp_dimension(intensity)
    total = hook * 35 + cliffhanger * 30 + clarity * 20 + intensity * 15
    return round(total / 10, 1)


def score_candidate(candidate: dict | None) -> dict:
    """Score one candidate. The input dictionary is not modified."""
    if not isinstance(candidate, dict):
        return _empty_score(None, ["invalid_candidate"])
    scored = dict(candidate)
    dialogue = _cues(candidate.get("dialogue"))
    action = _cues(candidate.get("action"))
    camera = _cues(candidate.get("camera"))
    spine = sorted(dialogue + action, key=_cue_order)
    skipped = _skipped_cue_count(candidate)
    start_ms = candidate.get("start_ms")
    end_ms = candidate.get("end_ms")
    flags: list[str] = []
    if not isinstance(start_ms, int) or not isinstance(end_ms, int) or end_ms < start_ms:
        flags.append("incomplete_timestamps")
        start_ms = None
        end_ms = None
    if not dialogue:
        flags.append("no_dialogue")
    if not action:
        flags.append("no_action")
    if not spine:
        flags.append("no_spine_cues")
    if skipped:
        flags.append("skipped_incomplete_cues")

    hook = _score_hook(spine, start_ms, end_ms, flags)
    cliffhanger = _score_cliffhanger(spine, start_ms, end_ms, flags)
    clarity = _score_clarity(dialogue, action, camera, flags)
    intensity = _score_intensity(spine)
    overall = weighted_overall(hook["score"], cliffhanger["score"], clarity["score"], intensity["score"])
    confidence = _confidence(flags)
    review = confidence == "low" or clarity["score"] <= 3
    scored["scores"] = {
        "hook": hook["score"],
        "cliffhanger": cliffhanger["score"],
        "standalone_clarity": clarity["score"],
        "emotional_intensity": intensity["score"],
        "overall": overall,
    }
    scored["weights"] = {
        "hook": HOOK_WEIGHT,
        "cliffhanger": CLIFFHANGER_WEIGHT,
        "standalone_clarity": CLARITY_WEIGHT,
        "emotional_intensity": INTENSITY_WEIGHT,
    }
    scored["explanations"] = {
        "hook": hook["explanation"],
        "cliffhanger": cliffhanger["explanation"],
        "standalone_clarity": clarity["explanation"],
        "emotional_intensity": intensity["explanation"],
        "overall": (
            f"{overall} = {hook['score']}×3.5 + {cliffhanger['score']}×3.0 + "
            f"{clarity['score']}×2.0 + {intensity['score']}×1.5."
        ),
    }
    scored["evidence"] = {
        "hook": hook["evidence"],
        "cliffhanger": cliffhanger["evidence"],
        "standalone_clarity": clarity["evidence"],
        "emotional_intensity": intensity["evidence"],
    }
    scored["confidence"] = confidence
    scored["needs_human_review"] = review
    scored["review_flags"] = sorted(set(flags))
    return scored


def score_candidates(payload: dict | list | None) -> dict:
    """Score every candidate in a generator payload without removing any."""
    rows, error = _rows(payload)
    scored = [score_candidate(row) if isinstance(row, dict) else score_candidate(None) for row in rows]
    return {
        "candidates": scored,
        "summary": _summary(scored, error),
    }


def load_candidates(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Read promo candidates and write ``outputs/scored_candidates.json``."""
    root = repo_root()
    argument_parser = argparse.ArgumentParser(description="Score promo candidates.")
    argument_parser.add_argument(
        "--input",
        default=str(root / "outputs" / "promo_candidates.json"),
        help="Candidate JSON from promo generation.",
    )
    argument_parser.add_argument(
        "--output",
        default=str(root / "outputs" / "scored_candidates.json"),
        help="Path of the scored JSON file to write.",
    )
    args = argument_parser.parse_args(argv)
    source = Path(args.input)
    if not source.is_file():
        print(f"Candidate file not found: {source}")
        return 1
    result = score_candidates(load_candidates(source))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = result["summary"]
    overall = summary["overall"]
    print(
        f"Scored {summary['candidate_count']} candidates. "
        f"Overall min {overall['min']}, max {overall['max']}, mean {overall['mean']}."
    )
    print(f"Wrote {output}")
    return 0


def _score_hook(spine: list[dict], start_ms: int | None, end_ms: int | None, flags: list[str]) -> dict:
    if start_ms is None:
        return _dimension(0, [], "Hook not scored because the candidate has no usable timestamps.")
    window_end = start_ms + HOOK_WINDOW_MS
    opening = [cue for cue in spine if _overlaps(cue, start_ms, window_end)]
    if not opening:
        flags.append("opening_has_no_spine_cue")
        return _dimension(
            0,
            [],
            "No dialogue or action cue overlaps the first 3 seconds.",
        )
    quotes = [_quote(cue, "Cue overlaps the first 3 seconds.") for cue in opening]
    signals = _collect_signals(opening)
    points = 0
    rules = []
    if signals["threat"] or signals["reveal"] or signals["physical_action"]:
        points += 4
        rules.append("threat, reveal, or physical action in the first 3 seconds: +4")
    if signals["intense"]:
        points += 3
        rules.append("intense delivery note in the first 3 seconds: +3")
    if signals["question"]:
        points += 3
        rules.append("suspenseful question in the first 3 seconds: +3")
    if signals["interrupted"]:
        points += 2
        rules.append("interrupted dialogue in the first 3 seconds: +2")
    if signals["exclamation"]:
        points += 1
        rules.append("exclamation in the first 3 seconds: +1")
    if points == 0:
        points = 1
        rules.append("opening cue has no listed hook signal: +1")
    score = _clamp_dimension(points)
    evidence = {"quotes": quotes, "rules": rules, "signals": signals["labels"]}
    explanation = "First 3 seconds: " + "; ".join(rules) + f" Score {score}."
    return _dimension(score, evidence, explanation)


def _score_cliffhanger(spine: list[dict], start_ms: int | None, end_ms: int | None, flags: list[str]) -> dict:
    if end_ms is None:
        return _dimension(0, [], "Cliffhanger not scored because the candidate has no usable timestamps.")
    window_start = max(start_ms or 0, end_ms - ENDING_WINDOW_MS)
    ending = [cue for cue in spine if _overlaps(cue, window_start, end_ms)]
    if not ending:
        flags.append("ending_has_no_spine_cue")
        return _dimension(0, [], "No dialogue or action cue overlaps the final 3 seconds.")
    quotes = [_quote(cue, "Cue overlaps the final 3 seconds.") for cue in ending]
    signals = _collect_signals(ending)
    points = 0
    rules = []
    if _any_question_mark(ending):
        points += 4
        rules.append("ending line is an unresolved question: +4")
    if signals["interrupted"]:
        points += 4
        rules.append("ending dialogue is interrupted: +4")
    if signals["threat"] or signals["reveal"]:
        points += 3
        rules.append("ending cue states a threat or revelation: +3")
    if signals["intense"]:
        points += 2
        rules.append("ending cue has an intense delivery note: +2")
    if _action_reaches_cut(ending, end_ms):
        points += 2
        rules.append("an action is still underway at the cut: +2")
    if _ending_resolves(ending):
        points -= 3
        rules.append("ending line uses resolution wording: -3")
    score = _clamp_dimension(points)
    evidence = {"quotes": quotes, "rules": rules, "signals": signals["labels"]}
    explanation = "Final 3 seconds: " + "; ".join(rules) + f" Score {score}."
    return _dimension(score, evidence, explanation)


def _score_clarity(dialogue: list[dict], action: list[dict], camera: list[dict], flags: list[str]) -> dict:
    rules = [{"rule": "base", "points": 4, "detail": "Base clarity is 4 before visible or spoken evidence."}]
    points = 4
    quotes = []
    heard = [_spoken(cue.get("text")) for cue in dialogue]
    visible = [_spoken(cue.get("text")) for cue in action + camera]
    long_spoken = [line for line in heard if len(line.split()) >= 8]
    long_visible = [line for line in visible if len(line) >= 80]
    if long_visible:
        points += 2
        rules.append({"rule": "visible_situation", "points": 2, "detail": "An action or camera line describes a visible situation: +2"})
        quotes.append({"text": long_visible[0], "note": "Visible description of at least 80 characters."})
    if long_spoken:
        points += 2
        rules.append({"rule": "full_statement", "points": 2, "detail": "A spoken line contains at least eight words: +2"})
        quotes.append({"text": long_spoken[0], "note": "Spoken line of at least eight words."})
    if len([line for line in heard if line]) >= 2:
        points += 1
        rules.append({"rule": "exchange", "points": 1, "detail": "At least two spoken lines form an exchange: +1"})
    callbacks = _callback_hits(heard)
    if callbacks:
        penalty = min(3, len(callbacks))
        points -= penalty
        rules.append(
            {
                "rule": "unexplained_reference",
                "points": -penalty,
                "detail": f"Callback phrasing {callbacks[:3]} is not explained inside the clip: -{penalty}",
            }
        )
    short_lines = [line for line in heard if line and len(line.split()) <= 3]
    if heard and len(short_lines) == len([line for line in heard if line]):
        points -= 2
        rules.append({"rule": "fragments_only", "points": -2, "detail": "Every spoken line is three words or fewer: -2"})
    if not dialogue:
        points -= 2
        rules.append({"rule": "no_dialogue", "points": -2, "detail": "No spoken line gives the viewer a stated conflict: -2"})
    if not action and not camera:
        points -= 2
        rules.append({"rule": "no_visible_cue", "points": -2, "detail": "No action or camera text describes what is visible: -2"})
    score = _clamp_dimension(points)
    if score <= 3:
        flags.append("low_clarity")
    explanation = "Standalone clarity uses only text inside the clip. " + " ".join(rule["detail"] for rule in rules) + f" Score {score}."
    return _dimension(score, {"quotes": quotes, "rules": rules}, explanation)


def _score_intensity(spine: list[dict]) -> dict:
    if not spine:
        return _dimension(0, {"quotes": [], "rules": []}, "No dialogue or action cue to judge emotional intensity. Score 0.")
    best = 0
    intense = 0
    quotes = []
    for cue in spine:
        scored = score_cue(cue)
        cue_score = scored["score"]
        if cue_score <= 0:
            continue
        intense += 1 if cue_score >= 3 else 0
        best = max(best, min(8, cue_score * 2))
        quotes.append(
            {
                **_quote(cue, "Lexical intensity signals on this cue."),
                "signals": list(scored["signals"]),
                "cue_score": cue_score,
            }
        )
    rules = [f"strongest cue contributes {best}"]
    points = best
    if intense >= 2:
        points += 1
        rules.append("two or more cues score at least 3: +1")
    if intense >= 4:
        points += 1
        rules.append("four or more cues score at least 3: +1")
    if any("!" in _spoken(cue.get("text")) for cue in spine):
        points += 1
        rules.append("an exclamation appears in the clip: +1")
    score = _clamp_dimension(points)
    explanation = "Emotional intensity reads delivery notes, conflict words, and actions. " + "; ".join(rules) + f". Score {score}."
    return _dimension(score, {"quotes": quotes, "rules": rules}, explanation)


def _collect_signals(cues: list[dict]) -> dict:
    found = {
        "threat": False,
        "reveal": False,
        "physical_action": False,
        "intense": False,
        "question": False,
        "interrupted": False,
        "exclamation": False,
        "labels": [],
    }
    for cue in cues:
        scored = score_cue(cue)
        categories = set(scored["categories"])
        found["threat"] = found["threat"] or "threat" in categories
        found["reveal"] = found["reveal"] or "reveal" in categories
        found["physical_action"] = found["physical_action"] or "physical_action" in categories
        found["intense"] = found["intense"] or "intense_delivery" in categories
        found["question"] = found["question"] or "suspenseful_question" in categories
        found["interrupted"] = found["interrupted"] or "trailing_interruption" in scored["signals"]
        if "!" in _spoken(cue.get("text")):
            found["exclamation"] = True
        found["labels"].extend(scored["signals"])
    return found


def _any_question_mark(cues: list[dict]) -> bool:
    return any(_spoken(cue.get("text")).rstrip().endswith("?") for cue in cues if cue.get("type") == "DIALOGUE")


def _ending_resolves(cues: list[dict]) -> bool:
    dialogue = [cue for cue in cues if cue.get("type") == "DIALOGUE"]
    if not dialogue:
        return False
    last = max(dialogue, key=_cue_order)
    spoken = _spoken(last.get("text")).lower()
    return any(phrase in spoken for phrase in _RESOLUTIONS)


def _action_reaches_cut(cues: list[dict], end_ms: int) -> bool:
    for cue in cues:
        if cue.get("type") != "ACTION":
            continue
        cue_end = cue.get("end_ms")
        spoken = _spoken(cue.get("text")).lower()
        if isinstance(cue_end, int) and cue_end >= end_ms:
            return True
        if any(phrase in spoken for phrase in _INCOMPLETE_ACTION):
            return True
    return False


def _callback_hits(lines: list[str]) -> list[str]:
    hits = []
    blob = " ".join(line.lower() for line in lines)
    for phrase in _CALLBACKS:
        if re_contains(blob, phrase):
            hits.append(phrase)
    return hits


def re_contains(blob: str, phrase: str) -> bool:
    return re.search(r"\b" + re.escape(phrase) + r"\b", blob) is not None


def _dimension(score: int, evidence: dict | list, explanation: str) -> dict:
    return {"score": score, "evidence": evidence, "explanation": explanation}


def _empty_score(candidate_id: str | None, flags: list[str]) -> dict:
    zero = _dimension(0, {"quotes": [], "rules": []}, "No usable candidate content. Score 0.")
    return {
        "candidate_id": candidate_id,
        "scores": {
            "hook": 0,
            "cliffhanger": 0,
            "standalone_clarity": 0,
            "emotional_intensity": 0,
            "overall": 0.0,
        },
        "weights": {
            "hook": HOOK_WEIGHT,
            "cliffhanger": CLIFFHANGER_WEIGHT,
            "standalone_clarity": CLARITY_WEIGHT,
            "emotional_intensity": INTENSITY_WEIGHT,
        },
        "explanations": {
            "hook": zero["explanation"],
            "cliffhanger": zero["explanation"],
            "standalone_clarity": zero["explanation"],
            "emotional_intensity": zero["explanation"],
            "overall": "0.0 = 0×3.5 + 0×3.0 + 0×2.0 + 0×1.5.",
        },
        "evidence": {name: zero["evidence"] for name in ("hook", "cliffhanger", "standalone_clarity", "emotional_intensity")},
        "confidence": "low",
        "needs_human_review": True,
        "review_flags": sorted(set(flags)),
    }


def _confidence(flags: list[str]) -> str:
    low_flags = {"no_spine_cues", "opening_has_no_spine_cue", "ending_has_no_spine_cue", "incomplete_timestamps", "invalid_candidate"}
    if low_flags.intersection(flags):
        return "low"
    if "no_dialogue" in flags or "no_action" in flags or "low_clarity" in flags:
        return "medium"
    return "high"


def _rows(payload: dict | list | None) -> tuple[list, str | None]:
    if payload is None:
        return [], "Input is empty."
    if isinstance(payload, list):
        return payload, None
    if not isinstance(payload, dict):
        return [], "Input must be a dictionary or a list."
    rows = payload.get("candidates")
    if rows is None:
        return [], "Input has no candidates."
    if not isinstance(rows, list):
        return [], "Candidates must be a list."
    return rows, None


def _cues(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [cue for cue in value if isinstance(cue, dict)]


def _skipped_cue_count(candidate: dict) -> int:
    skipped = 0
    for key in ("dialogue", "action", "camera", "sound"):
        value = candidate.get(key)
        if not isinstance(value, list):
            continue
        for cue in value:
            if isinstance(cue, dict) and not _has_time(cue):
                skipped += 1
    return skipped


def _has_time(cue: dict) -> bool:
    start = cue.get("start_ms")
    end = cue.get("end_ms")
    return isinstance(start, int) and isinstance(end, int) and end >= start


def _overlaps(cue: dict, start_ms: int, end_ms: int) -> bool:
    if not _has_time(cue):
        return False
    return cue["start_ms"] < end_ms and cue["end_ms"] > start_ms


def _cue_order(cue: dict) -> tuple:
    return (cue.get("start_ms") if isinstance(cue.get("start_ms"), int) else 0, cue.get("line") or 0)


def _quote(cue: dict, note: str) -> dict:
    quote = {
        "text": cue.get("text"),
        "type": cue.get("type"),
        "start": cue.get("start"),
        "end": cue.get("end"),
        "start_ms": cue.get("start_ms"),
        "end_ms": cue.get("end_ms"),
        "note": note,
    }
    if cue.get("speaker"):
        quote["speaker"] = cue.get("speaker")
    return quote


def _spoken(text: object) -> str:
    if not isinstance(text, str):
        return ""
    spoken = text.strip()
    if len(spoken) >= 2 and spoken[0] == '"' and spoken[-1] == '"':
        return spoken[1:-1]
    return spoken


def _clamp_dimension(value: float) -> int:
    if not isinstance(value, (int, float)):
        return 0
    return max(0, min(10, int(round(value))))


def _summary(scored: list[dict], error: str | None) -> dict:
    overalls = [row["scores"]["overall"] for row in scored if isinstance(row.get("scores"), dict)]
    dimensions = {}
    for name in ("hook", "cliffhanger", "standalone_clarity", "emotional_intensity"):
        values = [row["scores"][name] for row in scored if isinstance(row.get("scores"), dict)]
        dimensions[name] = _stats(values)
    histogram = {"0-19": 0, "20-39": 0, "40-59": 0, "60-79": 0, "80-100": 0}
    for value in overalls:
        histogram[_bucket(value)] += 1
    return {
        "candidate_count": len(scored),
        "overall": _stats(overalls),
        "dimensions": dimensions,
        "overall_histogram": histogram,
        "needs_human_review": sum(1 for row in scored if row.get("needs_human_review")),
        "weights": {
            "hook": HOOK_WEIGHT,
            "cliffhanger": CLIFFHANGER_WEIGHT,
            "standalone_clarity": CLARITY_WEIGHT,
            "emotional_intensity": INTENSITY_WEIGHT,
        },
        "errors": [error] if error else [],
    }


def _stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
    }


def _bucket(value: float) -> str:
    if value < 20:
        return "0-19"
    if value < 40:
        return "20-39"
    if value < 60:
        return "40-59"
    if value < 80:
        return "60-79"
    return "80-100"


if __name__ == "__main__":
    raise SystemExit(main())
