"""Rank scored promo candidates into a deterministic shortlist.

Sorting uses the overall score already stored by the scorer. The shortlist
then skips near-duplicates and extra clips from the same episode. Every input
candidate stays in the ranked list; exclusions are recorded with a reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_TOP_N = 10
DEFAULT_MAX_PER_EPISODE = 2
DEFAULT_TIME_IOU = 0.50
DEFAULT_TEXT_SIMILARITY = 0.60

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_CSV_FIELDS = (
    "score_rank",
    "shortlist_rank",
    "shortlist_status",
    "exclusion_reason",
    "similar_to",
    "candidate_id",
    "episode_number",
    "start",
    "end",
    "start_ms",
    "end_ms",
    "duration_seconds",
    "overall",
    "hook",
    "cliffhanger",
    "standalone_clarity",
    "emotional_intensity",
    "time_overlap",
    "text_similarity",
    "ranking_explanation",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rank_candidates(
    payload: dict | list | None,
    top_n: int = DEFAULT_TOP_N,
    max_per_episode: int | None = DEFAULT_MAX_PER_EPISODE,
    time_iou: float = DEFAULT_TIME_IOU,
    text_similarity: float = DEFAULT_TEXT_SIMILARITY,
) -> dict:
    """Rank candidates and choose a diverse top-N shortlist.

    Input dictionaries are not modified. ``top_n`` limits the shortlist.
    ``max_per_episode`` limits how many shortlist rows may come from one
    episode. Pass ``None`` to disable that cap.
    """
    rows, error = _rows(payload)
    limit = max(0, int(top_n))
    ordered = sorted((_prepare(row) for row in rows), key=_sort_key)
    selected: list[dict] = []
    episode_counts: dict[object, int] = {}

    for index, candidate in enumerate(ordered, start=1):
        candidate["score_rank"] = index
        if len(selected) >= limit:
            _mark_below_shortlist(candidate, limit)
            continue
        duplicate = _matching_selection(candidate, selected, time_iou, text_similarity)
        if duplicate is not None:
            _mark_excluded(candidate, duplicate)
            continue
        episode = candidate.get("episode_number")
        if max_per_episode is not None and episode_counts.get(episode, 0) >= max_per_episode:
            _mark_episode_cap(candidate, selected, episode, max_per_episode)
            continue
        episode_counts[episode] = episode_counts.get(episode, 0) + 1
        selected.append(candidate)

    for shortlist_rank, candidate in enumerate(selected, start=1):
        candidate["shortlist_rank"] = shortlist_rank
        candidate["shortlist_status"] = "recommended"
        candidate["ranking_explanation"] = _selection_explanation(candidate, len(selected))
    for candidate in ordered:
        candidate.pop("_tokens", None)
        candidate.pop("_exclusion_code", None)

    return {
        "recommendations": selected,
        "excluded_from_shortlist": [row for row in ordered if row["shortlist_status"] == "excluded"],
        "ranked_candidates": ordered,
        "summary": _summary(ordered, selected, limit, max_per_episode, time_iou, text_similarity, error),
    }


def load_scored(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_outputs(result: dict, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for candidate in result["ranked_candidates"]:
            writer.writerow(_csv_row(candidate))


def main(argv: list[str] | None = None) -> int:
    """Read scored candidates and write the ranked JSON and CSV files."""
    root = repo_root()
    argument_parser = argparse.ArgumentParser(description="Rank scored promo candidates.")
    argument_parser.add_argument("--input", default=str(root / "outputs" / "scored_candidates.json"))
    argument_parser.add_argument("--json", dest="json_path", default=str(root / "outputs" / "ranked_promos.json"))
    argument_parser.add_argument("--csv", dest="csv_path", default=str(root / "outputs" / "ranked_promos.csv"))
    argument_parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    argument_parser.add_argument("--max-per-episode", type=int, default=DEFAULT_MAX_PER_EPISODE)
    argument_parser.add_argument("--time-iou", type=float, default=DEFAULT_TIME_IOU)
    argument_parser.add_argument("--text-similarity", type=float, default=DEFAULT_TEXT_SIMILARITY)
    args = argument_parser.parse_args(argv)
    source = Path(args.input)
    if not source.is_file():
        print(f"Scored candidate file not found: {source}")
        return 1
    result = rank_candidates(
        load_scored(source),
        top_n=args.top_n,
        max_per_episode=args.max_per_episode,
        time_iou=args.time_iou,
        text_similarity=args.text_similarity,
    )
    write_outputs(result, Path(args.json_path), Path(args.csv_path))
    summary = result["summary"]
    print(
        f"Ranked {summary['candidate_count']} candidates. "
        f"Shortlist {summary['recommended_count']}. "
        f"Excluded {summary['excluded_count']}."
    )
    print(f"Wrote {args.json_path}")
    print(f"Wrote {args.csv_path}")
    return 0


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


def _prepare(row: object) -> dict:
    if not isinstance(row, dict):
        candidate = {
            "candidate_id": None,
            "episode_number": None,
            "scores": {},
            "review_flags": ["invalid_candidate"],
        }
    else:
        candidate = dict(row)
    scores = candidate.get("scores") if isinstance(candidate.get("scores"), dict) else {}
    candidate["scores"] = {
        "hook": scores.get("hook"),
        "cliffhanger": scores.get("cliffhanger"),
        "standalone_clarity": scores.get("standalone_clarity"),
        "emotional_intensity": scores.get("emotional_intensity"),
        "overall": scores.get("overall"),
    }
    candidate["shortlist_rank"] = None
    candidate["shortlist_status"] = "unranked"
    candidate["exclusion_reason"] = None
    candidate["similar_to"] = None
    candidate["time_overlap"] = None
    candidate["text_similarity"] = None
    candidate["ranking_explanation"] = None
    candidate["_tokens"] = _tokens(candidate)
    return candidate


def _sort_key(candidate: dict) -> tuple:
    scores = candidate["scores"]
    overall = scores.get("overall")
    return (
        overall is None,
        -(overall if isinstance(overall, (int, float)) else 0),
        -_number(scores.get("hook")),
        -_number(scores.get("cliffhanger")),
        -_number(scores.get("standalone_clarity")),
        -_number(scores.get("emotional_intensity")),
        candidate.get("episode_number") if isinstance(candidate.get("episode_number"), int) else 10**9,
        candidate.get("start_ms") if isinstance(candidate.get("start_ms"), int) else 10**12,
        candidate.get("candidate_id") or "",
    )


def _number(value: object) -> float:
    return value if isinstance(value, (int, float)) else 0


def _matching_selection(
    candidate: dict,
    selected: list[dict],
    time_iou: float,
    text_similarity: float,
) -> dict | None:
    for chosen in selected:
        if candidate.get("episode_number") != chosen.get("episode_number"):
            continue
        overlap = _time_iou(candidate, chosen)
        similarity = _jaccard(candidate.get("_tokens") or set(), chosen.get("_tokens") or set())
        if overlap >= time_iou:
            candidate["time_overlap"] = round(overlap, 4)
            candidate["text_similarity"] = round(similarity, 4)
            candidate["_exclusion_code"] = "near_duplicate_time"
            return chosen
        if similarity >= text_similarity and candidate.get("_tokens") and chosen.get("_tokens"):
            candidate["time_overlap"] = round(overlap, 4)
            candidate["text_similarity"] = round(similarity, 4)
            candidate["_exclusion_code"] = "near_duplicate_text"
            return chosen
    return None


def _mark_excluded(candidate: dict, chosen: dict) -> None:
    code = candidate.get("_exclusion_code") or "near_duplicate"
    overlap = candidate.get("time_overlap")
    similarity = candidate.get("text_similarity")
    if code == "near_duplicate_time":
        detail = (
            f"Excluded from the shortlist because it overlaps {chosen.get('candidate_id')} "
            f"in episode {candidate.get('episode_number')} "
            f"(timestamp overlap {overlap}, dialogue similarity {similarity})."
        )
    else:
        detail = (
            f"Excluded from the shortlist because its dialogue is too similar to "
            f"{chosen.get('candidate_id')} in episode {candidate.get('episode_number')} "
            f"(dialogue similarity {similarity}, timestamp overlap {overlap})."
        )
    candidate["shortlist_status"] = "excluded"
    candidate["exclusion_reason"] = code
    candidate["similar_to"] = chosen.get("candidate_id")
    candidate["ranking_explanation"] = (
        f"Score rank {candidate['score_rank']}. Overall {_overall_text(candidate)}. {detail} "
        "The original candidate is still listed in ranked_candidates."
    )


def _mark_episode_cap(candidate: dict, selected: list[dict], episode: object, limit: int) -> None:
    kept = [row.get("candidate_id") for row in selected if row.get("episode_number") == episode]
    candidate["shortlist_status"] = "excluded"
    candidate["exclusion_reason"] = "episode_cap"
    candidate["similar_to"] = kept[0] if kept else None
    candidate["ranking_explanation"] = (
        f"Score rank {candidate['score_rank']}. Overall {_overall_text(candidate)}. "
        f"Excluded from the shortlist because episode {episode} already has {limit} "
        f"selected promo(s): {', '.join(str(item) for item in kept)}. "
        "The original candidate is still listed in ranked_candidates."
    )


def _mark_below_shortlist(candidate: dict, limit: int) -> None:
    candidate["shortlist_status"] = "below_shortlist"
    candidate["exclusion_reason"] = None
    candidate["ranking_explanation"] = (
        f"Score rank {candidate['score_rank']}. Overall {_overall_text(candidate)}. "
        f"Not placed in the top {limit} after higher-scoring promos filled the shortlist."
    )


def _selection_explanation(candidate: dict, shortlist_size: int) -> str:
    scores = candidate["scores"]
    return (
        f"Shortlist rank {candidate['shortlist_rank']} of {shortlist_size}. "
        f"Score rank {candidate['score_rank']}. "
        f"Overall {scores.get('overall')} from hook {scores.get('hook')}, "
        f"cliffhanger {scores.get('cliffhanger')}, clarity {scores.get('standalone_clarity')}, "
        f"and intensity {scores.get('emotional_intensity')}. "
        f"Episode {candidate.get('episode_number')}, {candidate.get('start')}–{candidate.get('end')}. "
        "Selected as the highest remaining overall score that does not repeat a selected "
        "segment or exceed the per-episode cap."
    )


def _overall_text(candidate: dict) -> object:
    return candidate.get("scores", {}).get("overall")


def _tokens(candidate: dict) -> set[str]:
    parts: list[str] = []
    for key in ("dialogue", "action"):
        cues = candidate.get(key)
        if not isinstance(cues, list):
            continue
        for cue in cues:
            if isinstance(cue, dict) and isinstance(cue.get("text"), str):
                parts.append(cue["text"])
    spoken = " ".join(parts).lower().replace("\u2014", " ").replace("—", " ")
    return set(_TOKEN_RE.findall(spoken))


def _time_iou(left: dict, right: dict) -> float:
    start_left = left.get("start_ms")
    end_left = left.get("end_ms")
    start_right = right.get("start_ms")
    end_right = right.get("end_ms")
    if not all(isinstance(value, int) for value in (start_left, end_left, start_right, end_right)):
        return 0.0
    intersection = min(end_left, end_right) - max(start_left, start_right)
    if intersection <= 0:
        return 0.0
    union = (end_left - start_left) + (end_right - start_right) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _summary(
    ordered: list[dict],
    selected: list[dict],
    top_n: int,
    max_per_episode: int | None,
    time_iou: float,
    text_similarity: float,
    error: str | None,
) -> dict:
    excluded = [row for row in ordered if row["shortlist_status"] == "excluded"]
    reasons: dict[str, int] = {}
    for row in excluded:
        reason = row.get("exclusion_reason") or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    episodes = []
    for row in selected:
        episode = row.get("episode_number")
        if episode not in episodes:
            episodes.append(episode)
    return {
        "candidate_count": len(ordered),
        "recommended_count": len(selected),
        "excluded_count": len(excluded),
        "below_shortlist_count": sum(1 for row in ordered if row["shortlist_status"] == "below_shortlist"),
        "top_n": top_n,
        "max_per_episode": max_per_episode,
        "time_iou_threshold": time_iou,
        "text_similarity_threshold": text_similarity,
        "exclusion_reasons": reasons,
        "episodes_in_shortlist": episodes,
        "errors": [error] if error else [],
    }


def _csv_row(candidate: dict) -> dict:
    scores = candidate.get("scores") or {}
    return {
        "score_rank": candidate.get("score_rank"),
        "shortlist_rank": candidate.get("shortlist_rank") or "",
        "shortlist_status": candidate.get("shortlist_status"),
        "exclusion_reason": candidate.get("exclusion_reason") or "",
        "similar_to": candidate.get("similar_to") or "",
        "candidate_id": candidate.get("candidate_id") or "",
        "episode_number": candidate.get("episode_number") if candidate.get("episode_number") is not None else "",
        "start": candidate.get("start") or "",
        "end": candidate.get("end") or "",
        "start_ms": candidate.get("start_ms") if isinstance(candidate.get("start_ms"), int) else "",
        "end_ms": candidate.get("end_ms") if isinstance(candidate.get("end_ms"), int) else "",
        "duration_seconds": candidate.get("duration_seconds") if candidate.get("duration_seconds") is not None else "",
        "overall": scores.get("overall") if scores.get("overall") is not None else "",
        "hook": scores.get("hook") if scores.get("hook") is not None else "",
        "cliffhanger": scores.get("cliffhanger") if scores.get("cliffhanger") is not None else "",
        "standalone_clarity": scores.get("standalone_clarity") if scores.get("standalone_clarity") is not None else "",
        "emotional_intensity": scores.get("emotional_intensity") if scores.get("emotional_intensity") is not None else "",
        "time_overlap": candidate.get("time_overlap") if candidate.get("time_overlap") is not None else "",
        "text_similarity": candidate.get("text_similarity") if candidate.get("text_similarity") is not None else "",
        "ranking_explanation": candidate.get("ranking_explanation") or "",
    }


if __name__ == "__main__":
    raise SystemExit(main())
