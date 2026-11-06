"""Build 20–60 second promo candidates from parsed screenplay cues.

Dialogue and action cues set each window. Camera and sound cues are attached
only when their timestamps overlap that window. Times come from the parser;
nothing is estimated from word counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.event_detector import detect_moments


MIN_DURATION_MS = 20_000
MAX_DURATION_MS = 60_000
TARGET_BALANCED_MS = 35_000
TARGET_LONG_MS = 50_000
NEAR_DUPLICATE_IOU = 0.85


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def generate_candidates(parsed: dict | None) -> dict:
    """Return promo candidates and a generation summary for parsed episodes."""
    episodes, error = _episodes(parsed)
    raw: list[dict] = []
    moment_count = 0
    episode_numbers: list[int] = []
    if error is None:
        for episode in episodes:
            number = episode.get("episode_number") if isinstance(episode, dict) else None
            if isinstance(number, int):
                episode_numbers.append(number)
            spine, coverage, scenes = _tracks(episode)
            moments = detect_moments(episode) if isinstance(episode, dict) else []
            moment_count += len(moments)
            for moment in moments:
                raw.extend(_candidates_for_moment(episode, spine, coverage, scenes, moment))
    kept = dedupe_candidates(raw)
    kept.sort(key=lambda candidate: (candidate["episode_number"] or 0, candidate["start_ms"], candidate["end_ms"]))
    per_episode_index: dict[int, int] = {}
    for candidate in kept:
        episode_number = candidate["episode_number"] or 0
        per_episode_index[episode_number] = per_episode_index.get(episode_number, 0) + 1
        candidate["candidate_id"] = f"ep{episode_number:02d}-c{per_episode_index[episode_number]:04d}"
    return {
        "candidates": kept,
        "summary": _summary(kept, raw, moment_count, episode_numbers, error),
    }


def dedupe_candidates(candidates: list[dict], iou_threshold: float = NEAR_DUPLICATE_IOU) -> list[dict]:
    """Drop exact and heavily overlapping windows, keeping the stronger trigger."""
    grouped: dict[int, list[dict]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.get("episode_number"), []).append(candidate)
    kept: list[dict] = []
    for episode_candidates in grouped.values():
        ordered = sorted(episode_candidates, key=_dedupe_sort_key)
        chosen: list[dict] = []
        for candidate in ordered:
            if any(_iou(candidate, other) >= iou_threshold for other in chosen):
                continue
            chosen.append(candidate)
        kept.extend(chosen)
    return kept


def load_parsed(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """Read Milestone 1 JSON and write ``outputs/promo_candidates.json``."""
    root = repo_root()
    argument_parser = argparse.ArgumentParser(description="Generate promo candidates from parsed episodes.")
    argument_parser.add_argument(
        "--input",
        default=str(root / "outputs" / "parsed_episodes.json"),
        help="Parsed episode JSON from the screenplay parser.",
    )
    argument_parser.add_argument(
        "--output",
        default=str(root / "outputs" / "promo_candidates.json"),
        help="Path of the candidate JSON file to write.",
    )
    args = argument_parser.parse_args(argv)
    source = Path(args.input)
    if not source.is_file():
        print(f"Parsed episode file not found: {source}")
        return 1
    result = generate_candidates(load_parsed(source))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = result["summary"]
    print(
        f"Generated {summary['candidate_count']} candidates "
        f"from {summary['moment_count']} dramatic moments "
        f"({summary['candidates_before_dedupe']} before duplicate removal)."
    )
    print(f"Duration seconds: min {summary['duration_seconds']['min']}, max {summary['duration_seconds']['max']}")
    print(f"Wrote {output}")
    return 0


def _episodes(parsed: dict | None) -> tuple[list, str | None]:
    if parsed is None:
        return [], "Input is empty."
    if not isinstance(parsed, dict):
        return [], "Input must be a dictionary."
    episodes = parsed.get("episodes")
    if episodes is None:
        return [], "Input has no episodes."
    if not isinstance(episodes, list):
        return [], "Episodes must be a list."
    return [episode for episode in episodes if isinstance(episode, dict)], None


def _tracks(episode: dict) -> tuple[list[dict], list[dict], list[dict]]:
    scenes = episode.get("scenes")
    if not isinstance(scenes, list):
        return [], [], []
    spine: list[dict] = []
    coverage: list[dict] = []
    scene_rows: list[dict] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_rows.append(scene)
        cues = scene.get("cues")
        if not isinstance(cues, list):
            continue
        for cue in cues:
            if not isinstance(cue, dict) or not _timed(cue):
                continue
            annotated = dict(cue)
            annotated["scene_number"] = scene.get("scene_number")
            annotated["scene_heading"] = scene.get("scene_heading")
            if cue.get("type") in {"DIALOGUE", "ACTION"}:
                spine.append(annotated)
            elif cue.get("type") in {"CAMERA", "SOUND FX"}:
                coverage.append(annotated)
    spine.sort(key=lambda cue: (cue["start_ms"], cue["end_ms"], cue.get("line") or 0))
    return spine, coverage, scene_rows


def _timed(cue: dict) -> bool:
    start = cue.get("start_ms")
    end = cue.get("end_ms")
    return isinstance(start, int) and isinstance(end, int) and end >= start


def _candidates_for_moment(
    episode: dict,
    spine: list[dict],
    coverage: list[dict],
    scenes: list[dict],
    moment: dict,
) -> list[dict]:
    index = moment.get("spine_index")
    if not isinstance(index, int) or index < 0 or index >= len(spine):
        return []
    plans = (
        ("forward_from_trigger", ("forward", "backward"), None),
        ("backward_to_trigger", ("backward", "forward"), None),
        ("balanced_35s", None, TARGET_BALANCED_MS),
        ("balanced_50s", None, TARGET_LONG_MS),
    )
    seen: set[tuple[int, int]] = set()
    built = []
    for strategy, order, target in plans:
        bounds = _grow(spine, index, strategy, order, target)
        if bounds is None or bounds in seen:
            continue
        seen.add(bounds)
        built.append(_build_candidate(episode, spine, coverage, scenes, moment, bounds, strategy))
    return built


def _grow(
    spine: list[dict],
    trigger: int,
    strategy: str,
    order: tuple[str, str] | None,
    target: int | None,
) -> tuple[int, int] | None:
    left = right = trigger
    if _duration(spine, left, right) > MAX_DURATION_MS:
        return None
    while _duration(spine, left, right) < MIN_DURATION_MS:
        nxt = _next_bounds(spine, left, right, trigger, strategy, order)
        if nxt is None:
            return None
        left, right = nxt
    if target is not None:
        while _duration(spine, left, right) < target:
            nxt = _next_bounds(spine, left, right, trigger, strategy, order)
            if nxt is None or _duration(spine, *nxt) > MAX_DURATION_MS:
                break
            left, right = nxt
    if not MIN_DURATION_MS <= _duration(spine, left, right) <= MAX_DURATION_MS:
        return None
    return left, right


def _next_bounds(
    spine: list[dict],
    left: int,
    right: int,
    trigger: int,
    strategy: str,
    order: tuple[str, str] | None,
) -> tuple[int, int] | None:
    directions = order if order is not None else _balanced_directions(spine, left, right, trigger)
    for direction in directions:
        if direction == "forward" and right + 1 < len(spine):
            candidate = (left, right + 1)
        elif direction == "backward" and left - 1 >= 0:
            candidate = (left - 1, right)
        else:
            continue
        if _duration(spine, *candidate) <= MAX_DURATION_MS or _duration(spine, left, right) < MIN_DURATION_MS:
            if _duration(spine, *candidate) <= MAX_DURATION_MS:
                return candidate
    return None


def _balanced_directions(spine: list[dict], left: int, right: int, trigger: int) -> tuple[str, ...]:
    trigger_mid = (spine[trigger]["start_ms"] + spine[trigger]["end_ms"]) / 2

    def distance(bounds: tuple[int, int]) -> float:
        start = spine[bounds[0]]["start_ms"]
        end = spine[bounds[1]]["end_ms"]
        return abs(((start + end) / 2) - trigger_mid)

    ranked = []
    if right + 1 < len(spine):
        ranked.append((distance((left, right + 1)), 0, "forward"))
    if left - 1 >= 0:
        ranked.append((distance((left - 1, right)), 1, "backward"))
    ranked.sort()
    return tuple(direction for _, _, direction in ranked)


def _duration(spine: list[dict], left: int, right: int) -> int:
    return spine[right]["end_ms"] - spine[left]["start_ms"]


def _build_candidate(
    episode: dict,
    spine: list[dict],
    coverage: list[dict],
    scenes: list[dict],
    moment: dict,
    bounds: tuple[int, int],
    strategy: str,
) -> dict:
    left, right = bounds
    selected = spine[left : right + 1]
    start_ms = selected[0]["start_ms"]
    end_ms = selected[-1]["end_ms"]
    dialogue = [_public_cue(cue) for cue in selected if cue["type"] == "DIALOGUE"]
    action = [_public_cue(cue) for cue in selected if cue["type"] == "ACTION"]
    camera = [_public_cue(cue) for cue in coverage if cue["type"] == "CAMERA" and _overlaps(cue, start_ms, end_ms)]
    sound = [_public_cue(cue) for cue in coverage if cue["type"] == "SOUND FX" and _overlaps(cue, start_ms, end_ms)]
    trigger_cue = moment["cue"]
    categories = ", ".join(moment["categories"])
    signals = ", ".join(moment["signals"])
    reason = (
        f"{strategy.replace('_', ' ')}: consecutive dialogue and action "
        f"from {selected[0]['start']} to {selected[-1]['end']} "
        f"include a {categories} cue. Signals: {signals}."
    )
    return {
        "candidate_id": None,
        "episode_number": episode.get("episode_number"),
        "catalog_id": episode.get("catalog_id"),
        "start": selected[0]["start"],
        "end": selected[-1]["end"],
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "duration_seconds": (end_ms - start_ms) / 1000,
        "scenes": _overlapping_scenes(scenes, start_ms, end_ms),
        "dialogue": dialogue,
        "action": action,
        "camera": camera,
        "sound": sound,
        "trigger": {
            "categories": list(moment["categories"]),
            "signals": list(moment["signals"]),
            "score": moment["score"],
            "scene_number": trigger_cue.get("scene_number"),
            "scene_heading": trigger_cue.get("scene_heading"),
            "cue_type": trigger_cue.get("type"),
            "speaker": trigger_cue.get("speaker"),
            "raw_label": trigger_cue.get("raw_label"),
            "text": trigger_cue.get("text"),
            "start": trigger_cue.get("start"),
            "end": trigger_cue.get("end"),
            "start_ms": trigger_cue.get("start_ms"),
            "end_ms": trigger_cue.get("end_ms"),
        },
        "reason": reason,
        "generation_strategy": strategy,
    }


def _public_cue(cue: dict) -> dict:
    kept = {
        "type": cue.get("type"),
        "raw_label": cue.get("raw_label"),
        "text": cue.get("text"),
        "start": cue.get("start"),
        "end": cue.get("end"),
        "start_ms": cue.get("start_ms"),
        "end_ms": cue.get("end_ms"),
        "line": cue.get("line"),
        "scene_number": cue.get("scene_number"),
        "scene_heading": cue.get("scene_heading"),
    }
    if cue.get("type") == "DIALOGUE":
        kept["speaker"] = cue.get("speaker")
        kept["parentheticals"] = list(cue.get("parentheticals") or [])
        kept["markers"] = list(cue.get("markers") or [])
    return kept


def _overlaps(cue: dict, start_ms: int, end_ms: int) -> bool:
    return cue["start_ms"] < end_ms and cue["end_ms"] > start_ms


def _overlapping_scenes(scenes: list[dict], start_ms: int, end_ms: int) -> list[dict]:
    rows = []
    for scene in scenes:
        scene_start = scene.get("scene_start_ms")
        scene_end = scene.get("scene_end_ms")
        if not isinstance(scene_start, int) or not isinstance(scene_end, int):
            continue
        if scene_end <= start_ms or scene_start >= end_ms:
            continue
        rows.append(
            {
                "scene_number": scene.get("scene_number"),
                "scene_heading": scene.get("scene_heading"),
                "scene_start": scene.get("scene_start"),
                "scene_end": scene.get("scene_end"),
                "scene_start_ms": scene_start,
                "scene_end_ms": scene_end,
            }
        )
    return rows


def _dedupe_sort_key(candidate: dict) -> tuple:
    trigger = candidate.get("trigger") or {}
    start = candidate["start_ms"]
    end = candidate["end_ms"]
    trigger_start = trigger.get("start_ms") if isinstance(trigger.get("start_ms"), int) else start
    trigger_end = trigger.get("end_ms") if isinstance(trigger.get("end_ms"), int) else end
    center_distance = abs(((start + end) / 2) - ((trigger_start + trigger_end) / 2))
    return (
        -(trigger.get("score") or 0),
        center_distance,
        abs(candidate["duration_ms"] - TARGET_BALANCED_MS),
        start,
        end,
    )


def _iou(left: dict, right: dict) -> float:
    intersection = min(left["end_ms"], right["end_ms"]) - max(left["start_ms"], right["start_ms"])
    if intersection <= 0:
        return 0.0
    union = (left["end_ms"] - left["start_ms"]) + (right["end_ms"] - right["start_ms"]) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _summary(
    kept: list[dict],
    raw: list[dict],
    moment_count: int,
    episode_numbers: list[int],
    error: str | None,
) -> dict:
    per_episode = {str(number): 0 for number in sorted(set(episode_numbers))}
    histogram = {"20-29": 0, "30-39": 0, "40-49": 0, "50-60": 0}
    durations = [candidate["duration_seconds"] for candidate in kept]
    for candidate in kept:
        per_episode[str(candidate["episode_number"])] = per_episode.get(str(candidate["episode_number"]), 0) + 1
        histogram[_bucket(candidate["duration_seconds"])] += 1
    if durations:
        duration_stats = {
            "min": min(durations),
            "max": max(durations),
            "mean": round(sum(durations) / len(durations), 2),
        }
    else:
        duration_stats = {"min": None, "max": None, "mean": None}
    summary = {
        "candidate_count": len(kept),
        "candidates_before_dedupe": len(raw),
        "duplicates_removed": len(raw) - len(kept),
        "moment_count": moment_count,
        "episodes_with_candidates": sum(1 for count in per_episode.values() if count),
        "candidates_per_episode": per_episode,
        "duration_seconds": duration_stats,
        "duration_histogram": histogram,
        "min_duration_ms": MIN_DURATION_MS,
        "max_duration_ms": MAX_DURATION_MS,
        "near_duplicate_iou": NEAR_DUPLICATE_IOU,
        "errors": [error] if error else [],
    }
    return summary


def _bucket(seconds: float) -> str:
    if seconds < 30:
        return "20-29"
    if seconds < 40:
        return "30-39"
    if seconds < 50:
        return "40-49"
    return "50-60"


if __name__ == "__main__":
    raise SystemExit(main())
