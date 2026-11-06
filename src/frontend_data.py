"""Load promo JSON and store reviewer decisions.

These helpers do not score, rank, or validate candidates. Reviewer decisions are
written only to the decisions file passed by the caller.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DECISION_CHOICES = ("APPROVED", "REJECTED", "HOLD")
DECISION_PLACEHOLDER = "Select a decision"


def read_json(path: Path | str) -> tuple[object | None, str | None]:
    """Return parsed JSON, or an error message when the file cannot be used."""
    file_path = Path(path)
    if not file_path.is_file():
        return None, f"File not found: {file_path.name}"
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"Could not read {file_path.name}: {exc.strerror or exc}"
    if not text.strip():
        return None, f"{file_path.name} is empty."
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"{file_path.name} is not valid JSON ({exc.msg})."


def candidate_rows(payload: object) -> tuple[list[dict] | None, str | None]:
    """Accept a candidate list or the ranked and validated file shapes."""
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("candidates")
        if not isinstance(rows, list):
            rows = payload.get("ranked_candidates")
        if not isinstance(rows, list):
            return None, "JSON has no candidates list."
    else:
        return None, "JSON must be an object or a list."
    if any(not isinstance(row, dict) for row in rows):
        return None, "A candidate record was not an object."
    return rows, None


def load_candidates(path: Path | str) -> tuple[list[dict], str | None]:
    payload, error = read_json(path)
    if error:
        return [], error
    rows, shape_error = candidate_rows(payload)
    if shape_error:
        return [], shape_error
    return rows or [], None


def load_object(path: Path | str) -> tuple[dict, str | None]:
    payload, error = read_json(path)
    if error:
        return {}, error
    if not isinstance(payload, dict):
        return {}, f"{Path(path).name} must be a JSON object."
    return payload, None


def validation_of(candidate: dict) -> dict:
    value = candidate.get("validation")
    return value if isinstance(value, dict) else {}


def family_id(candidate: dict) -> str | None:
    family = validation_of(candidate).get("family")
    if not isinstance(family, dict):
        return None
    value = family.get("family_id")
    return value if isinstance(value, str) and value else None


def priority_of(candidate: dict) -> str:
    value = validation_of(candidate).get("priority")
    return value if isinstance(value, str) and value else "none"


def overall_score(candidate: dict) -> float | None:
    scores = candidate.get("scores")
    if not isinstance(scores, dict):
        return None
    value = scores.get("overall")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def dashboard_counts(candidates: list[dict], queue: dict | None = None) -> dict:
    """Count what is in the loaded records. Nothing here is a fixed dataset size."""
    priorities = Counter(priority_of(candidate) for candidate in candidates)
    families = {family_id(candidate) for candidate in candidates}
    families.discard(None)
    items = queue.get("items") if isinstance(queue, dict) else None
    if isinstance(items, list):
        queue_size = len(items)
    else:
        queue_size = sum(
            1
            for candidate in candidates
            if (validation_of(candidate).get("inspection") or {}).get("in_queue")
        )
    return {
        "total": len(candidates),
        "high": priorities.get("high", 0),
        "medium": priorities.get("medium", 0),
        "low": priorities.get("low", 0),
        "families": len(families),
        "inspection_queue": queue_size,
    }


def score_distribution(candidates: list[dict]) -> list[dict]:
    """Ten-point bands for the existing overall score."""
    counts = [0] * 10
    for candidate in candidates:
        score = overall_score(candidate)
        if score is None:
            continue
        band = min(9, max(0, int(score // 10)))
        counts[band] += 1
    rows = []
    for index, count in enumerate(counts):
        start = index * 10
        label = "90–100" if index == 9 else f"{start}–{start + 9}"
        rows.append({"band": label, "candidates": count})
    return rows


def episode_distribution(candidates: list[dict]) -> list[dict]:
    counts: dict[object, int] = {}
    for candidate in candidates:
        episode = candidate.get("episode_number")
        counts[episode] = counts.get(episode, 0) + 1

    def sort_key(item: tuple[object, int]) -> tuple:
        episode = item[0]
        return (0, episode) if isinstance(episode, int) else (1, str(episode))

    return [
        {"episode": episode, "candidates": count}
        for episode, count in sorted(counts.items(), key=sort_key)
    ]


def recommended_promos(candidates: list[dict]) -> list[dict]:
    picked = [row for row in candidates if row.get("shortlist_status") == "recommended"]
    return sorted(
        picked,
        key=lambda row: (
            row.get("shortlist_rank") is None,
            row.get("shortlist_rank") if isinstance(row.get("shortlist_rank"), int) else 10**9,
            row.get("candidate_id") or "",
        ),
    )


def filter_candidates(
    candidates: list[dict],
    *,
    text: str = "",
    episode: int | None = None,
    priority: str | None = None,
    min_score: float | None = None,
    max_score: float | None = None,
    family: str | None = None,
    ranking_status: str | None = None,
) -> list[dict]:
    needle = text.strip().lower()
    selected = []
    for candidate in candidates:
        if needle and needle not in str(candidate.get("candidate_id") or "").lower():
            continue
        if episode is not None and candidate.get("episode_number") != episode:
            continue
        if priority and priority_of(candidate) != priority:
            continue
        if min_score is not None or max_score is not None:
            score = overall_score(candidate)
            if score is None:
                continue
            if min_score is not None and score < min_score:
                continue
            if max_score is not None and score > max_score:
                continue
        if family and family_id(candidate) != family:
            continue
        if ranking_status and candidate.get("shortlist_status") != ranking_status:
            continue
        selected.append(candidate)
    return selected


def group_review_queue(queue: dict | None) -> list[dict]:
    """Group queue items by family. Representatives stay ahead of alternatives."""
    items = queue.get("items") if isinstance(queue, dict) else None
    if not isinstance(items, list):
        return []
    grouped: dict[str, list[dict]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        family = item.get("family_id") or "unassigned"
        grouped.setdefault(str(family), []).append(item)
    families = []
    for family, members in grouped.items():
        ordered = sorted(
            members,
            key=lambda item: (
                0 if item.get("role") == "representative" else 1,
                item.get("queue_position") if isinstance(item.get("queue_position"), int) else 10**9,
                item.get("candidate_id") or "",
            ),
        )
        families.append(
            {
                "family_id": family,
                "seed_candidate_id": ordered[0].get("family_seed_id"),
                "items": ordered,
            }
        )
    return sorted(families, key=lambda family: family["family_id"])


def cue_lines(cues: object) -> list[dict]:
    if not isinstance(cues, list):
        return []
    lines = []
    for cue in cues:
        if not isinstance(cue, dict):
            continue
        lines.append(
            {
                "start": cue.get("start"),
                "end": cue.get("end"),
                "speaker": cue.get("speaker") or "",
                "text": cue.get("text") or "",
            }
        )
    return lines


def evidence_quotes(candidate: dict, dimension: str) -> list[dict]:
    evidence = candidate.get("evidence")
    if not isinstance(evidence, dict):
        return []
    block = evidence.get(dimension)
    if not isinstance(block, dict) or not isinstance(block.get("quotes"), list):
        return []
    return [quote for quote in block["quotes"] if isinstance(quote, dict)]


def decision_widget_index(existing: dict | None) -> int | None:
    """Return no selection until a saved APPROVED, REJECTED, or HOLD exists."""
    if not isinstance(existing, dict):
        return None
    decision = existing.get("decision")
    if decision not in DECISION_CHOICES:
        return None
    return DECISION_CHOICES.index(decision)


def save_is_enabled(choice: str | None) -> bool:
    """The placeholder and an empty choice cannot be saved."""
    return choice in DECISION_CHOICES


def load_decisions(path: Path | str) -> tuple[dict, str | None]:
    file_path = Path(path)
    if not file_path.is_file():
        return {"decisions": {}}, None
    payload, error = read_json(file_path)
    if error:
        return {"decisions": {}}, error
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), dict):
        return {"decisions": {}}, f"{file_path.name} has no decisions object."
    decisions = {}
    for key, value in payload["decisions"].items():
        if not isinstance(value, dict) or value.get("decision") not in DECISION_CHOICES:
            continue
        candidate_id = str(value.get("candidate_id") or key)
        decisions[candidate_id] = {
            "candidate_id": candidate_id,
            "decision": value["decision"],
            "comment": value.get("comment") if isinstance(value.get("comment"), str) else "",
            "updated_at": value.get("updated_at") if isinstance(value.get("updated_at"), str) else "",
        }
    return {"decisions": decisions}, None


def save_decision(
    path: Path | str,
    candidate_id: str,
    decision: str,
    comment: str = "",
    *,
    protected_paths: tuple[Path | str, ...] = (),
    clock=None,
) -> dict:
    """Store one human decision. Generated pipeline files are refused."""
    file_path = Path(path).resolve()
    protected = {Path(item).resolve() for item in protected_paths}
    if file_path in protected:
        raise ValueError("Refusing to modify a generated ranking or validation file.")
    if decision not in DECISION_CHOICES:
        raise ValueError("Decision must be APPROVED, REJECTED, or HOLD.")
    if not str(candidate_id).strip():
        raise ValueError("A candidate ID is required.")
    if file_path.is_file():
        current, error = load_decisions(file_path)
        if error:
            raise ValueError(error)
    else:
        current = {"decisions": {}}
    stamp = clock() if clock else datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    record = {
        "candidate_id": candidate_id,
        "decision": decision,
        "comment": comment or "",
        "updated_at": stamp,
    }
    current["decisions"][candidate_id] = record
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_path.with_suffix(file_path.suffix + ".tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(file_path)
    return record
