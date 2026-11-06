"""Deterministic parser for Kuku microdrama screenplays.

The parser reads the custom text format in ``data/episodes``. It does not
estimate times, merge character aliases, or rewrite source files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EXPECTED_EPISODE_COUNT = 50
CUE_TYPES = ("CAMERA", "ACTION", "SOUND FX", "DIALOGUE")
STRUCTURAL_TYPES = frozenset({"CAMERA", "ACTION", "SOUND FX"})

_FILENAME_RE = re.compile(r"^episode-(\d+)-Episode (\d+)\.txt$")
_SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$")
_BOLD_RE = re.compile(r"^\*\*(.+):\*\*\s*$")
_TIMESTAMP_LINE_RE = re.compile(
    r"^Start:\s*\(([^)]+)\)\s*\|\s*End:\s*\(([^)]+)\)\s*$"
)
_TIMESTAMP_RE = re.compile(r"^(\d{2}):(\d{2})\.(\d{2})$")
_PAREN_RE = re.compile(r"\(([^)]*)\)")
_MARKERS = (
    ("O.S.", re.compile(r"O\.S\.", re.IGNORECASE)),
    ("V.O.", re.compile(r"V\.O\.", re.IGNORECASE)),
    ("Off-screen", re.compile(r"Off-screen", re.IGNORECASE)),
)
_META_KEYS = frozenset(
    {"SCENE HEADING", "SCENE START", "SCENE END", "ACTIONS AND DIALOGUE"}
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp_to_ms(value: str) -> int:
    """Convert ``MM:SS.CC`` to integer milliseconds.

    ``CC`` is centiseconds (hundredths of a second). Minutes may exceed 59
    because the screenplays write durations as ``03:20.75``. Seconds may not.
    """
    text = value.strip()
    match = _TIMESTAMP_RE.match(text)
    if match is None:
        raise ValueError(f"Timestamp {value!r} is not MM:SS.CC.")
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    centiseconds = int(match.group(3))
    if seconds > 59:
        raise ValueError(f"Timestamp {value!r} has seconds greater than 59.")
    return ((minutes * 60) + seconds) * 1000 + centiseconds * 10


def parse_episodes(directory: Path | None = None) -> dict:
    """Parse every episode file in ``directory`` and validate the set of 50."""
    source = Path(directory) if directory is not None else repo_root() / "data" / "episodes"
    files = sorted(path for path in source.glob("episode-*.txt") if path.is_file())
    episodes = [parse_episode(path) for path in files]
    episodes.sort(key=lambda episode: (episode["filename_index"] is None, episode["filename_index"] or 0))
    return {
        "episodes": episodes,
        "validation": _validation_summary(episodes),
    }


def parse_episode(path: Path) -> dict:
    """Parse one screenplay file into a dictionary."""
    file_path = Path(path)
    filename = file_path.name
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        episode = _empty_episode(filename)
        _add(
            episode,
            "errors",
            "file_unreadable",
            f"Could not read {filename}: {exc}",
        )
        return episode
    return parse_episode_text(text, filename)


def parse_episode_text(text: str, filename: str) -> dict:
    """Parse screenplay text. ``filename`` supplies the catalog id and index."""
    episode = _empty_episode(filename)
    index, catalog_id = _filename_parts(filename)
    episode["filename_index"] = index
    episode["catalog_id"] = catalog_id
    if index is None or catalog_id is None:
        _add(
            episode,
            "errors",
            "filename_unrecognized",
            f"Filename {filename!r} does not match episode-NNN-Episode CATALOG.txt.",
        )

    lines = _numbered_lines(text)
    header_end = _parse_header(lines, episode)
    body, footer = _split_script(lines[header_end:], episode)
    scenes = _split_scenes(body, episode)
    for scene_lines in scenes:
        episode["scenes"].append(_parse_scene(scene_lines, episode))
    if footer:
        for line_no, line in footer:
            if line.strip() == "":
                continue
            episode["unparsed_lines"].append({"line": line_no, "text": line})
            _add(
                episode,
                "errors",
                "unparsed_line",
                f"Line {line_no} appears after the script end marker.",
                line=line_no,
                text=line,
            )
    _validate_episode_structure(episode)
    return episode


def main(argv: list[str] | None = None) -> int:
    """Parse all episodes and write ``outputs/parsed_episodes.json``."""
    root = repo_root()
    argument_parser = argparse.ArgumentParser(description="Parse Kuku screenplay episodes.")
    argument_parser.add_argument(
        "--input",
        default=str(root / "data" / "episodes"),
        help="Directory containing episode-*.txt screenplays.",
    )
    argument_parser.add_argument(
        "--output",
        default=str(root / "outputs" / "parsed_episodes.json"),
        help="Path of the JSON file to write.",
    )
    args = argument_parser.parse_args(argv)
    result = parse_episodes(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = result["validation"]
    print(
        f"Parsed {summary['episodes_parsed']} episodes, "
        f"{summary['scenes']} scenes, {summary['cues']} cues."
    )
    print(f"Errors: {len(summary['errors'])}")
    print(f"Warnings: {len(summary['warnings'])}")
    print(f"Wrote {output}")
    return 0 if not summary["errors"] else 1


def _empty_episode(filename: str) -> dict:
    return {
        "episode_number": None,
        "filename_index": None,
        "catalog_id": None,
        "filename": filename,
        "logline": None,
        "continuity_notes": None,
        "scenes": [],
        "unparsed_lines": [],
        "errors": [],
        "warnings": [],
    }


def _filename_parts(filename: str) -> tuple[int | None, str | None]:
    match = _FILENAME_RE.match(filename)
    if match is None:
        return None, None
    return int(match.group(1)), match.group(2)


def _numbered_lines(text: str) -> list[tuple[int, str]]:
    raw_lines = text.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()
    return [(index + 1, line.rstrip("\r")) for index, line in enumerate(raw_lines)]


def _parse_header(lines: list[tuple[int, str]], episode: dict) -> int:
    expected = (
        ("EPISODE NUMBER", "episode_number_line"),
        ("EPISODE LOGLINE", "logline"),
        ("CONTINUITY NOTES", "continuity_notes"),
    )
    cursor = 0
    for key, field in expected:
        while cursor < len(lines) and lines[cursor][1].strip() == "":
            cursor += 1
        if cursor >= len(lines):
            _add(episode, "errors", "missing_header", f"Missing header field {key}.")
            continue
        line_no, line = lines[cursor]
        value = _header_value(line, key)
        if value is None:
            _add(
                episode,
                "errors",
                "missing_header",
                f"Line {line_no} is not the {key} header.",
                line=line_no,
                text=line,
            )
            cursor += 1
            continue
        if field == "episode_number_line":
            number_match = re.search(r"(\d+)", value)
            if number_match is None:
                _add(
                    episode,
                    "errors",
                    "missing_episode_number",
                    f"Header {value!r} does not contain an episode number.",
                    line=line_no,
                )
            else:
                episode["episode_number"] = int(number_match.group(1))
        else:
            episode[field] = value
        cursor += 1

    if (
        episode["episode_number"] is not None
        and episode["filename_index"] is not None
        and episode["episode_number"] != episode["filename_index"]
    ):
        _add(
            episode,
            "errors",
            "episode_number_mismatch",
            (
                f"Header episode {episode['episode_number']} does not match "
                f"filename index {episode['filename_index']}."
            ),
        )
    return cursor


def _header_value(line: str, key: str) -> str | None:
    label, separator, value = line.partition(":")
    if separator != ":" or label.strip().upper() != key:
        return None
    return value.strip()


def _split_script(
    lines: list[tuple[int, str]],
    episode: dict,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    start_at = next((index for index, (_, line) in enumerate(lines) if line.strip() == "--- SCRIPT START ---"), None)
    end_at = next((index for index, (_, line) in enumerate(lines) if line.strip() == "--- SCRIPT END ---"), None)
    if start_at is None:
        _add(episode, "errors", "missing_script_start", "Missing --- SCRIPT START --- marker.")
        return [], []
    if end_at is None:
        _add(episode, "errors", "missing_script_end", "Missing --- SCRIPT END --- marker.")
        return lines[start_at + 1 :], []
    if end_at < start_at:
        _add(episode, "errors", "script_markers_out_of_order", "Script end marker appears before the start marker.")
        return [], lines
    between = lines[start_at + 1 : end_at]
    return between, lines[end_at + 1 :]


def _split_scenes(
    lines: list[tuple[int, str]],
    episode: dict,
) -> list[list[tuple[int, str]]]:
    scenes: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] | None = None
    for line_no, line in lines:
        if _SCENE_RE.match(line):
            if current is not None:
                scenes.append(current)
            current = [(line_no, line)]
            continue
        if current is None:
            if line.strip() == "":
                continue
            episode["unparsed_lines"].append({"line": line_no, "text": line})
            _add(
                episode,
                "errors",
                "unparsed_line",
                f"Line {line_no} appears before the first scene.",
                line=line_no,
                text=line,
            )
            continue
        current.append((line_no, line))
    if current is not None:
        scenes.append(current)
    if not scenes and not any(error["code"] == "missing_script_start" for error in episode["errors"]):
        _add(episode, "errors", "missing_scenes", "Script contains no SCENE blocks.")
    return scenes


def _parse_scene(lines: list[tuple[int, str]], episode: dict) -> dict:
    scene_match = _SCENE_RE.match(lines[0][1])
    scene_number = int(scene_match.group(1)) if scene_match else None
    scene = {
        "scene_number": scene_number,
        "scene_heading": None,
        "scene_start": None,
        "scene_start_ms": None,
        "scene_end": None,
        "scene_end_ms": None,
        "cues": [],
        "unparsed_lines": [],
    }
    metadata_done = False
    cursor = 1
    while cursor < len(lines) and not metadata_done:
        line_no, line = lines[cursor]
        if line.strip() == "":
            cursor += 1
            continue
        key, value = _meta_field(line)
        if key is None:
            scene["unparsed_lines"].append({"line": line_no, "text": line})
            _add(
                episode,
                "errors",
                "unparsed_line",
                f"Line {line_no} is not a scene field or cue in scene {scene_number}.",
                scene_number=scene_number,
                line=line_no,
                text=line,
            )
            cursor += 1
            continue
        if key == "SCENE HEADING":
            scene["scene_heading"] = value
        elif key == "SCENE START":
            scene["scene_start"] = value
            scene["scene_start_ms"] = _optional_ms(value, episode, scene_number, line_no, "scene start")
        elif key == "SCENE END":
            scene["scene_end"] = value
            scene["scene_end_ms"] = _optional_ms(value, episode, scene_number, line_no, "scene end")
        elif key == "ACTIONS AND DIALOGUE":
            metadata_done = True
        cursor += 1

    for field, label in (
        ("scene_heading", "scene heading"),
        ("scene_start", "scene start"),
        ("scene_end", "scene end"),
    ):
        if scene[field] is None:
            _add(
                episode,
                "errors",
                "missing_scene_field",
                f"Scene {scene_number} is missing its {label}.",
                scene_number=scene_number,
            )
    if not metadata_done:
        _add(
            episode,
            "errors",
            "missing_scene_field",
            f"Scene {scene_number} is missing the actions-and-dialogue header.",
            scene_number=scene_number,
        )

    scene["cues"] = _parse_cues(lines[cursor:], episode, scene, scene_number)
    _validate_scene_bounds(scene, episode)
    return scene


def _meta_field(line: str) -> tuple[str | None, str | None]:
    label, separator, value = line.partition(":")
    if separator != ":":
        return None, None
    key = label.strip().upper()
    if key not in _META_KEYS:
        return None, None
    return key, value.strip()


def _parse_cues(
    lines: list[tuple[int, str]],
    episode: dict,
    scene: dict,
    scene_number: int | None,
) -> list[dict]:
    cues: list[dict] = []
    cursor = 0
    while cursor < len(lines):
        line_no, line = lines[cursor]
        if line.strip() == "":
            cursor += 1
            continue
        bold = _BOLD_RE.match(line)
        if bold is None:
            _keep_unparsed(episode, scene, scene_number, line_no, line)
            cursor += 1
            continue
        raw_label = bold.group(1).strip()
        timestamp_line = lines[cursor + 1] if cursor + 1 < len(lines) else None
        timestamp_match = _TIMESTAMP_LINE_RE.match(timestamp_line[1]) if timestamp_line else None
        if timestamp_match is None:
            cue = _cue_shell(raw_label, line_no)
            cues.append(cue)
            _add(
                episode,
                "errors",
                "missing_timestamp",
                f"Cue {raw_label!r} on line {line_no} has no Start/End timestamp.",
                scene_number=scene_number,
                line=line_no,
            )
            cursor += 1
            continue
        start_text, end_text = timestamp_match.group(1).strip(), timestamp_match.group(2).strip()
        content_index = cursor + 2
        content = lines[content_index] if content_index < len(lines) else None
        if content is None or content[1].strip() == "" or _BOLD_RE.match(content[1]):
            cue = _build_cue(raw_label, line_no, start_text, end_text, "")
            cues.append(cue)
            _finish_cue_times(cue, episode, scene_number, line_no)
            _add(
                episode,
                "errors",
                "missing_cue_text",
                f"Cue {raw_label!r} on line {line_no} has no text line.",
                scene_number=scene_number,
                line=line_no,
            )
            cursor = content_index
            continue
        cue = _build_cue(raw_label, line_no, start_text, end_text, content[1])
        cues.append(cue)
        _finish_cue_times(cue, episode, scene_number, line_no)
        cursor = content_index + 1
    return cues


def _cue_shell(raw_label: str, line_no: int) -> dict:
    cue = {
        "type": _cue_type(raw_label),
        "raw_label": raw_label,
        "text": None,
        "start": None,
        "start_ms": None,
        "end": None,
        "end_ms": None,
        "line": line_no,
    }
    if cue["type"] == "DIALOGUE":
        _add_dialogue_fields(cue, raw_label)
    return cue


def _build_cue(
    raw_label: str,
    line_no: int,
    start_text: str,
    end_text: str,
    text: str,
) -> dict:
    cue = {
        "type": _cue_type(raw_label),
        "raw_label": raw_label,
        "text": text,
        "start": start_text,
        "start_ms": None,
        "end": end_text,
        "end_ms": None,
        "line": line_no,
    }
    if cue["type"] == "DIALOGUE":
        _add_dialogue_fields(cue, raw_label)
    return cue


def _cue_type(raw_label: str) -> str:
    if raw_label in STRUCTURAL_TYPES:
        return raw_label
    return "DIALOGUE"


def _add_dialogue_fields(cue: dict, raw_label: str) -> None:
    parentheticals = _PAREN_RE.findall(raw_label)
    speaker = _PAREN_RE.sub("", raw_label).strip()
    cue["speaker"] = speaker
    cue["parentheticals"] = parentheticals
    cue["markers"] = _markers_in(parentheticals)


def _markers_in(parentheticals: list[str]) -> list[str]:
    found: list[str] = []
    for parenthetical in parentheticals:
        for name, pattern in _MARKERS:
            if pattern.search(parenthetical):
                found.append(name)
    return found


def _finish_cue_times(cue: dict, episode: dict, scene_number: int | None, line_no: int) -> None:
    cue["start_ms"] = _optional_ms(cue["start"], episode, scene_number, line_no, "cue start")
    cue["end_ms"] = _optional_ms(cue["end"], episode, scene_number, line_no, "cue end")
    if cue["start_ms"] is not None and cue["end_ms"] is not None and cue["end_ms"] < cue["start_ms"]:
        _add(
            episode,
            "errors",
            "cue_end_before_start",
            (
                f"Scene {scene_number} cue {cue['raw_label']!r} ends at {cue['end']} "
                f"before it starts at {cue['start']}."
            ),
            scene_number=scene_number,
            line=line_no,
        )


def _optional_ms(
    value: str | None,
    episode: dict,
    scene_number: int | None,
    line_no: int,
    role: str,
) -> int | None:
    if value is None:
        return None
    try:
        return timestamp_to_ms(value)
    except ValueError as exc:
        _add(
            episode,
            "errors",
            "invalid_timestamp",
            f"Scene {scene_number} {role} is invalid: {exc}",
            scene_number=scene_number,
            line=line_no,
        )
        return None


def _validate_scene_bounds(scene: dict, episode: dict) -> None:
    start_ms = scene["scene_start_ms"]
    end_ms = scene["scene_end_ms"]
    scene_number = scene["scene_number"]
    if start_ms is not None and end_ms is not None and end_ms < start_ms:
        _add(
            episode,
            "errors",
            "scene_end_before_start",
            (
                f"Scene {scene_number} ends at {scene['scene_end']} "
                f"before it starts at {scene['scene_start']}."
            ),
            scene_number=scene_number,
        )
    timed = [
        cue
        for cue in scene["cues"]
        if cue["start_ms"] is not None and cue["end_ms"] is not None and cue["end_ms"] >= cue["start_ms"]
    ]
    if start_ms is None or end_ms is None or not timed:
        return
    for cue in timed:
        if cue["start_ms"] < start_ms:
            _add(
                episode,
                "warnings",
                "cue_starts_before_scene",
                (
                    f"Scene {scene_number} cue {cue['raw_label']!r} starts at {cue['start']}, "
                    f"before the scene start {scene['scene_start']}."
                ),
                scene_number=scene_number,
                line=cue["line"],
            )
        if cue["end_ms"] > end_ms:
            _add(
                episode,
                "warnings",
                "cue_ends_after_scene",
                (
                    f"Scene {scene_number} cue {cue['raw_label']!r} ends at {cue['end']}, "
                    f"after the scene end {scene['scene_end']}."
                ),
                scene_number=scene_number,
                line=cue["line"],
            )
    earliest = min(cue["start_ms"] for cue in timed)
    latest = max(cue["end_ms"] for cue in timed)
    if earliest > start_ms:
        first = min(timed, key=lambda cue: (cue["start_ms"], cue["line"]))
        _add(
            episode,
            "warnings",
            "scene_starts_before_first_cue",
            (
                f"Scene {scene_number} starts at {scene['scene_start']}, "
                f"before the first cue at {first['start']}."
            ),
            scene_number=scene_number,
            line=first["line"],
        )
    if latest < end_ms:
        last = max(timed, key=lambda cue: (cue["end_ms"], cue["line"]))
        _add(
            episode,
            "warnings",
            "scene_ends_after_last_cue",
            (
                f"Scene {scene_number} ends at {scene['scene_end']}, "
                f"after the last cue at {last['end']}."
            ),
            scene_number=scene_number,
            line=last["line"],
        )


def _validate_episode_structure(episode: dict) -> None:
    expected = 1
    previous = None
    for scene in episode["scenes"]:
        number = scene["scene_number"]
        if number != expected:
            _add(
                episode,
                "errors",
                "scene_number_out_of_sequence",
                f"Expected scene {expected:02d}, found scene {number}.",
                scene_number=number,
            )
        if (
            previous is not None
            and previous["scene_end_ms"] is not None
            and scene["scene_start_ms"] is not None
            and previous["scene_end_ms"] != scene["scene_start_ms"]
        ):
            _add(
                episode,
                "errors",
                "scene_boundary_not_joined",
                (
                    f"Scene {previous['scene_number']} ends at {previous['scene_end']} "
                    f"but scene {number} starts at {scene['scene_start']}."
                ),
                scene_number=number,
            )
        previous = scene
        expected += 1


def _keep_unparsed(
    episode: dict,
    scene: dict | None,
    scene_number: int | None,
    line_no: int,
    line: str,
) -> None:
    target = scene["unparsed_lines"] if scene is not None else episode["unparsed_lines"]
    target.append({"line": line_no, "text": line})
    _add(
        episode,
        "errors",
        "unparsed_line",
        f"Line {line_no} in scene {scene_number} was not recognized and was kept unparsed.",
        scene_number=scene_number,
        line=line_no,
        text=line,
    )


def _validation_summary(episodes: list[dict]) -> dict:
    counts = {cue_type: 0 for cue_type in CUE_TYPES}
    scenes = 0
    cues = 0
    errors: list[dict] = []
    warnings: list[dict] = []
    present: set[int] = set()
    for episode in episodes:
        index = episode["filename_index"]
        if index is not None:
            present.add(index)
        scenes += len(episode["scenes"])
        for scene in episode["scenes"]:
            for cue in scene["cues"]:
                cues += 1
                counts[cue["type"]] = counts.get(cue["type"], 0) + 1
        errors.extend(episode["errors"])
        warnings.extend(episode["warnings"])

    expected = set(range(1, EXPECTED_EPISODE_COUNT + 1))
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    for number in missing:
        errors.append(
            {
                "code": "missing_episode",
                "episode_number": number,
                "filename_index": number,
                "scene_number": None,
                "line": None,
                "message": f"Episode {number} is not present in the input directory.",
            }
        )
    for number in unexpected:
        errors.append(
            {
                "code": "unexpected_episode",
                "episode_number": number,
                "filename_index": number,
                "scene_number": None,
                "line": None,
                "message": f"Filename index {number} is outside episodes 1-{EXPECTED_EPISODE_COUNT}.",
            }
        )
    return {
        "episodes_parsed": len(episodes),
        "expected_episodes": EXPECTED_EPISODE_COUNT,
        "missing_episodes": missing,
        "unexpected_episodes": unexpected,
        "scenes": scenes,
        "cues": cues,
        "cue_counts_by_type": counts,
        "errors": errors,
        "warnings": warnings,
    }


def _add(
    episode: dict,
    bucket: str,
    code: str,
    message: str,
    scene_number: int | None = None,
    line: int | None = None,
    text: str | None = None,
) -> None:
    issue = {
        "code": code,
        "episode_number": episode.get("episode_number"),
        "filename_index": episode.get("filename_index"),
        "scene_number": scene_number,
        "line": line,
        "message": message,
    }
    if text is not None:
        issue["text"] = text
    episode[bucket].append(issue)


if __name__ == "__main__":
    raise SystemExit(main())
