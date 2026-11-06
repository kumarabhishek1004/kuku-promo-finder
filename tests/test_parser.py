"""Unit tests for the deterministic screenplay parser."""

from pathlib import Path

import pytest

from src.parser import parse_episode, parse_episode_text, parse_episodes, timestamp_to_ms


ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "data" / "episodes"


def _screenplay(body: str, number: int = 1, catalog: str = "100") -> tuple[str, str]:
    text = (
        f"EPISODE NUMBER: Episode {number}\n"
        "EPISODE LOGLINE: Logline.\n"
        "CONTINUITY NOTES: Notes.\n"
        "\n"
        "--- SCRIPT START ---\n"
        f"{body}"
        "--- SCRIPT END ---\n"
    )
    return text, f"episode-{number:03d}-Episode {catalog}.txt"


def _cues(episode: dict) -> list[dict]:
    return [cue for scene in episode["scenes"] for cue in scene["cues"]]


def _codes(issues: list[dict]) -> list[str]:
    return [issue["code"] for issue in issues]


def test_timestamp_conversion():
    assert timestamp_to_ms("00:00.00") == 0
    assert timestamp_to_ms("00:01.50") == 1_500
    assert timestamp_to_ms("00:09.85") == 9_850
    assert timestamp_to_ms("02:18.00") == 138_000
    assert timestamp_to_ms("03:20.75") == 200_750
    assert timestamp_to_ms(" 00:04.80 ") == 4_800


@pytest.mark.parametrize(
    "value",
    ["00:61.00", "1:00.00", "00:00.5", "00:00:00", "3.5", "", "00:09.850"],
)
def test_invalid_timestamp_strings_raise(value):
    with pytest.raises(ValueError):
        timestamp_to_ms(value)


def test_scene_parsing_preserves_heading_and_bounds():
    episode = parse_episode(EPISODES / "episode-001-Episode 1671703.txt")
    assert episode["errors"] == []
    assert episode["episode_number"] == 1
    assert episode["filename_index"] == 1
    assert episode["catalog_id"] == "1671703"
    assert episode["logline"].startswith("Lady Juliana arrives")
    assert "fiancé" in episode["logline"]
    assert len(episode["scenes"]) == 1
    scene = episode["scenes"][0]
    assert scene["scene_number"] == 1
    assert scene["scene_heading"] == "INT. CATHEDRAL – DAY"
    assert "–" in scene["scene_heading"]
    assert scene["scene_start"] == "00:00.00"
    assert scene["scene_start_ms"] == 0
    assert scene["scene_end"] == "02:18.00"
    assert scene["scene_end_ms"] == 138_000
    assert scene["unparsed_lines"] == []


def test_hyphen_scene_heading_is_not_rewritten():
    episode = parse_episode(EPISODES / "episode-007-Episode 1671709.txt")
    assert episode["scenes"][0]["scene_heading"] == "EXT. WOODLAND COTTAGE - DAY"


def test_camera_action_and_sound_cues():
    episode = parse_episode(EPISODES / "episode-001-Episode 1671703.txt")
    cues = _cues(episode)
    camera, action = cues[0], cues[1]
    assert camera["type"] == "CAMERA"
    assert camera["raw_label"] == "CAMERA"
    assert camera["start"] == "00:00.00"
    assert camera["end"] == "00:04.80"
    assert camera["start_ms"] == 0
    assert camera["end_ms"] == 4_800
    assert camera["text"].startswith("Symmetrical framing")
    assert "speaker" not in camera

    assert action["type"] == "ACTION"
    assert action["start"] == "00:01.00"
    assert action["end_ms"] == 4_800
    assert action["text"].startswith("Juliana pauses at the threshold")

    sound = next(cue for cue in cues if cue["type"] == "SOUND FX")
    assert sound["raw_label"] == "SOUND FX"
    assert sound["start"] == "02:15.00"
    assert sound["end"] == "02:17.50"
    assert sound["start_ms"] == 135_000
    assert sound["end_ms"] == 137_500
    assert sound["text"] == "Solemn low-pitched string drone swells to an ominous cliffhanger peak."


def test_dialogue_parentheticals_and_offscreen_markers():
    former_groom = parse_episode(EPISODES / "episode-002-Episode 1671704.txt")
    split_line = next(
        cue for cue in _cues(former_groom) if cue.get("speaker") == "THE FORMER GROOM" and cue["start"] == "00:06.80"
    )
    assert split_line["type"] == "DIALOGUE"
    assert split_line["parentheticals"] == ["Off-screen", "Dismissive"]
    assert split_line["markers"] == ["Off-screen"]
    assert split_line["text"] == '"...far too silent..."'
    voice = next(cue for cue in _cues(former_groom) if cue.get("speaker") == "OFF-SCREEN VOICE")
    assert voice["markers"] == []
    assert voice["speaker"] == "OFF-SCREEN VOICE"

    episode_one = parse_episode(EPISODES / "episode-001-Episode 1671703.txt")
    matriarch = next(cue for cue in _cues(episode_one) if cue.get("raw_label") == "MATRIARCH (O.S.) (Firm)")
    assert matriarch["speaker"] == "MATRIARCH"
    assert matriarch["parentheticals"] == ["O.S.", "Firm"]
    assert matriarch["markers"] == ["O.S."]

    voiceover = parse_episode(EPISODES / "episode-006-Episode 1671708.txt")
    juliana = next(cue for cue in _cues(voiceover) if cue.get("raw_label") == "JULIANA (V.O.) (Reflective, Vulnerable)")
    assert juliana["speaker"] == "JULIANA"
    assert juliana["parentheticals"] == ["V.O.", "Reflective, Vulnerable"]
    assert juliana["markers"] == ["V.O."]
    assert juliana["text"] == '"I lost my home today."'

    embedded = parse_episode(EPISODES / "episode-022-Episode 1678575.txt")
    housekeeper = next(cue for cue in _cues(embedded) if cue.get("speaker") == "HOUSEKEEPER")
    assert housekeeper["parentheticals"] == ["Sharp, Stern, O.S."]
    assert housekeeper["markers"] == ["O.S."]
    assert housekeeper["text"] == '"Prudence."'

    episode_five = parse_episode(EPISODES / "episode-005-Episode 1671707.txt")
    continues = next(cue for cue in _cues(episode_five) if "O.S. / Continues" in cue.get("raw_label", ""))
    assert continues["markers"] == ["O.S."]
    assert continues["speaker"] == "LORD ASHCROFT"


def test_uppercase_scene_metadata():
    episode = parse_episode(EPISODES / "episode-006-Episode 1671708.txt")
    assert episode["errors"] == []
    assert len(episode["scenes"]) == 2
    first, second = episode["scenes"]
    assert first["scene_heading"] == "EXT. WOODLAND PATH – NIGHT"
    assert first["scene_start"] == "00:00.00"
    assert first["scene_end"] == "00:33.70"
    assert first["scene_end_ms"] == 33_700
    assert second["scene_number"] == 2
    assert second["scene_start_ms"] == 33_700
    hundredths = next(cue for cue in first["cues"] if cue["end"] == "00:09.85")
    assert hundredths["end_ms"] == 9_850
    assert hundredths["text"] == '"I know this is not the life you came from."'


def test_overlapping_cues_are_preserved():
    episode = parse_episode(EPISODES / "episode-001-Episode 1671703.txt")
    camera, action = _cues(episode)[:2]
    assert camera["start_ms"] < action["start_ms"] < camera["end_ms"]
    assert action["end_ms"] == camera["end_ms"]
    assert episode["errors"] == []
    assert episode["warnings"] == []


def test_em_dash_dialogue_is_preserved():
    episode = parse_episode(EPISODES / "episode-024-Episode 1678577.txt")
    cutoff = next(cue for cue in _cues(episode) if cue["text"] == '"My lady—"')
    assert cutoff["speaker"] == "PRUDENCE"
    assert "\u2014" in cutoff["text"]


def test_character_names_are_not_merged():
    parsed = parse_episodes(EPISODES)
    speakers = {
        cue["speaker"]
        for episode in parsed["episodes"]
        for cue in _cues(episode)
        if cue["type"] == "DIALOGUE"
    }
    assert "JULIANA" in speakers
    assert "LADY JULIANA" in speakers
    assert "THE FORMER GROOM" in speakers
    assert "ALEXANDER'S MOTHER" in speakers
    assert "MYSTERIOUS BEARDED MAN" in speakers
    episode_one_speakers = {
        cue["speaker"]
        for cue in _cues(parsed["episodes"][0])
        if cue["type"] == "DIALOGUE"
    }
    assert "THE FORMER GROOM" in episode_one_speakers
    assert "GERALD" not in episode_one_speakers


def test_invalid_timestamps_keep_the_cue():
    text, filename = _screenplay(
        "\n".join(
            [
                "SCENE 01",
                "Scene heading: INT. ROOM – DAY",
                "Scene start: 00:00.00",
                "Scene end: 00:10.00",
                "",
                "Actions and dialogue:",
                "",
                "**CAMERA:**",
                "Start: (00:00.00) | End: (00:61.00)",
                "A shot that cannot exist.",
                "",
                "**ACTION:**",
                "Start: (00:05.00) | End: (00:04.00)",
                "The action runs backward.",
                "",
                "This line is not a cue.",
                "",
            ]
        )
    )
    episode = parse_episode_text(text, filename)
    cues = _cues(episode)
    assert [cue["type"] for cue in cues] == ["CAMERA", "ACTION"]
    assert cues[0]["text"] == "A shot that cannot exist."
    assert cues[0]["end"] == "00:61.00"
    assert cues[0]["start_ms"] == 0
    assert cues[0]["end_ms"] is None
    assert cues[1]["text"] == "The action runs backward."
    assert cues[1]["start_ms"] == 5_000
    assert cues[1]["end_ms"] == 4_000
    assert episode["scenes"][0]["unparsed_lines"] == [{"line": 21, "text": "This line is not a cue."}]
    assert "invalid_timestamp" in _codes(episode["errors"])
    assert "cue_end_before_start" in _codes(episode["errors"])
    assert "unparsed_line" in _codes(episode["errors"])


def test_unjoined_scene_boundaries_are_errors_and_are_not_rewritten():
    text, filename = _screenplay(
        "\n".join(
            [
                "SCENE 01",
                "SCENE HEADING: INT. ROOM - DAY",
                "SCENE START: 00:00.00",
                "SCENE END: 00:05.00",
                "",
                "ACTIONS AND DIALOGUE:",
                "",
                "**CAMERA:**",
                "Start: (00:00.00) | End: (00:05.00)",
                "First room.",
                "",
                "SCENE 03",
                "Scene heading: EXT. YARD – DAY",
                "Scene start: 00:08.00",
                "Scene end: 00:09.00",
                "",
                "Actions and dialogue:",
                "",
                "**ACTION:**",
                "Start: (00:08.00) | End: (00:09.00)",
                "Outside.",
                "",
            ]
        )
    )
    episode = parse_episode_text(text, filename)
    assert episode["scenes"][0]["scene_heading"] == "INT. ROOM - DAY"
    assert episode["scenes"][0]["scene_end"] == "00:05.00"
    assert episode["scenes"][1]["scene_number"] == 3
    assert episode["scenes"][1]["scene_start"] == "00:08.00"
    assert "scene_number_out_of_sequence" in _codes(episode["errors"])
    assert "scene_boundary_not_joined" in _codes(episode["errors"])


def test_episode_22_boundary_warnings_do_not_change_timestamps():
    episode = parse_episode(EPISODES / "episode-022-Episode 1678575.txt")
    assert episode["episode_number"] == 22
    assert episode["catalog_id"] == "1678575"
    assert episode["errors"] == []
    scene_one, scene_two = episode["scenes"][0], episode["scenes"][1]
    assert scene_one["scene_end"] == "00:42.50"
    assert scene_one["scene_end_ms"] == 42_500
    overrun = next(cue for cue in scene_one["cues"] if cue["end"] == "00:42.80")
    assert overrun["type"] == "SOUND FX"
    assert overrun["end_ms"] == 42_800
    assert overrun["text"] == "Heavy wooden latch clicks shut with a thud."
    overrun_warnings = [
        issue
        for issue in episode["warnings"]
        if issue["code"] == "cue_ends_after_scene" and issue["scene_number"] == 1
    ]
    assert len(overrun_warnings) == 1
    assert "00:42.80" in overrun_warnings[0]["message"]
    assert "00:42.50" in overrun_warnings[0]["message"]

    assert scene_two["scene_start"] == "00:42.50"
    assert scene_two["scene_start_ms"] == 42_500
    first_cue = scene_two["cues"][0]
    assert first_cue["start"] == "00:43.00"
    assert first_cue["start_ms"] == 43_000
    late_start = [
        issue
        for issue in episode["warnings"]
        if issue["code"] == "scene_starts_before_first_cue" and issue["scene_number"] == 2
    ]
    assert len(late_start) == 1
    assert "00:43.00" in late_start[0]["message"]


def test_all_fifty_episodes_parse_without_dropping_cues():
    parsed = parse_episodes(EPISODES)
    summary = parsed["validation"]
    assert summary["episodes_parsed"] == 50
    assert summary["missing_episodes"] == []
    assert summary["unexpected_episodes"] == []
    assert summary["scenes"] == 100
    assert summary["cues"] == 3727
    assert summary["cue_counts_by_type"] == {
        "CAMERA": 1754,
        "ACTION": 372,
        "SOUND FX": 114,
        "DIALOGUE": 1487,
    }
    assert summary["errors"] == []
    assert [episode["episode_number"] for episode in parsed["episodes"]] == list(range(1, 51))

    for episode in parsed["episodes"]:
        assert episode["episode_number"] == episode["filename_index"]
        assert episode["catalog_id"].isdigit()
        source = (EPISODES / episode["filename"]).read_text(encoding="utf-8")
        bold_cues = sum(1 for line in source.splitlines() if line.startswith("**"))
        assert len(_cues(episode)) == bold_cues
        assert episode["unparsed_lines"] == []
        assert all(scene["unparsed_lines"] == [] for scene in episode["scenes"])
        numbers = [scene["scene_number"] for scene in episode["scenes"]]
        assert numbers == list(range(1, len(numbers) + 1))
        for earlier, later in zip(episode["scenes"], episode["scenes"][1:]):
            assert earlier["scene_end_ms"] == later["scene_start_ms"]

    warning_codes = {(issue["episode_number"], issue["scene_number"], issue["code"]) for issue in summary["warnings"]}
    assert (22, 1, "cue_ends_after_scene") in warning_codes
    assert (22, 2, "scene_starts_before_first_cue") in warning_codes
