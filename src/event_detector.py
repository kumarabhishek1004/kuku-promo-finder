"""Lexical heuristics for dramatic screenplay moments.

These rules match words, punctuation, and parenthetical delivery notes. They do
not track the plot, resolve character aliases, or decide which moment is the
best promo. A cue becomes a moment only when the matched signals reach
``MOMENT_SCORE``.
"""

from __future__ import annotations

import re


MOMENT_SCORE = 3

# One hit from a strong list is enough. Medium hits must combine with another
# signal so ordinary lines are not all treated as dramatic.
_STRONG_PAREN = (
    "screaming",
    "furious",
    "heartbroken",
    "devastated",
    "horrified",
    "sobbing",
    "hysterical",
    "seething",
    "tormented",
    "roaring",
    "panicked",
    "enraged",
    "venomous",
    "vicious",
    "shattered",
    "murderous",
    "outraged",
    "weeping",
    "fury",
    "lethal",
    "deadly",
    "sinister",
    "vindictive",
    "malicious",
    "ruthless",
    "chilling",
    "vengeance",
)
_MEDIUM_PAREN = (
    "desperate",
    "gasping",
    "taunting",
    "gloating",
    "scornful",
    "mocking",
    "cruel",
    "threatening",
    "trembling",
    "betrayed",
    "shocked",
    "snarling",
    "pleading",
    "crying",
    "angry",
    "shouting",
)
_THREAT_TERMS = (
    "kill",
    "killing",
    "killed",
    "murder",
    "murdered",
    "poison",
    "poisoned",
    "arsenic",
    "arrest",
    "arrested",
    "hang",
    "hanged",
    "execute",
    "execution",
    "traitor",
    "impostor",
    "imposter",
    "whip",
    "sword",
    "knife",
    "blood",
    "burn",
    "burning",
    "chains",
    "manacles",
    "strangle",
    "dead",
    "prisoner",
    "how dare you",
    "get out",
    "leave her",
    "don't touch",
    "do not touch",
)
_REVEAL_TERMS = (
    "secret",
    "secrets",
    "betrayed",
    "betrayal",
    "lied",
    "liar",
    "the truth",
    "rightful",
    "illegitimate",
    "heir",
    "mistress",
    "affair",
    "i changed my mind",
    "never going to marry",
)
_ACTION_TERMS = (
    "slap",
    "slaps",
    "slapped",
    "strike",
    "strikes",
    "struck",
    "grab",
    "grabs",
    "grips",
    "gripped",
    "blood",
    "wound",
    "sword",
    "knife",
    "gun",
    "flame",
    "flames",
    "fire",
    "poison",
    "chain",
    "chains",
    "manacle",
    "collapse",
    "collapses",
    "scream",
    "screams",
    "whip",
    "crop",
    "laceration",
    "stab",
    "stabs",
    "stabbed",
    "punch",
    "punches",
    "kiss",
    "kisses",
    "embrace",
    "shove",
    "shoves",
    "choke",
    "chokes",
    "kneel",
    "kneels",
)
_MARKER_ONLY = re.compile(r"^(?:O\.S\.|V\.O\.|Off-screen)$", re.IGNORECASE)
_QUESTION_WORD = re.compile(r"\b(?:who|why|what|where|how|whose)\b", re.IGNORECASE)


def detect_moments(episode: dict) -> list[dict]:
    """Return dramatic dialogue and action cues for one parsed episode."""
    if not isinstance(episode, dict):
        return []
    moments = []
    for cue in _spine(episode):
        scored = score_cue(cue)
        if scored["score"] < MOMENT_SCORE:
            continue
        moments.append(
            {
                "episode_number": episode.get("episode_number"),
                "scene_number": cue.get("scene_number"),
                "scene_heading": cue.get("scene_heading"),
                "spine_index": cue["spine_index"],
                "categories": scored["categories"],
                "signals": scored["signals"],
                "score": scored["score"],
                "cue": cue,
            }
        )
    return moments


def score_cue(cue: dict) -> dict:
    """Score one cue. The result lists every signal that contributed."""
    if not isinstance(cue, dict):
        return {"score": 0, "categories": [], "signals": []}
    text = _spoken(cue.get("text"))
    lowered = text.lower()
    categories: list[str] = []
    signals: list[str] = []
    score = 0

    if cue.get("type") == "ACTION":
        found = _find_terms(lowered, _ACTION_TERMS)
        if found:
            categories.append("physical_action")
            signals.extend(f"action:{term}" for term in found)
            score += 3
    elif cue.get("type") == "DIALOGUE":
        threats = _find_terms(lowered, _THREAT_TERMS)
        if threats:
            categories.append("threat")
            signals.extend(f"keyword:{term}" for term in threats)
            score += 3
        reveals = _find_terms(lowered, _REVEAL_TERMS)
        if reveals:
            categories.append("reveal")
            signals.extend(f"keyword:{term}" for term in reveals)
            score += 3
        paren_score, paren_signals = _parenthetical_signals(cue.get("parentheticals") or [])
        if paren_score:
            categories.append("intense_delivery")
            signals.extend(paren_signals)
            score += paren_score
        if text.endswith("?") and _QUESTION_WORD.search(text):
            categories.append("suspenseful_question")
            signals.append("question")
            score += 2
        interruption = _interruption(text)
        if interruption:
            categories.append("interrupted_dialogue")
            signals.append(interruption)
            score += 3 if interruption == "trailing_interruption" else 1

    return {
        "score": score,
        "categories": categories,
        "signals": signals,
    }


def _spine(episode: dict) -> list[dict]:
    scenes = episode.get("scenes")
    if not isinstance(scenes, list):
        return []
    spine = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        cues = scene.get("cues")
        if not isinstance(cues, list):
            continue
        for cue in cues:
            if not isinstance(cue, dict) or cue.get("type") not in {"DIALOGUE", "ACTION"}:
                continue
            if not _timed(cue):
                continue
            annotated = dict(cue)
            annotated["scene_number"] = scene.get("scene_number")
            annotated["scene_heading"] = scene.get("scene_heading")
            annotated["spine_index"] = len(spine)
            spine.append(annotated)
    spine.sort(key=lambda cue: (cue["start_ms"], cue["end_ms"], cue.get("line") or 0))
    for index, cue in enumerate(spine):
        cue["spine_index"] = index
    return spine


def _timed(cue: dict) -> bool:
    start = cue.get("start_ms")
    end = cue.get("end_ms")
    return isinstance(start, int) and isinstance(end, int) and end >= start


def _spoken(text: object) -> str:
    if not isinstance(text, str):
        return ""
    spoken = text.strip()
    if len(spoken) >= 2 and spoken[0] == '"' and spoken[-1] == '"':
        return spoken[1:-1]
    return spoken


def _find_terms(lowered: str, terms: tuple[str, ...]) -> list[str]:
    found = []
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, lowered, re.IGNORECASE):
            found.append(term)
    return found


def _parenthetical_signals(parentheticals: list) -> tuple[int, list[str]]:
    words = []
    for parenthetical in parentheticals:
        if not isinstance(parenthetical, str) or _MARKER_ONLY.match(parenthetical.strip()):
            continue
        pieces = re.split(r"[,/]", parenthetical)
        for piece in pieces:
            cleaned = piece.strip()
            if cleaned and not _MARKER_ONLY.match(cleaned):
                words.append(cleaned.lower())
    blob = " ".join(words)
    for word in _STRONG_PAREN:
        if re.search(r"\b" + re.escape(word) + r"\b", blob):
            return 3, [f"parenthetical:{word}"]
    for word in _MEDIUM_PAREN:
        if re.search(r"\b" + re.escape(word) + r"\b", blob):
            return 2, [f"parenthetical:{word}"]
    return 0, []


def _interruption(text: str) -> str | None:
    stripped = text.rstrip()
    if stripped.endswith("...") or stripped.endswith("\u2014") or stripped.endswith("—"):
        return "trailing_interruption"
    if stripped.startswith("...") or stripped.startswith("\u2026"):
        return "leading_continuation"
    return None
