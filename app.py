"""Streamlit interface for ranked and validated promo candidates.

The screenplays supply text and timestamps. This application does not play video
and does not rewrite ranking or validation files.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

import config
from src.frontend_data import (
    cue_lines,
    dashboard_counts,
    episode_distribution,
    evidence_quotes,
    decision_widget_index,
    family_id,
    filter_candidates,
    group_review_queue,
    load_candidates,
    load_decisions,
    load_object,
    overall_score,
    priority_of,
    recommended_promos,
    save_decision,
    save_is_enabled,
    score_distribution,
)


def main() -> None:
    st.set_page_config(page_title=config.APP_TITLE, layout="wide")
    _style()
    st.sidebar.title(config.APP_TITLE)
    st.sidebar.caption("Text and timestamps only. No video.")
    page = st.sidebar.radio("Section", config.PAGES, label_visibility="collapsed")

    candidates, candidate_error = _cached_candidates(str(config.VALIDATED_PATH), str(config.RANKED_PATH))
    summary, summary_error = _cached_object(str(config.SUMMARY_PATH))
    queue, queue_error = _cached_object(str(config.QUEUE_PATH))
    decisions, decision_error = load_decisions(config.DECISIONS_PATH)
    records = {row.get("candidate_id"): row for row in candidates if row.get("candidate_id")}

    if page == "Dashboard":
        _dashboard(candidates, summary, queue, candidate_error, summary_error, queue_error)
    elif page == "Top Promos":
        _top_promos(candidates, decisions["decisions"], candidate_error)
    elif page == "Human Review":
        _human_review(queue, records, decisions["decisions"], queue_error)
    elif page == "All Candidates":
        _all_candidates(candidates, decisions["decisions"], candidate_error)
    else:
        _review_decisions(candidates, decisions, decision_error)


@st.cache_data(show_spinner="Loading promo records...")
def _cached_candidates(validated_path: str, ranked_path: str) -> tuple[list[dict], str | None]:
    rows, error = load_candidates(validated_path)
    if not error:
        return rows, None
    fallback, ranked_error = load_candidates(ranked_path)
    if not ranked_error:
        return fallback, f"{error} Showing ranked records without validation fields."
    return [], error


@st.cache_data(show_spinner=False)
def _cached_object(path: str) -> tuple[dict, str | None]:
    return load_object(path)


def _dashboard(candidates, summary, queue, candidate_error, summary_error, queue_error) -> None:
    st.header("Dashboard")
    _show_errors(candidate_error, summary_error, queue_error)
    counts = dashboard_counts(candidates, queue)
    if summary and summary.get("candidate_count") not in (None, counts["total"]):
        st.caption("Summary file candidate count differs from the loaded candidate records. Cards use the loaded records.")
    first, second, third = st.columns(3)
    first.metric("Candidates", counts["total"])
    second.metric("High priority", counts["high"])
    third.metric("Medium priority", counts["medium"])
    fourth, fifth, sixth = st.columns(3)
    fourth.metric("Low priority", counts["low"])
    fifth.metric("Promo families", counts["families"])
    sixth.metric("Inspection queue", counts["inspection_queue"])

    score_col, episode_col = st.columns(2)
    with score_col:
        st.subheader("Overall score")
        st.bar_chart(pd.DataFrame(score_distribution(candidates)), x="band", y="candidates", color="#3E5C4A")
    with episode_col:
        st.subheader("Candidates by episode")
        st.bar_chart(pd.DataFrame(episode_distribution(candidates)), x="episode", y="candidates", color="#8A6A2F")
    st.caption("Priority is a review hint. It does not approve or reject a promo.")


def _top_promos(candidates, decisions, error) -> None:
    st.header("Top Promos")
    _show_errors(error)
    ranked = recommended_promos(candidates)
    if not ranked:
        st.info("No ranked recommendations are in the loaded file.")
        return
    table = pd.DataFrame(
        [
            {
                "Rank": row.get("shortlist_rank"),
                "Candidate": row.get("candidate_id"),
                "Episode": row.get("episode_number"),
                "Start": row.get("start"),
                "End": row.get("end"),
                "Seconds": row.get("duration_seconds"),
                "Overall": overall_score(row),
                "Priority": priority_of(row),
            }
            for row in ranked
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    labels = [f"{row.get('shortlist_rank')}. {row.get('candidate_id')}" for row in ranked]
    choice = st.selectbox("Open a recommendation", labels)
    selected = ranked[labels.index(choice)]
    _candidate_detail(selected, decisions.get(selected.get("candidate_id")))


def _human_review(queue, records, decisions, error) -> None:
    st.header("Human Review")
    st.caption("Representatives and alternative cuts from the inspection queue. Same final line is not repeated.")
    _show_errors(error)
    families = group_review_queue(queue)
    if not families:
        st.info("The inspection queue is empty or could not be read.")
        return
    st.metric("Queue items", sum(len(family["items"]) for family in families))
    labels = [
        f"{family['family_id']} · {len(family['items'])} cut{'s' if len(family['items']) != 1 else ''}"
        for family in families
    ]
    choice = st.selectbox("Promo family", labels)
    family = families[labels.index(choice)]
    st.write(f"Family seed: `{family['seed_candidate_id']}`")
    columns = st.columns(len(family["items"]))
    for column, item in zip(columns, family["items"]):
        with column:
            _queue_card(item, records.get(item.get("candidate_id")), decisions.get(item.get("candidate_id")))


def _queue_card(item: dict, candidate: dict | None, decision: dict | None) -> None:
    role = item.get("role") or "cut"
    st.markdown(f"**{role.title()}**")
    st.write(item.get("candidate_id"))
    scores = item.get("scores") if isinstance(item.get("scores"), dict) else {}
    st.write(f"{item.get('start')} – {item.get('end')}")
    st.write(f"Overall {scores.get('overall')}")
    for reason in item.get("reasons") or []:
        if not isinstance(reason, dict):
            continue
        status = reason.get("status") or "heuristic"
        st.markdown(f"<span class='pill pill-{status}'>{status}</span> {reason.get('review_reason')}", unsafe_allow_html=True)
    if item.get("same_ending_candidate_ids"):
        st.caption("Same ending also appears on " + ", ".join(item["same_ending_candidate_ids"]))
    if decision:
        st.caption(f"Reviewer decision: {decision['decision']}")
    if candidate is None:
        st.warning("This queue item has no matching candidate record.")
        return
    with st.expander("Dialogue and action"):
        _lines("Dialogue", cue_lines(candidate.get("dialogue")))
        _lines("Action", cue_lines(candidate.get("action")))


def _all_candidates(candidates, decisions, error) -> None:
    st.header("All Candidates")
    _show_errors(error)
    if not candidates:
        st.info("No candidates are loaded.")
        return
    episodes = sorted({row.get("episode_number") for row in candidates if isinstance(row.get("episode_number"), int)})
    priorities = [value for value in config.PRIORITY_ORDER if any(priority_of(row) == value for row in candidates)]
    families = sorted({family_id(row) for row in candidates if family_id(row)})
    statuses = sorted({row.get("shortlist_status") for row in candidates if isinstance(row.get("shortlist_status"), str)})
    scores = [overall_score(row) for row in candidates if overall_score(row) is not None]
    low = min(scores) if scores else 0.0
    high = max(scores) if scores else 100.0

    query = st.text_input("Candidate ID")
    filter_col = st.columns(4)
    episode = filter_col[0].selectbox("Episode", ["All"] + episodes)
    priority = filter_col[1].selectbox("Priority", ["All"] + priorities)
    family = filter_col[2].selectbox("Family", ["All"] + families)
    status = filter_col[3].selectbox("Ranking status", ["All"] + statuses)
    score_range = st.slider("Overall score", float(low), float(high), (float(low), float(high)))
    matched = filter_candidates(
        candidates,
        text=query,
        episode=None if episode == "All" else episode,
        priority=None if priority == "All" else priority,
        min_score=score_range[0],
        max_score=score_range[1],
        family=None if family == "All" else family,
        ranking_status=None if status == "All" else status,
    )
    st.caption(f"{len(matched)} of {len(candidates)} candidates match.")
    st.dataframe(pd.DataFrame([_table_row(row, decisions) for row in matched]), hide_index=True, width="stretch")
    if not matched:
        return
    labels = [row.get("candidate_id") for row in matched]
    choice = st.selectbox("Open a candidate", labels)
    selected = matched[labels.index(choice)]
    _candidate_detail(selected, decisions.get(selected.get("candidate_id")))


def _review_decisions(candidates, decisions, decision_error) -> None:
    st.header("Review Decisions")
    st.caption("Choices are stored in outputs/reviewer_decisions.json. Ranking and validation files are not changed.")
    if decision_error:
        st.error(decision_error)
        st.info("Save is disabled until the decision file can be read.")
        return
    if not candidates:
        st.info("Load candidate records before recording a decision.")
        return
    labels = [row.get("candidate_id") for row in candidates if row.get("candidate_id")]
    candidate_id = st.selectbox("Candidate", labels)
    existing = decisions["decisions"].get(candidate_id)
    decision = st.selectbox(
        "Decision",
        config.DECISION_CHOICES,
        index=decision_widget_index(existing),
        placeholder="Select a decision",
        key=f"decision-{candidate_id}",
    )
    comment = st.text_area(
        "Comment",
        value=existing["comment"] if existing else "",
        key=f"comment-{candidate_id}",
    )
    submitted = st.button("Save decision", disabled=not save_is_enabled(decision))
    if submitted:
        if decision not in config.DECISION_CHOICES:
            st.error("Choose APPROVED, REJECTED, or HOLD.")
        else:
            try:
                save_decision(
                    config.DECISIONS_PATH,
                    candidate_id,
                    decision,
                    comment,
                    protected_paths=config.GENERATED_PATHS,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success(f"Saved {decision} for {candidate_id}.")
                st.rerun()
    saved = list(decisions["decisions"].values())
    if saved:
        st.subheader("Saved decisions")
        st.dataframe(pd.DataFrame(saved), hide_index=True, width="stretch")
    else:
        st.info("No reviewer decisions have been saved yet.")


def _candidate_detail(candidate: dict, decision: dict | None) -> None:
    scores = candidate.get("scores") if isinstance(candidate.get("scores"), dict) else {}
    st.subheader(candidate.get("candidate_id"))
    st.markdown(_pill(priority_of(candidate)), unsafe_allow_html=True)
    meta = st.columns(4)
    meta[0].write(f"Episode {candidate.get('episode_number')}")
    meta[1].write(f"{candidate.get('start')} – {candidate.get('end')}")
    meta[2].write(f"{candidate.get('duration_seconds')} seconds")
    meta[3].write(f"Family {family_id(candidate) or '—'}")
    score_cols = st.columns(5)
    for column, key, label in zip(
        score_cols,
        ("overall", "hook", "cliffhanger", "standalone_clarity", "emotional_intensity"),
        ("Overall", "Hook", "Cliffhanger", "Clarity", "Intensity"),
    ):
        column.metric(label, scores.get(key, "—"))
    if candidate.get("ranking_explanation"):
        st.write(candidate["ranking_explanation"])
    if decision:
        st.caption(f"Reviewer decision: {decision['decision']}. {decision.get('comment') or ''}")

    _quote_block("Hook evidence", evidence_quotes(candidate, "hook"))
    _quote_block("Cliffhanger evidence", evidence_quotes(candidate, "cliffhanger"))
    _lines("Dialogue", cue_lines(candidate.get("dialogue")))
    _lines("Action", cue_lines(candidate.get("action")))
    _warnings(candidate)


def _quote_block(title: str, quotes: list[dict]) -> None:
    st.markdown(f"**{title}**")
    if not quotes:
        st.caption("No evidence quotes are stored for this dimension.")
        return
    for quote in quotes:
        timing = " ".join(part for part in (quote.get("start"), quote.get("end")) if part)
        note = quote.get("note") or ""
        st.write(f"{quote.get('text') or ''}")
        if timing or note:
            st.caption(" · ".join(part for part in (timing, note) if part))


def _lines(title: str, lines: list[dict]) -> None:
    st.markdown(f"**{title}**")
    if not lines:
        st.caption(f"No {title.lower()} is stored in this window.")
        return
    st.dataframe(pd.DataFrame(lines), hide_index=True, width="stretch")


def _warnings(candidate: dict) -> None:
    st.markdown("**Validation**")
    reasons = (candidate.get("validation") or {}).get("priority_reasons")
    if not isinstance(reasons, list) or not reasons:
        st.caption("No validation reasons are stored.")
        return
    for reason in reasons:
        if not isinstance(reason, dict):
            continue
        level = reason.get("level") or "none"
        status = reason.get("status") or "informational"
        st.markdown(
            f"<span class='pill pill-{level}'>{level}</span> "
            f"<span class='pill pill-{status}'>{status}</span> {reason.get('review_reason')}",
            unsafe_allow_html=True,
        )
    following = ((candidate.get("validation") or {}).get("ending") or {}).get("following_line") or {}
    if isinstance(following, dict) and following.get("present"):
        st.caption(f"Next screenplay line after the cut: {following.get('text')}")


def _table_row(candidate: dict, decisions: dict) -> dict:
    decision = decisions.get(candidate.get("candidate_id")) or {}
    return {
        "Candidate": candidate.get("candidate_id"),
        "Episode": candidate.get("episode_number"),
        "Start": candidate.get("start"),
        "End": candidate.get("end"),
        "Seconds": candidate.get("duration_seconds"),
        "Overall": overall_score(candidate),
        "Priority": priority_of(candidate),
        "Family": family_id(candidate),
        "Ranking": candidate.get("shortlist_status"),
        "Decision": decision.get("decision") or "",
    }


def _pill(priority: str) -> str:
    return f"<span class='pill pill-{priority}'>{priority}</span>"


def _show_errors(*errors: str | None) -> None:
    for error in errors:
        if error:
            st.warning(error)


def _style() -> None:
    st.markdown(
        """
        <style>
        .pill { display:inline-block; margin:0 0.3rem 0.3rem 0; padding:0.1rem 0.5rem;
                 border-radius:999px; font-size:0.78rem; letter-spacing:0.02em; }
        .pill-high { background:#F4E4E6; color:#8C2F39; }
        .pill-medium { background:#F6F0E4; color:#8A6A2F; }
        .pill-low, .pill-informational { background:#E7EFE9; color:#3E5C4A; }
        .pill-none { background:#EEF0F2; color:#5C6570; }
        .pill-confirmed { background:#E7EFE9; color:#245C43; }
        .pill-heuristic { background:#F6F0E4; color:#8A6A2F; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
