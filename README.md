# Kuku Promo Finder

An AI-assisted tool to discover and rank promotional clips from microdrama screenplays.

## Objective

Find strong 20–60 second promotional clips from 50 screenplay episodes.

Each promo should:
- Start with a strong hook.
- End with an unresolved cliffhanger.
- Make sense to a new viewer.
- Include accurate episode numbers and timestamps.

## Pipeline

1. Parse screenplay files.
2. Detect dramatic events.
3. Generate promo candidates.
4. Score candidates.
5. Rank candidates.
6. Validate cuts and group overlapping windows into promo families.

## Run the application

Install the Python dependencies, then start the Streamlit app from the project root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads the generated files in `outputs/`:

- `validated_promos.json`
- `validation_summary.json`
- `human_review_queue.json`
- `ranked_promos.json`, used only when the validated file cannot be read

The interface has five sections: Dashboard, Top Promos, Human Review, All Candidates, and Review Decisions. A reviewer can mark a candidate APPROVED, REJECTED, or HOLD and add an optional comment. Those choices are stored in `outputs/reviewer_decisions.json`. The app does not rewrite the ranking or validation files, and priority does not set a decision.

The source files contain screenplay text and timestamps. The app does not play video.

## Project Structure

- data/episodes: Input screenplays.
- src: Core Python modules.
- outputs: Generated promo results.
- app.py: Application entry point.
- config.py: Configuration.

## Technology

Python, Streamlit, and Pandas.
