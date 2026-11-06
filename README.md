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
3. Generate multiple promo candidates.
4. Validate timestamps and duration.
5. Score and rank promos.
6. Export ranked results.

## Project Structure

- data/episodes: Input screenplays.
- src: Core Python modules.
- outputs: Generated promo results.
- app.py: Application entry point.
- config.py: Configuration.

## Technology

Python, Streamlit, Pandas, and interchangeable LLM providers.
