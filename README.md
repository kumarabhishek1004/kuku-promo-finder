# Kuku Promo Finder

An AI-assisted prototype for discovering, evaluating, and ranking promotional clips from microdrama screenplays.

**Assignment:** Kuku AI Generalist — Problem 1: Find the Promos (Ads)

## 1. Objective

The goal is to identify strong 20–60 second promotional moments from microdrama screenplays.

A good promo should:

- **Open with a hook:** Capture attention within the first 3 seconds through a reveal, shocking line, confrontation, or dramatic action.
- **End with a cliffhanger:** Cut at an unresolved moment that makes viewers want to know what happens next.
- **Stand alone:** Provide enough context for someone who has never watched the show.
- **Provide accurate cut points:** Include the episode number and start/end timestamps for each selected segment.
- **Explain its potential:** Provide evidence and reasoning for why the candidate could work as a promotional clip.

The prototype was evaluated using the supplied 50-episode English-language screenplay pack.

Its purpose is to identify potential promotional moments and cut points, not to edit or render videos.

## 2. Solution Overview

The solution uses a modular screenplay-to-promo pipeline.

### Processing stages

1. **Screenplay Parsing:** Read screenplay files and extract episode metadata, scenes, dialogue, timestamps, camera directions, actions, and sound cues.
2. **Event Detection:** Identify dramatic events and relevant screenplay cues.
3. **Candidate Generation:** Generate potential 20–60 second promotional windows using detected events.
4. **Candidate Scoring:** Evaluate hook strength, cliffhanger potential, standalone clarity, and emotional intensity.
5. **Ranking:** Rank candidates and reduce repetitive recommendations.
6. **Validation:** Check candidate quality signals and group overlapping cuts into promo families.
7. **Human Review:** Present candidates for editorial inspection and allow reviewers to record decisions.

### Scoring approach

Candidates are evaluated using four dimensions:

- Hook strength
- Cliffhanger potential
- Standalone clarity
- Emotional intensity

The weighted scoring formula is:

`Total Score = (Hook × 3.5) + (Cliffhanger × 3) + (Clarity × 2) + (Intensity × 1.5)`

Each dimension is scored from 0 to 10, producing a total score from 0 to 100.

A high score or validation priority is a recommendation for human inspection, not automatic approval.

## 3. Technology Stack

- Python 3.12
- Streamlit
- Pandas
- Modular Python processing pipeline
- Automated tests using pytest

The current processing pipeline uses deterministic code and rule-based evaluation. It does not require a paid LLM API.

## 4. Project Structure

The public repository contains source code, tests, configuration, and documentation.

```text
kuku-promo-finder/
│
├── src/
│   ├── parser.py
│   ├── event_detector.py
│   ├── promo_generator.py
│   ├── promo_scorer.py
│   ├── ranking.py
│   ├── validator.py
│   ├── frontend_data.py
│   └── llm_client.py
│
├── tests/
│
├── app.py
├── config.py
├── requirements.txt
├── .gitignore
└── README.md
```

**Confidentiality:** The original screenplay input pack and generated screenplay-derived outputs are intentionally excluded from this public repository. The raw outputs are supplied separately through the assignment submission.

## 5. Setup and Installation

### Prerequisites

- Python 3.12
- pip
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/kumarabhishek1004/kuku-promo-finder.git
cd kuku-promo-finder
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

## 6. Running the Application

The Streamlit dashboard reads generated JSON files from an `outputs/` directory in the project root.

The original input pack and generated outputs are not included in the public repository because the screenplay material is confidential. The generated output files are provided separately as part of the assignment submission.

To run the dashboard:

1. Obtain the generated output files from the assignment submission.
2. Create an `outputs/` directory in the project root.
3. Place the required JSON files in that directory.
4. Start the application.

Run:

```bash
streamlit run app.py
```

The dashboard will normally be available at:

```text
http://localhost:8501
```

The dashboard uses these files:

- `validated_promos.json`
- `validation_summary.json`
- `human_review_queue.json`
- `ranked_promos.json` (fallback when the validated file cannot be read)

Reviewer decisions are stored separately in:

```text
outputs/reviewer_decisions.json
```

## 7. Dashboard Features

The application provides five main sections.

### 1. Dashboard

Provides an overview of generated candidates and validation results.

### 2. Top Promos

Displays the highest-ranked promotional candidates.

### 3. Human Review

Displays candidates prioritized for editorial inspection.

### 4. All Candidates

Allows reviewers to explore the generated promo candidates.

### 5. Review Decisions

Allows reviewers to record one of three decisions:

- `APPROVED`
- `REJECTED`
- `HOLD`

Reviewers can also add optional comments.

Human decisions are stored separately from the original ranking and validation results. The application does not automatically approve or reject candidates based on their scores or validation priorities.

## 8. Generated Outputs

The pipeline produces the following files:

| File | Description |
|---|---|
| `parsed_episodes.json` | Structured screenplay parsing results |
| `promo_candidates.json` | Generated promotional candidates |
| `scored_candidates.json` | Candidates with evaluation scores |
| `ranked_promos.json` | Ranked promo results |
| `ranked_promos.csv` | Tabular ranked results |
| `validated_promos.json` | Candidates with validation metadata |
| `validation_summary.json` | Validation statistics and summary |
| `human_review_queue.json` | Candidates prioritized for human inspection |
| `reviewer_decisions.json` | Human editorial decisions recorded through the dashboard |

### Results from the supplied input pack

The measured run on the 50-episode input pack produced:

| Measure | Result |
|---|---:|
| Episodes processed | 50 |
| Scenes | 100 |
| Timed cues parsed | 3,727 |
| Dialogue cues | 1,487 |
| Promo candidates generated | 1,117 |
| Promo families | 243 |
| High-priority inspection items | 52 |
| Automated tests passed | 61 |

Validation priorities:

- HIGH: 75
- MEDIUM: 178
- LOW: 864

These priorities are review signals. They do not automatically determine whether a promo should be published.

## 9. Reproducibility and Architecture

The project is organized into independent processing modules so that parsing, event detection, candidate generation, scoring, ranking, validation, and frontend data preparation can be inspected and improved separately.

The processing flow is:

```text
Screenplay Inputs
       |
       v
     Parser
       |
       v
 Event Detection
       |
       v
 Promo Generation
       |
       v
 Candidate Scoring
       |
       v
     Ranking
       |
       v
   Validation
       |
       v
 Generated Outputs
       |
       v
 Streamlit Dashboard
```

The pipeline uses screenplay timestamps as the source of truth for candidate start and end times. Timestamps are converted into integer milliseconds during processing.

The current implementation focuses on single continuous segments within individual episodes.

For a new screenplay pack, the input files must follow the expected screenplay format. A different source format may require changes to the parser.

The modular design allows individual processing stages to be inspected, tested, and improved without redesigning the entire application.

## 10. Testing

The project includes automated tests covering:

- Screenplay parsing
- Promo candidate generation
- Candidate scoring
- Ranking and duplicate handling
- Validation
- Frontend data preparation

The reported test run completed with **61 passing tests**.

To run the test suite, install pytest if it is not already available:

```bash
python -m pip install pytest
```

Then run from the project root:

```bash
python -m pytest
```

## 11. Quality Evaluation

The top 10 ranked candidates were manually checked against their screenplay dialogue and surrounding context.

The qualitative review found:

- 4 excellent candidates
- 2 good candidates
- 4 average or poor candidates as finished promotional clips

Strong examples included:

- Episode 33 interrogation — `ep33-c0020`
- Episode 1 altar rejection — `ep01-c0027`
- Episode 42 return reveal — `ep42-c0010`
- Episode 28 poison accusation — `ep28-c0031`

This was a qualitative review of the top 10 candidates, not a measured precision or recall evaluation across all 1,117 candidates.

## 12. Validation Improvements and Known Failures

The initial validator produced some incorrect warnings:

- It treated every ending ellipsis as a possible incomplete sentence, incorrectly flagging dramatic pauses such as character reactions.
- A pronoun-count heuristic flagged some coherent dialogue that had sufficient context.
- An intersection-over-union-only duplicate threshold missed two overlapping Episode 28 cuts.

The validation logic was revised to distinguish sentence continuation from dramatic pauses and to group overlapping variants using both intersection-over-union and shorter-clip coverage.

All candidates were retained. Validation adds review metadata rather than automatically deleting candidates.

## 13. Limitations and Future Improvements

The current prototype has several limitations:

- Scoring and validation signals are heuristic and cannot fully replace editorial judgment.
- Standalone comprehension and emotional impact require human review.
- The current implementation focuses on continuous segments within individual episodes.
- Multi-segment and cross-episode promo assembly is not implemented.
- The prototype does not render or edit video.
- It does not add music, subtitles, or text overlays.
- The inspection queue is a prioritization aid, not an approval of all remaining candidates.
- A new screenplay format may require parser adaptation.

Potential improvements include:

- Collecting feedback from human editors.
- Evaluating a larger sample of candidates.
- Adding richer semantic checks for standalone comprehension and emotional impact.
- Supporting multi-segment and cross-episode promo generation.
- Introducing optional model-assisted evaluation with a provider-swappable interface.

## 14. AI Tools Used

The following tools supported development:

- **ChatGPT:** Problem framing, workflow planning, debugging, and review.
- **Cursor Agent:** Code inspection, implementation, testing support, and Streamlit dashboard development.

The production pipeline is Python-based and does not require a paid LLM API.

### Example of an AI-assisted mistake and correction

During validation development, an initial heuristic treated every ending ellipsis as a possible sentence cut. This incorrectly flagged dramatic pauses such as character reactions.

A pronoun-count heuristic also produced unnecessary warnings for some coherent clips.

These checks were revised to distinguish sentence continuation from dramatic pauses and to treat pronouns as informational rather than automatic evidence of poor context.

This illustrates why automated results are retained for human review instead of being treated as final editorial decisions.

## 15. Scope

Kuku Promo Finder identifies and ranks potential promotional moments, provides episode timestamps and supporting evidence, groups overlapping variants, and supports human editorial decisions.

It focuses on discovering the right moments and cut points. Video editing and final advertisement production are outside the scope of this prototype.
