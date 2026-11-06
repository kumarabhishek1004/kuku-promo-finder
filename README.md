# Kuku Promo Finder

An AI-assisted tool to discover, evaluate, and rank promotional clips from microdrama screenplays.

## Objective

Find strong 20–60 second promotional clips from screenplay episodes.

Each promo should:

- Start with a strong hook within the first 3 seconds.
- End with an unresolved cliffhanger.
- Make sense to a new viewer who has not watched the show.
- Include accurate episode numbers and start/end timestamps.
- Provide evidence explaining why the promo works.

The project processes the supplied 50-episode English-language screenplay pack and generates ranked promo candidates with validation and human-review guidance.

## Pipeline

The solution follows a modular screenplay-to-promo pipeline:

1. **Parse:** Read screenplay files and extract episode metadata, scenes, dialogue, timestamps, camera directions, actions, and sound cues.
2. **Detect events:** Identify dramatic events and relevant screenplay cues.
3. **Generate candidates:** Create potential 20–60 second promo windows from the detected events.
4. **Score candidates:** Evaluate hook strength, cliffhanger potential, standalone clarity, and emotional intensity.
5. **Rank promos:** Rank candidates and reduce repetitive recommendations.
6. **Validate:** Check candidate quality signals and group overlapping cuts into promo families.
7. **Human review:** Allow reviewers to inspect candidates and record editorial decisions.

A high score or validation priority is a recommendation for review, not automatic approval.

## Technology Stack

- Python 3.12
- Streamlit
- Pandas
- Modular Python-based processing pipeline

The current pipeline uses deterministic code and rule-based scoring. It does not require a paid LLM API.

## Project Structure

```text
kuku-promo-finder/
│
├── data/
│   └── episodes/              # Input screenplay files
│
├── src/
│   ├── parser.py              # Screenplay parsing
│   ├── event_detector.py      # Dramatic event detection
│   ├── promo_generator.py     # Promo candidate generation
│   ├── promo_scorer.py        # Candidate scoring
│   ├── ranking.py             # Ranking and duplicate handling
│   ├── validator.py           # Validation and promo family grouping
│   ├── frontend_data.py       # Dashboard data preparation
│   └── llm_client.py          # Model integration interface
│
├── outputs/
│   ├── parsed_episodes.json
│   ├── promo_candidates.json
│   ├── scored_candidates.json
│   ├── ranked_promos.json
│   ├── ranked_promos.csv
│   ├── validated_promos.json
│   ├── validation_summary.json
│   ├── human_review_queue.json
│   └── reviewer_decisions.json
│
├── tests/                     # Automated tests
│
├── app.py                     # Streamlit application entry point
├── config.py                  # Configuration
├── requirements.txt           # Python dependencies
└── README.md
```

## Setup and Installation

### Prerequisites

- Python 3.12
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/kumarabhishek1004/kuku-promo-finder.git
cd kuku-promo-finder
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

The Streamlit dashboard reads the generated JSON files from the `outputs/` directory.

Make sure the required output files are present before starting the application.

Run:

```bash
streamlit run app.py
```

The dashboard will be available locally, usually at:

```text
http://localhost:8501
```

## Dashboard Features

The application provides five main sections:

### 1. Dashboard
Provides an overview of generated candidates and validation results.

### 2. Top Promos
Displays the highest-ranked promotional candidates.

### 3. Human Review
Displays candidates requiring editorial inspection.

### 4. All Candidates
Allows reviewers to explore the generated promo candidates.

### 5. Review Decisions
Allows reviewers to record decisions:

- `APPROVED`
- `REJECTED`
- `HOLD`

Reviewers can also add optional comments.

Human decisions are stored separately in:

```text
outputs/reviewer_decisions.json
```

The application does not overwrite the original ranking or validation results when reviewers make decisions.

## Generated Outputs

The pipeline produces the following files:

| File | Description |
|---|---|
| `parsed_episodes.json` | Structured screenplay parsing results |
| `promo_candidates.json` | Generated promo candidates |
| `scored_candidates.json` | Candidates with evaluation scores |
| `ranked_promos.json` | Ranked promo results |
| `ranked_promos.csv` | Tabular ranked results |
| `validated_promos.json` | Candidates with validation metadata |
| `validation_summary.json` | Validation statistics and summary |
| `human_review_queue.json` | Candidates prioritized for human inspection |
| `reviewer_decisions.json` | Human editorial decisions, created through the dashboard |

## Reproducibility and Pipeline Execution

The project is organized into independent processing modules so that each stage can be inspected, tested, and improved separately.

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

To run the pipeline again on a new screenplay pack, provide the input files in the expected screenplay format and execute the processing stages using their configured input and output paths.

The parser may need adaptation if a new screenplay source uses a different format.

The dashboard can be run independently once the generated output files are available.

## Testing

The project includes automated tests covering:

- Screenplay parsing
- Promo candidate generation
- Candidate scoring
- Ranking and duplicate handling
- Validation
- Frontend data preparation

The reported test run completed with **61 passing tests**.

## Quality Checks and Limitations

The top-ranked candidates were manually reviewed against their screenplay dialogue and surrounding context.

The evaluation identified strong candidates as well as cases where the automated system selected weak or incomplete cuts.

The current system has the following limitations:

- Scoring and validation signals are heuristic and cannot fully replace editorial judgment.
- Standalone comprehension and emotional impact require human review.
- The current prototype focuses on continuous segments within individual episodes.
- It does not render or edit video.
- It does not add music, subtitles, or text overlays.
- It does not automatically assemble multi-segment or cross-episode promos.
- The human-review queue is a prioritization aid, not automatic approval.

## AI Tools Used

- **ChatGPT:** Problem framing, workflow planning, debugging, and review.
- **Cursor Agent:** Code implementation, testing support, and Streamlit dashboard development.

The current processing pipeline is Python-based and does not require a paid model API.

## Scope

This project identifies and ranks potential promotional moments, provides timestamps and supporting evidence, and supports human editorial review.

It focuses on discovering the right moments and cut points. Video editing and final ad production are outside the scope of this prototype.
