# Survey Analysis Report

## Dataset Overview
- This report is generated from the survey CSV in the repo root.
- Run the analysis pipeline to produce all tables, figures, and text outputs:
  - `pip install -r requirements.txt`
  - `python -m analysis.analysis_pipeline`

## RQ1: Awareness vs CPA Intent
- Table: `outputs/tables/rq1_awareness_by_intent.csv`
- Figure: `outputs/figures/rq1_intent_by_awareness.png`
- Optional Table: `outputs/tables/rq1_impact_desire_cpa_by_intent.csv`

## RQ2: Grad School Desire
- Table: `outputs/tables/rq2_q52_distribution.csv`
- Table: `outputs/tables/rq2_q52_by_q35.csv`
- Figure: `outputs/figures/rq2_q52_distribution.png`
- Q36 Themes: `outputs/text/q36_themes.md`

## RQ4: Curriculum Preferences
- Table: `outputs/tables/rq4_overall.csv`
- Figure: `outputs/figures/rq4_overall.png`
- Subgroup Tables: `outputs/tables/rq4_subgroup_*.csv`
- Figure: `outputs/figures/rq4_subgroup.png`

## RQ5: Work Experience and CPA Value
- Tables: `outputs/tables/rq5_*.csv`
- Figure: `outputs/figures/rq5_cpa_value_by_work.png`

## Repro Steps
```bash
pip install -r requirements.txt
python -m analysis.analysis_pipeline
```
