import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis import utils

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Alternative CPA Pathways Survey_December 31, 2025_09.45.csv"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs")))
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
TEXT_DIR = OUTPUT_DIR / "text"
COLUMN_MAP_PATH = BASE_DIR / "analysis" / "column_map.py"


def build_column_map(df: pd.DataFrame) -> Dict[str, object]:
    columns = df.columns
    mapping: Dict[str, object] = {}

    mapping["Q29"] = utils.find_column(
        columns,
        ["Q29"],
        [r"q29", r"cpa.*(likely|intent|pursue)"],
    )
    mapping["Q31"] = utils.find_column(
        columns,
        ["Q31"],
        [r"q31", r"aware.*(alternative|pathway)", r"alternative.*pathway"],
    )
    mapping["Q51"] = utils.find_column(
        columns,
        ["Q51"],
        [r"q51", r"impact.*desire.*cpa"],
    )
    mapping["Q52"] = utils.find_column(
        columns,
        ["Q52"],
        [r"q52", r"impact.*(grad|graduate)"],
    )
    mapping["Q35"] = utils.find_column(
        columns,
        ["Q35"],
        [r"q35", r"how likely.*(grad|graduate).*(known|earlier)"],
    )
    mapping["Q36"] = utils.find_column(
        columns,
        ["Q36"],
        [r"q36", r"why.*(grad|graduate)"],
    )
    q8_map = utils.detect_q8_columns(columns)
    mapping.update({f"Q8_{i}": q8_map.get(f"Q8_{i}") for i in range(1, 8)})

    mapping["Q27"] = utils.find_column(
        columns,
        ["Q27"],
        [r"q27", r"undergrad", r"graduate student", r"program level"],
    )
    mapping["Q16"] = utils.find_column(
        columns,
        ["Q16"],
        [r"q16", r"full-time", r"part-time"],
    )
    mapping["Q46"] = utils.find_column(
        columns,
        ["Q46"],
        [r"q46", r"hours.*work"],
    )
    mapping["Q47"] = utils.find_column(
        columns,
        ["Q47"],
        [r"q47", r"cpa firm", r"public accounting", r"firm experience"],
    )
    mapping["Q60"] = utils.find_column(
        columns,
        ["Q60"],
        [r"q60", r"state"],
    )

    value_keywords = ["cpa"]
    value_suffixes = [
        "value",
        "worth",
        "roi",
        "benefit",
        "career",
        "salary",
        "opportunity",
        "credential",
    ]
    value_columns = []
    for col in columns:
        col_norm = utils.normalize_column_name(col)
        if "cpa" in col_norm and any(keyword in col_norm for keyword in value_suffixes):
            value_columns.append(col)
    mapping["cpa_value_columns"] = value_columns
    mapping["q8_columns"] = [q8_map.get(f"Q8_{i}") for i in range(1, 8) if q8_map.get(f"Q8_{i}")]

    return mapping


def select_work_experience_column(mapping: Dict[str, object], df: pd.DataFrame) -> Optional[str]:
    for key in ["Q47", "Q46"]:
        if mapping.get(key):
            return mapping[key]
    keyword_candidates = []
    for col in df.columns:
        col_norm = utils.normalize_column_name(col)
        if any(token in col_norm for token in ["intern", "job", "work", "firm", "experience"]):
            keyword_candidates.append(col)
    return keyword_candidates[0] if keyword_candidates else None


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=True)


def rq1_analysis(df: pd.DataFrame, mapping: Dict[str, object], notes: List[str]) -> None:
    q29 = mapping.get("Q29")
    q31 = mapping.get("Q31")
    if q29 and q31:
        table = utils.crosstab_with_row_pct(df, q31, q29)
        write_csv(table, TABLE_DIR / "rq1_awareness_by_intent.csv")
        counts = pd.crosstab(df[q31], df[q29], dropna=False)
        chi = utils.chi_square_test(counts)
        if chi:
            pd.DataFrame([chi]).to_csv(TABLE_DIR / "rq1_awareness_by_intent_chi2.csv", index=False)
        else:
            notes.append("RQ1: Chi-square not run for Q31 vs Q29 due to sparse expected counts.")
        pct = counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100
        utils.plot_stacked_bar(
            pct,
            "CPA Intent by Awareness of Alternative Pathway",
            q31,
            "Percent",
            FIG_DIR / "rq1_intent_by_awareness.png",
        )
    else:
        notes.append("RQ1: Missing Q29 or Q31; awareness vs intent table not generated.")

    q51 = mapping.get("Q51")
    if q29 and q51:
        table = utils.crosstab_with_row_pct(df, q51, q29)
        write_csv(table, TABLE_DIR / "rq1_impact_desire_cpa_by_intent.csv")
    elif q51 is None:
        notes.append("RQ1: Q51 (impact on desire to pursue CPA) not found; skipped comparison.")


def rq2_analysis(df: pd.DataFrame, mapping: Dict[str, object], notes: List[str]) -> None:
    q52 = mapping.get("Q52")
    q35 = mapping.get("Q35")
    q36 = mapping.get("Q36")

    if q52:
        dist = utils.safe_value_counts(df[q52])
        dist.to_csv(TABLE_DIR / "rq2_q52_distribution.csv")
        utils.plot_bar(
            dist,
            "Impact on Desire to Pursue Graduate Degree",
            q52,
            "Count",
            FIG_DIR / "rq2_q52_distribution.png",
        )
    else:
        notes.append("RQ2: Q52 (impact on desire for grad degree) not found.")

    if q52 and q35:
        table = utils.crosstab_with_row_pct(df, q52, q35)
        write_csv(table, TABLE_DIR / "rq2_q52_by_q35.csv")
    elif q35 is None:
        notes.append("RQ2: Q35 (likelihood of grad school if known earlier) not found.")

    if q36:
        theme_keywords = {
            "Cost/Time": ["cost", "tuition", "money", "time", "expense"],
            "Career Advancement": ["career", "advancement", "promotion", "opportunity"],
            "Flexibility": ["flexible", "online", "schedule", "part-time"],
            "Licensure Pathway": ["cpa", "license", "credential", "pathway"],
            "Interest/Need": ["interest", "need", "required", "value"],
            "Employer Support": ["employer", "firm", "support", "reimbursement"],
        }
        text_series = df[q36].dropna().astype(str)
        if not text_series.empty:
            keyword_df = utils.keyword_counts(text_series, theme_keywords)
            keyword_df.to_csv(TABLE_DIR / "q36_keywords.csv", index=False)
            quotes = utils.extract_theme_quotes(text_series, theme_keywords)
            theme_lines = ["# Q36 Themes", ""]
            for theme, quote_list in quotes.items():
                theme_lines.append(f"## {theme}")
                if quote_list:
                    for quote in quote_list:
                        theme_lines.append(f"- \"{quote}\"")
                else:
                    theme_lines.append("- (No representative quotes)")
                theme_lines.append("")
            (TEXT_DIR / "q36_themes.md").write_text("\n".join(theme_lines), encoding="utf-8")
        else:
            notes.append("RQ2: Q36 present but no text responses available.")
    else:
        notes.append("RQ2: Q36 open-ended question not found.")


def rq4_analysis(df: pd.DataFrame, mapping: Dict[str, object], notes: List[str]) -> None:
    q8_columns = mapping.get("q8_columns", [])
    if not q8_columns:
        notes.append("RQ4: No Q8 ranking/rating columns found.")
        return

    overall = utils.summarize_rankings(df, q8_columns)
    overall.to_csv(TABLE_DIR / "rq4_overall.csv", index=False)

    metric_col = "pct_top_3" if "pct_top_3" in overall.columns else "mean_rating"
    if metric_col in overall.columns:
        metric_series = overall.set_index("item")[metric_col]
        utils.plot_horizontal_bar(
            metric_series,
            "Curriculum Specialization Preferences",
            metric_col.replace("_", " ").title(),
            "Specialization",
            FIG_DIR / "rq4_overall.png",
        )

    subgroup_columns = {
        "Q27": mapping.get("Q27"),
        "Q16": mapping.get("Q16"),
        "Q46": mapping.get("Q46"),
        "Q47": mapping.get("Q47"),
        "Q60": mapping.get("Q60"),
    }

    subgroup_metric_ready = None
    for key, col in subgroup_columns.items():
        if col:
            sub_df = df.copy()
            if key == "Q46":
                sub_df[col] = utils.bin_hours(sub_df[col])
            if key == "Q60":
                sub_df[col] = utils.group_states(sub_df[col])
            summary = utils.subgroup_rankings(sub_df, q8_columns, col)
            if not summary.empty:
                summary.to_csv(TABLE_DIR / f"rq4_subgroup_{key.lower()}.csv", index=False)
                if subgroup_metric_ready is None:
                    subgroup_metric_ready = (summary, col)
        else:
            notes.append(f"RQ4: {key} not found; subgroup analysis skipped.")

    if subgroup_metric_ready:
        summary, label = subgroup_metric_ready
        if metric_col in summary.columns:
            pivot = summary.pivot(index="group", columns="item", values=metric_col)
            top_items = (
                overall.sort_values(metric_col, ascending=False)
                .head(3)["item"]
                .tolist()
            )
            plot_data = pivot[top_items]
            utils.plot_stacked_bar(
                plot_data,
                "Top Curriculum Preferences by Subgroup",
                label,
                metric_col.replace("_", " ").title(),
                FIG_DIR / "rq4_subgroup.png",
            )


def rq5_analysis(df: pd.DataFrame, mapping: Dict[str, object], notes: List[str]) -> None:
    work_col = select_work_experience_column(mapping, df)
    if not work_col:
        notes.append("RQ5: No work experience column found.")
        return

    value_columns = mapping.get("cpa_value_columns", [])
    proxy_used = False
    if not value_columns:
        proxy_used = True
        if mapping.get("Q29"):
            value_columns = [mapping["Q29"]]
            notes.append("RQ5: No CPA value perception items found; using Q29 as proxy.")
        else:
            notes.append("RQ5: No CPA value perception items found and Q29 missing; skipped.")
            return

    pd.DataFrame({"work_experience_column": [work_col]}).to_csv(
        TABLE_DIR / "rq5_work_experience_column_used.csv", index=False
    )
    pd.DataFrame({"value_column": value_columns}).to_csv(
        TABLE_DIR / "rq5_value_columns_used.csv", index=False
    )

    work_series = df[work_col]
    if work_col == mapping.get("Q46"):
        work_series = utils.bin_hours(work_series)

    for value_col in value_columns:
        series = df[value_col]
        if utils.detect_numeric_series(series):
            numeric = pd.to_numeric(series, errors="coerce")
            grouped = df.assign(work=work_series, value=numeric).dropna(subset=["work", "value"])
            if grouped.empty:
                continue
            stats_table = grouped.groupby("work")["value"].agg(["count", "median", "mean"]).reset_index()
            stats_table.to_csv(TABLE_DIR / f"rq5_numeric_{value_col}.csv", index=False)
            groups = [g["value"].values for _, g in grouped.groupby("work")]
            if len(groups) == 2:
                test = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                pd.DataFrame([
                    {"test": "mannwhitneyu", "statistic": test.statistic, "p_value": test.pvalue}
                ]).to_csv(TABLE_DIR / f"rq5_numeric_{value_col}_test.csv", index=False)
            elif len(groups) > 2:
                test = stats.kruskal(*groups)
                pd.DataFrame([
                    {"test": "kruskal", "statistic": test.statistic, "p_value": test.pvalue}
                ]).to_csv(TABLE_DIR / f"rq5_numeric_{value_col}_test.csv", index=False)
        else:
            table = utils.crosstab_with_row_pct(df.assign(work=work_series), "work", value_col)
            write_csv(table, TABLE_DIR / f"rq5_{value_col}_crosstab.csv")
            counts = pd.crosstab(work_series, series, dropna=False)
            chi = utils.chi_square_test(counts)
            if chi:
                pd.DataFrame([chi]).to_csv(
                    TABLE_DIR / f"rq5_{value_col}_chi2.csv", index=False
                )
            pct = counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100
            utils.plot_stacked_bar(
                pct,
                "CPA Value Perception by Work Experience",
                work_col,
                "Percent",
                FIG_DIR / "rq5_cpa_value_by_work.png",
            )
    if proxy_used:
        notes.append("RQ5: CPA value perception analysis uses Q29 as proxy.")


def generate_report(df: pd.DataFrame, mapping: Dict[str, object], notes: List[str]) -> None:
    lines = [
        "# Survey Analysis Report",
        "",
        "## Dataset Overview",
        f"- Rows: {df.shape[0]}",
        f"- Columns: {df.shape[1]}",
        "- Column mapping saved in outputs/tables/column_mapping_used.csv",
        "",
        "## RQ1: Awareness vs CPA Intent",
        "- Table: outputs/tables/rq1_awareness_by_intent.csv",
        "- Figure: outputs/figures/rq1_intent_by_awareness.png",
        "- Optional Table: outputs/tables/rq1_impact_desire_cpa_by_intent.csv",
        "",
        "## RQ2: Grad School Desire",
        "- Table: outputs/tables/rq2_q52_distribution.csv",
        "- Table: outputs/tables/rq2_q52_by_q35.csv",
        "- Figure: outputs/figures/rq2_q52_distribution.png",
        "- Q36 Themes: outputs/text/q36_themes.md",
        "",
        "## RQ4: Curriculum Preferences",
        "- Table: outputs/tables/rq4_overall.csv",
        "- Figure: outputs/figures/rq4_overall.png",
        "- Subgroup Tables: outputs/tables/rq4_subgroup_*.csv",
        "- Figure: outputs/figures/rq4_subgroup.png",
        "",
        "## RQ5: Work Experience and CPA Value",
        "- Tables: outputs/tables/rq5_*.csv",
        "- Figure: outputs/figures/rq5_cpa_value_by_work.png",
        "",
        "## Notes",
    ]
    if notes:
        lines.extend([f"- {note}" for note in notes])
    else:
        lines.append("- All requested analyses completed.")

    lines.extend(
        [
            "",
            "## Repro Steps",
            "```bash",
            "pip install -r requirements.txt",
            "python -m analysis.analysis_pipeline",
            "```",
        ]
    )
    (BASE_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    utils.ensure_directory(TABLE_DIR)
    utils.ensure_directory(FIG_DIR)
    utils.ensure_directory(TEXT_DIR)

    df = pd.read_csv(DATA_PATH)
    df = utils.trim_and_normalize_missing(df)

    mapping = build_column_map(df)
    utils.write_column_map(COLUMN_MAP_PATH, mapping)
    pd.DataFrame(
        {"expected": list(mapping.keys()), "actual": list(mapping.values())}
    ).to_csv(TABLE_DIR / "column_mapping_used.csv", index=False)

    notes: List[str] = []

    rq1_analysis(df, mapping, notes)
    rq2_analysis(df, mapping, notes)
    rq4_analysis(df, mapping, notes)
    rq5_analysis(df, mapping, notes)

    generate_report(df, mapping, notes)


if __name__ == "__main__":
    main()
