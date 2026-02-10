import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

MISSING_VALUES = {"", " ", "na", "n/a", "none", "null", "nan", "-"}


def trim_and_normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].astype(str).str.strip()
            cleaned[col] = cleaned[col].replace(
                {value: np.nan for value in MISSING_VALUES}
            )
            cleaned[col] = cleaned[col].replace({"nan": np.nan})
    return cleaned


def normalize_column_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def find_column(
    columns: Iterable[str],
    exact_keys: Iterable[str],
    patterns: Iterable[str],
) -> Optional[str]:
    column_map = {normalize_column_name(col): col for col in columns}
    for key in exact_keys:
        match = column_map.get(normalize_column_name(key))
        if match:
            return match
    for col in columns:
        col_norm = normalize_column_name(col)
        for pattern in patterns:
            if re.search(pattern, col_norm):
                return col
    return None


def detect_q8_columns(columns: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    regexes = [
        re.compile(r"q8[\s_\-]*([1-7])"),
        re.compile(r"rank[\s_\-]*([1-7])"),
    ]
    for col in columns:
        col_norm = normalize_column_name(col)
        for regex in regexes:
            match = regex.search(col_norm)
            if match:
                idx = match.group(1)
                mapping[f"Q8_{idx}"] = col
                break
    return mapping


def detect_numeric_series(series: pd.Series) -> bool:
    if series.dropna().empty:
        return False
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.notna().sum() >= max(3, int(0.5 * series.dropna().shape[0]))


def detect_rank_or_rating(series: pd.Series) -> Tuple[str, Optional[pd.Series]]:
    if detect_numeric_series(series):
        numeric = pd.to_numeric(series, errors="coerce")
        unique_vals = sorted(numeric.dropna().unique())
        if unique_vals and max(unique_vals) <= 7 and min(unique_vals) >= 1:
            return "rank", numeric
        return "rating", numeric
    return "categorical", None


def crosstab_with_row_pct(
    df: pd.DataFrame, row_col: str, col_col: str
) -> pd.DataFrame:
    counts = pd.crosstab(df[row_col], df[col_col], dropna=False)
    row_pct = counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100
    combined = pd.concat(
        {"count": counts, "row_pct": row_pct.round(1)}, axis=1
    )
    return combined


def chi_square_test(table: pd.DataFrame) -> Optional[Dict[str, float]]:
    try:
        chi2, p, dof, expected = stats.chi2_contingency(table)
    except ValueError:
        return None
    if expected.min() < 5:
        return None
    n = table.values.sum()
    phi2 = chi2 / n
    r, k = table.shape
    cramers_v = np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else np.nan
    return {"chi2": chi2, "p_value": p, "dof": dof, "cramers_v": cramers_v}


def save_plot(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.clf()


def plot_stacked_bar(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    ax = df.plot(kind="bar", stacked=True, figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Response", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_plot(ax.get_figure(), path)


def plot_bar(
    series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path
) -> None:
    import matplotlib.pyplot as plt

    ax = series.plot(kind="bar", figsize=(8, 4))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_plot(ax.get_figure(), path)


def plot_horizontal_bar(
    series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path
) -> None:
    import matplotlib.pyplot as plt

    ax = series.sort_values().plot(kind="barh", figsize=(8, 4))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_plot(ax.get_figure(), path)


def write_column_map(path: Path, mapping: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("column_map = ")
        json.dump(mapping, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def keyword_counts(text_series: pd.Series, keywords: Dict[str, List[str]]) -> pd.DataFrame:
    counts = []
    for theme, words in keywords.items():
        pattern = re.compile(r"(" + "|".join(re.escape(w) for w in words) + r")", re.I)
        theme_count = text_series.dropna().str.contains(pattern).sum()
        counts.append({"theme": theme, "count": int(theme_count)})
    return pd.DataFrame(counts).sort_values("count", ascending=False)


def extract_theme_quotes(
    text_series: pd.Series, keywords: Dict[str, List[str]], max_quotes: int = 3
) -> Dict[str, List[str]]:
    quotes: Dict[str, List[str]] = {}
    for theme, words in keywords.items():
        pattern = re.compile(r"(" + "|".join(re.escape(w) for w in words) + r")", re.I)
        matched = text_series.dropna()[text_series.dropna().str.contains(pattern)]
        selected = (
            matched.astype(str)
            .str.strip()
            .str.slice(0, 140)
            .drop_duplicates()
            .head(max_quotes)
            .tolist()
        )
        quotes[theme] = selected
    return quotes


def group_states(series: pd.Series, top_n: int = 8) -> pd.Series:
    counts = series.value_counts(dropna=True)
    top_states = set(counts.head(top_n).index)
    return series.apply(lambda x: x if x in top_states else "Other")


def bin_hours(series: pd.Series) -> pd.Series:
    if detect_numeric_series(series):
        numeric = pd.to_numeric(series, errors="coerce")
        bins = pd.cut(
            numeric,
            bins=[-0.1, 0, 10, 20, 30, np.inf],
            labels=["0", "1-10", "11-20", "21-30", "31+"],
        )
        return bins.astype(str).replace("nan", np.nan)
    return series


def find_keyword_columns(columns: Iterable[str], keywords: List[str]) -> List[str]:
    results = []
    for col in columns:
        col_norm = normalize_column_name(col)
        if all(keyword in col_norm for keyword in keywords):
            results.append(col)
    return results


def summarize_rankings(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    results = []
    for col in columns:
        series = df[col]
        kind, numeric = detect_rank_or_rating(series)
        if kind == "rank" and numeric is not None:
            results.append(
                {
                    "item": col,
                    "metric_type": "rank",
                    "mean_rank": numeric.mean(),
                    "median_rank": numeric.median(),
                    "pct_top_1": (numeric == 1).mean() * 100,
                    "pct_top_3": (numeric <= 3).mean() * 100,
                }
            )
        elif kind == "rating" and numeric is not None:
            results.append(
                {
                    "item": col,
                    "metric_type": "rating",
                    "mean_rating": numeric.mean(),
                }
            )
        else:
            results.append(
                {
                    "item": col,
                    "metric_type": "categorical",
                }
            )
    return pd.DataFrame(results)


def subgroup_rankings(df: pd.DataFrame, columns: List[str], group_col: str) -> pd.DataFrame:
    grouped = []
    for group, sub_df in df.groupby(group_col, dropna=True):
        summary = summarize_rankings(sub_df, columns)
        summary.insert(0, "group", group)
        grouped.append(summary)
    if not grouped:
        return pd.DataFrame()
    return pd.concat(grouped, ignore_index=True)


def safe_value_counts(series: pd.Series) -> pd.Series:
    return series.value_counts(dropna=False)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
