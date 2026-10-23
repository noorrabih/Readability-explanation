"""
Evaluate system reasoning codes against human reasoning annotations.

Each input is either
  - a CSV with an 'ID' column and one column per category (Orthography, word count,
    Morphology, syntactic, vocab, content) holding comma-separated reasoning codes,
    e.g. "S10-2, S10-3" (the format written by scripts/predict.py), or
  - a BAREC-X parquet file (https://huggingface.co/datasets/Noorrabie/test), e.g.
    annotations.parquet, with 'id' and list columns spelling ... content.

Two granularities:
  RC  compare the reasoning codes themselves (e.g. S10-2 vs S10-3)
  C   compare only which categories were used (any code in a category -> its letter)

    python scripts/evaluate.py --gold annotations_RC.csv --pred outputs/dev/label_reasons.csv \
        --criteria RC --out outputs/dev/eval_RC.csv

Metrics per category and for all categories combined: precision, recall, F1,
Jaccard, minimal match (at least one shared code) and exact match, as macro
(mean over sentences) and micro (pooled TP/FP/FN) averages. Per-sentence scores
and summary rows are written to --out.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ['Orthography', 'word count', 'Morphology', 'syntactic', 'vocab', 'content']
CATEGORY_LETTERS = {'Orthography': 'O', 'word count': 'W', 'Morphology': 'M',
                    'syntactic': 'S', 'vocab': 'V', 'content': 'C'}
ALL = 'all_combined'
# BAREC-X dataset column -> category column used here
BARECX_COLUMNS = {'id': 'ID', 'sentence': 'Sentence', 'spelling': 'Orthography',
                  'word_count': 'word count', 'morphology': 'Morphology', 'syntax': 'syntactic',
                  'vocabulary': 'vocab', 'content': 'content'}


def load_table(path) -> pd.DataFrame:
    """Read a predict.py-style CSV, or a BAREC-X parquet file converted to the same format."""
    df = pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path)
    if 'id' in df.columns and 'ID' not in df.columns:
        df = df.rename(columns=BARECX_COLUMNS)
        df['ID'] = df['ID'].astype('int64')
        for cat in CATEGORIES:
            df[cat] = df[cat].apply(lambda codes: ", ".join(codes))
    return df


def parse_codes(cell) -> set:
    """Comma-separated string -> set of trimmed codes. NaN or empty -> empty set."""
    if pd.isna(cell):
        return set()
    return {code.strip() for code in str(cell).split(",") if code.strip()}


def pr_counts(true_set: set, pred_set: set):
    """(tp, fp, fn, precision, recall, f1). P/R/F1 are NaN when both sets are empty."""
    if not true_set and not pred_set:
        return 0, 0, 0, np.nan, np.nan, np.nan
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return tp, fp, fn, precision, recall, f1


# The three set-agreement scores count "both empty" as full agreement (1).
def jaccard(a: set, b: set):
    if not a and not b:
        return 1
    return len(a & b) / len(a | b)


def minimal_match(a: set, b: set):
    if not a and not b:
        return 1
    return 1 if a & b else 0


def exact_match(a: set, b: set):
    if not a and not b:
        return 1
    return 1 if a == b else 0


def to_categories(df: pd.DataFrame) -> pd.DataFrame:
    """C granularity: replace any non-empty category cell with the category letter."""
    df = df.copy()
    for cat, letter in CATEGORY_LETTERS.items():
        df[cat] = df[cat].apply(lambda x: letter if parse_codes(x) else "")
    return df


def score(df: pd.DataFrame, name: str, gold_sets, pred_sets) -> dict:
    """Add per-sentence metric columns for `name`; return its macro and micro summaries."""
    rows = [pr_counts(t, p) for t, p in zip(gold_sets, pred_sets)]
    tp, fp, fn, prec, rec, f1 = (list(col) for col in zip(*rows))
    df[f"precision_{name}"], df[f"recall_{name}"], df[f"f1_{name}"] = prec, rec, f1
    df[f"tp_{name}"], df[f"fp_{name}"], df[f"fn_{name}"] = tp, fp, fn
    df[f"jaccard_{name}"] = [jaccard(t, p) for t, p in zip(gold_sets, pred_sets)]
    df[f"minimal_match_{name}"] = [minimal_match(t, p) for t, p in zip(gold_sets, pred_sets)]
    df[f"exact_match_{name}"] = [exact_match(t, p) for t, p in zip(gold_sets, pred_sets)]

    macro = {m: float(np.nanmean(df[f"{m}_{name}"]))
             for m in ("precision", "recall", "f1", "jaccard", "minimal_match", "exact_match")}

    tp_sum, fp_sum, fn_sum = sum(tp), sum(fp), sum(fn)
    micro = {
        "precision": tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else np.nan,
        "recall": tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else np.nan,
        "f1": 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum) if (tp_sum + fp_sum + fn_sum) > 0 else np.nan,
        "jaccard": tp_sum / (tp_sum + fp_sum + fn_sum) if (tp_sum + fp_sum + fn_sum) > 0 else np.nan,
        "tp": tp_sum, "fp": fp_sum, "fn": fn_sum,
    }
    return {"macro": macro, "micro": micro}


def evaluate(gold: pd.DataFrame, pred: pd.DataFrame, criteria: str):
    for name, frame in (("gold", gold), ("pred", pred)):
        missing = [c for c in ["ID"] + CATEGORIES if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns in {name} file: {missing}")

    if criteria == "C":
        gold, pred = to_categories(gold), to_categories(pred)

    keep_gold = ["ID"] + (["Sentence"] if "Sentence" in gold.columns else []) + CATEGORIES
    df = pd.merge(gold[keep_gold].rename(columns={c: f"gold-{c}" for c in CATEGORIES}),
                  pred[["ID"] + CATEGORIES].rename(columns={c: f"system-{c}" for c in CATEGORIES}),
                  on="ID", how="inner").sort_values("ID").reset_index(drop=True)
    print(f"Gold rows: {len(gold)}, system rows: {len(pred)}, evaluated: {len(df)}")

    summaries = {}
    gold_all = [set() for _ in range(len(df))]
    pred_all = [set() for _ in range(len(df))]
    for cat in CATEGORIES:
        gold_sets = [parse_codes(x) for x in df[f"gold-{cat}"]]
        pred_sets = [parse_codes(x) for x in df[f"system-{cat}"]]
        summaries[cat] = score(df, cat, gold_sets, pred_sets)
        gold_all = [a | b for a, b in zip(gold_all, gold_sets)]
        pred_all = [a | b for a, b in zip(pred_all, pred_sets)]
    summaries[ALL] = score(df, ALL, gold_all, pred_all)
    return df, summaries


def summary_table(summaries: dict) -> pd.DataFrame:
    rows = []
    for name, s in summaries.items():
        rows.append({"category": name, "average": "macro", **s["macro"]})
        rows.append({"category": name, "average": "micro", **s["micro"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gold', required=True, help='Human annotations (CSV or BAREC-X parquet)')
    parser.add_argument('--pred', required=True, help='System output (CSV from scripts/predict.py, or BAREC-X parquet)')
    parser.add_argument('--criteria', choices=['RC', 'C'], default='RC')
    parser.add_argument('--out', required=True, help='Per-sentence scores CSV; a *_summary.csv is written next to it')
    args = parser.parse_args()

    df, summaries = evaluate(load_table(args.gold), load_table(args.pred), args.criteria)
    summary = summary_table(summaries)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(summary.to_string(index=False))
    print(f"\nSaved per-sentence scores to {out} and the summary to {summary_path}")


if __name__ == '__main__':
    main()
