"""
Run the rule-based system on extracted features.

Modes:
  predict  Predict a BAREC level per sentence and the features that justify it.
           For levels 12-19 the gold label is used when the rules do not reach it.
  explain  Explain the gold label: list the features in the sentence that the
           guidelines map to that level.

    python scripts/predict.py --features outputs/dev/full_features.csv --out outputs/dev/predictions.csv

The features file must contain 'ID' and 'Readability_Level_19' columns.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rules import (  # noqa: E402
    CATEGORY_NAMES, get_arabic_descriptions, predict_final_level_and_reasons,
    reasons_and_codes_for_level)

ID_COL = 'ID'
LABEL_COL = 'Readability_Level_19'


def run(df, mode):
    rows = []
    for _, row in df.iterrows():
        instance = row.to_dict()
        label = int(row[LABEL_COL])
        out = {ID_COL: row[ID_COL], 'Label': label}

        if mode == 'predict':
            pred, reasons, codes, source = predict_final_level_and_reasons(instance, label)
            out.update({'Predicted_Level': pred, 'Source': source})
        else:
            reasons, codes = reasons_and_codes_for_level(label, instance)

        out.update({
            'Reasons': json.dumps(reasons, ensure_ascii=False),
            'Codes': json.dumps(codes, ensure_ascii=False),
            'Arabic_Descriptions': json.dumps(get_arabic_descriptions(reasons), ensure_ascii=False),
        })
        # One column per category with comma-joined codes
        for cat in CATEGORY_NAMES.values():
            out[cat] = ", ".join(codes.get(cat, []))
        rows.append(out)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features', required=True, help='full_features.csv from extract_features.py')
    parser.add_argument('--out', required=True, help='Output CSV path')
    parser.add_argument('--mode', choices=['predict', 'explain'], default='predict')
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    for col in (ID_COL, LABEL_COL):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {args.features}.")

    out = run(df, args.mode)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"Processed {len(out)} rows. Saved to {args.out}")
    if args.mode == 'predict':
        print(f"Exact-match accuracy vs. gold: {(out['Predicted_Level'] == out['Label']).mean():.3f}")


if __name__ == '__main__':
    main()
