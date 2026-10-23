"""
Extract morphological and syntactic features for every sentence of a dataset.

Input is either a split of the BAREC corpus on the Hugging Face Hub, or a CSV
with 'ID' and 'Sentence' columns (and optionally 'Readability_Level_19').

    python scripts/extract_features.py --split dev --out-dir outputs/dev
    python scripts/extract_features.py --input my_sentences.csv --out-dir outputs/mine

Writes <out-dir>/full_features.csv, one row per sentence.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_extraction.disambig_features import extract_disambig_feats  # noqa: E402
from feature_extraction.parser_features import process_conllx_directory  # noqa: E402
from feature_extraction.tree_parsing import batch_data, parse_data  # noqa: E402

BAREC_DATASET = "hf://datasets/CAMeL-Lab/BAREC-Corpus-v1.0/"
BAREC_SPLITS = {
    'train': 'data/train-00000-of-00001.parquet',
    'dev': 'data/dev-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet',
}
LABEL_COL = 'Readability_Level_19'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--split', choices=BAREC_SPLITS, help='BAREC corpus split to process')
    source.add_argument('--input', help="CSV file with 'ID' and 'Sentence' columns")
    parser.add_argument('--out-dir', required=True, help='Directory for intermediate files and the output')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split:
        # The BAREC corpus is gated: run `huggingface-cli login` first.
        df = pd.read_parquet(BAREC_DATASET + BAREC_SPLITS[args.split])
    else:
        df = pd.read_csv(args.input)

    # 1. Morphological features (CAMeL Tools disambiguation)
    disambig_feats = extract_disambig_feats(df)
    disambig_feats.to_csv(out_dir / 'disambig_features.csv', index=False)
    print('Done: morphological features')

    # 2. Dependency parsing (CAMeL Parser), 100 sentences per batch
    batch_dir = out_dir / 'batches'
    parsed_dir = out_dir / 'parsed'
    batch_data(df, batch_dir)
    parse_data(batch_dir, parsed_dir)
    print('Done: parsing')

    # 3. Syntactic features from the parse trees
    parser_feats_path = out_dir / 'parser_features.csv'
    process_conllx_directory(parsed_dir, parser_feats_path)
    # Read back so IDs get the same dtype as in disambig_features
    parser_feats = pd.read_csv(parser_feats_path)
    print('Done: syntactic features')

    # 4. Merge everything on ID
    full_features = disambig_feats.merge(parser_feats, on='ID', how='left')
    if LABEL_COL in df.columns:
        full_features = full_features.merge(df[['ID', LABEL_COL]], on='ID', how='left')

    output_path = out_dir / 'full_features.csv'
    full_features.to_csv(output_path, index=False)
    print(f'Saved {len(full_features)} rows to {output_path}')


if __name__ == '__main__':
    main()
