# Readability-explanation

This repository contains the scripts for the rule-based explanation system and its evaluation in our
paper [Benchmarking Fine-Grained Readability Explanations for Arabic](PAPER_LINK). The BAREC-X
dataset is available on [Hugging Face](https://huggingface.co/datasets/Noorrabie/test).

Given an Arabic sentence and its readability level on the 19-level
[BAREC](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Corpus-v1.0) scale, the system returns the
**reasons** that justify that level, following the BAREC annotation guidelines. Each reason is a
label code (e.g. `S11-2`) with its Arabic description, grouped into six dimensions: spelling,
word count, morphology, syntax, vocabulary and content.

The system is training-free. It analyses the sentence with
[CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) (morphology) and the
[CAMeL Parser](https://github.com/CAMeL-Lab/camel_parser) (dependency syntax), and returns the
linguistic features found in the sentence whose guideline level matches the given level.

Example: for `عالم غريب المعالم` at level 11, it returns

```
ID,Label,Reasons,Codes,Arabic_Descriptions
10600010006,11,"[""idafa_lafzia""]","{""syntactic"": [""S11-2""]}","[""إضافة خيالية (لفظية) طويل القامة""]"
```

## Repository layout

```
src/
  disambig_features.py   morphological / orthographic features (CAMeL Tools + CAMeLBERT disambiguation)
  tree_parsing.py        batching + dependency parsing to CoNLL-X (CAMeL Parser, CATiB)
  parser_features.py     syntactic features from the parse trees
  rules.py               feature -> guideline level rules and the reasons for a given level
  config.py              resource paths (overridable with environment variables)
scripts/
  download_models.py     download the CAMeL Parser models
  extract_features.py    sentences -> full_features.csv
  predict.py             full_features.csv + levels -> reasons (use --mode explain)
  evaluate.py            reasons vs. human explanations (C / RC)
resources/
  feature_levels.csv     feature -> BAREC level, label code, Arabic description
```

## Setup

Tested with Python 3.9.

```bash
# Python dependencies (includes conllx_df, pinned to the commit used in our experiments)
pip install -r requirements.txt

# CAMeL Parser, pinned to the commit used in our experiments (later versions parse differently).
# The sed line makes one absolute import relative so it can be imported from this repo.
git clone https://github.com/CAMeL-Lab/camel_parser third_party/camel_parser
git -C third_party/camel_parser checkout be07811eeabf3e11cefbe440922d8eab6ba573d0
sed -i 's/^from src.utils.conll_fixes/from .utils.conll_fixes/' third_party/camel_parser/src/data_preparation.py

# CAMeL Tools data: the MSA BERT disambiguator and the calima-msa-s31 morphology DB
# (calima-msa-s31 requires an LDC license; see the CAMeL Tools documentation)
camel_data -i disambig-bert-unfactored-msa
camel_data -i morphology-db-msa-s31

# CAMeL Parser models -> resources/parser_models/
python scripts/download_models.py
```

You also need the **CAMeL Morph MSA database** (`camel_morph_msa_v1.0.db`) from
[CAMeL Morph](https://github.com/CAMeL-Lab/camel_morph). Put it at `resources/camel_morph_msa_v1.0.db`
or point to it with:

```bash
export CAMEL_MORPH_DB=/path/to/camel_morph_msa_v1.0.db
```

## Usage

The input is a CSV with `ID`, `Sentence` and `Readability_Level_19` (the level to explain) columns.

**1. Extract features.**

```bash
python scripts/extract_features.py --input my_sentences.csv --out-dir outputs/mine
```

This writes `outputs/mine/full_features.csv` along with the intermediate files (batches, CoNLL-X
parses, per-module feature files).

**2. Get the reasons for each sentence's level.**

```bash
python scripts/predict.py --features outputs/mine/full_features.csv --out outputs/mine/reasons.csv --mode explain
```

For each sentence, `Reasons` lists the features found in the sentence that the guidelines assign to
its level, `Codes` gives their label codes grouped by dimension, and `Arabic_Descriptions` their
descriptions. There is also one column per dimension with comma-separated codes.

**3. Evaluate against human explanations.**

```bash
python scripts/evaluate.py --gold annotations.parquet --pred outputs/mine/reasons.csv --criteria RC --out outputs/mine/eval_RC.csv
python scripts/evaluate.py --gold annotations.parquet --pred outputs/mine/reasons.csv --criteria C  --out outputs/mine/eval_C.csv
```

| Criteria | What is compared |
|---|---|
| `RC` | The label codes themselves (e.g. `S10-2` vs. `S10-3`) |
| `C` | Only which dimensions were used (any code in a dimension counts as that dimension) |

For each dimension and for all dimensions combined, it reports precision, recall, F1, Jaccard,
minimal match (at least one shared code) and exact match, as macro (mean over sentences) and micro
(pooled TP/FP/FN) averages. Per-sentence scores go to `--out` and the summary to `*_summary.csv`.
`--gold` and `--pred` accept BAREC-X parquet files or CSVs in the format written by `predict.py`.

## Reproducing the BAREC-X results

```bash
huggingface-cli download Noorrabie/test --repo-type dataset --local-dir data/barecx
python -c "import pandas as pd; pd.read_parquet('data/barecx/annotations.parquet').rename(columns={'id': 'ID', 'sentence': 'Sentence', 'readability_level': 'Readability_Level_19'})[['ID', 'Sentence', 'Readability_Level_19']].to_csv('data/barecx/input.csv', index=False)"

python scripts/extract_features.py --input data/barecx/input.csv --out-dir outputs/barecx
python scripts/predict.py --features outputs/barecx/full_features.csv --out outputs/barecx/reasons.csv --mode explain
python scripts/evaluate.py --gold data/barecx/annotations.parquet --pred outputs/barecx/reasons.csv --criteria RC --out outputs/barecx/eval_RC.csv
python scripts/evaluate.py --gold data/barecx/annotations.parquet --pred outputs/barecx/reasons.csv --criteria C  --out outputs/barecx/eval_C.csv
```

This reproduces the dataset's `rule_based.parquet` exactly (micro F1: 0.666 RC, 0.780 C).

## How the rules work

- `resources/feature_levels.csv` assigns each linguistic feature the BAREC level at which the
  guidelines introduce it, together with its label code and Arabic description.
- A feature counts when it is present in the sentence (non-zero count or true flag).
- The number of unique words and the maximum number of syllables per word use range tables
  (`RANGES_UNIQUE_WORDS`, `RANGES_SYLLABLES` in `rules.py`), with level-specific codes
  (`W{level}-1`, `O{level}-1`).
- The given level selects the feature set: levels 1–11 use the morphological/syntactic features
  (`RELEVANT_COLUMNS_11`); levels 12–19 use `RELEVANT_COLUMNS_19`.
- The reasons are the features in the sentence whose guideline level equals the given level. For
  levels 12–19, when none is found, the vocabulary labels of that level are returned.

## Citation

```bibtex
TODO
```

## Acknowledgements

This project builds on [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools),
[CAMeL Parser](https://github.com/CAMeL-Lab/camel_parser), [conllx_df](https://github.com/CAMeL-Lab/conllx_df),
[CAMeL Morph](https://github.com/CAMeL-Lab/camel_morph) and the
[BAREC corpus](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Corpus-v1.0).
