# Readability-explanation

An interpretable, rule-based system for **Arabic sentence readability** on the 19-level
[BAREC](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Corpus-v1.0) scale.

Each sentence is analysed with [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools) (morphology)
and the [CAMeL Parser](https://github.com/CAMeL-Lab/camel_parser) (dependency syntax). The resulting
linguistic features (e.g. dual pronouns, passive voice, كان وأخواتها, broken plurals, number of
syllables) are mapped to BAREC levels following the BAREC annotation guidelines. The system outputs
a level for each sentence together with the **reasons**: the features that put it at that level,
their reasoning codes, and Arabic descriptions.

```
ID,Label,Predicted_Level,Source,Reasons,Codes,Arabic_Descriptions
2,10,10,rules,"[""kana_wa_akhawataha""]","{""syntactic"": [""S10-2""]}","[""كان وأخواتها""]"
```

## Repository layout

```
src/
  disambig_features.py   morphological / orthographic features (CAMeL Tools + CAMeLBERT disambiguation)
  tree_parsing.py        batching + dependency parsing to CoNLL-X (CAMeL Parser, CATiB)
  parser_features.py     syntactic features from the parse trees
  rules.py               feature -> level rules, prediction and reasons
  config.py              resource paths (overridable with environment variables)
scripts/
  download_models.py     download the CAMeL Parser models
  extract_features.py    text -> full_features.csv
  predict.py             full_features.csv -> levels + reasons
  evaluate.py            reasoning codes vs. human annotations (C / RC)
resources/
  feature_levels.csv     feature -> BAREC level, reasoning code, Arabic description
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

The BAREC corpus is gated on the Hugging Face Hub. Accept its terms and run `huggingface-cli login`
before using `--split`.

## Usage

**1. Extract features.** Use a BAREC split, or any CSV with `ID`, `Readability_Level_19`, and `Sentence` columns:

```bash
python scripts/extract_features.py --split dev --out-dir outputs/dev
python scripts/extract_features.py --input my_sentences.csv --out-dir outputs/mine
```

This writes `outputs/<name>/full_features.csv` along with the intermediate files (batches, CoNLL-X
parses, per-module feature files).

**2. Predict levels and reasons.** The features file needs a `Readability_Level_19` column:

```bash
python scripts/predict.py --features outputs/dev/full_features.csv --out outputs/dev/predictions.csv
python scripts/predict.py --features outputs/dev/full_features.csv --out outputs/dev/label_reasons.csv --mode explain
```

| Mode | Output |
|---|---|
| `predict` | `Predicted_Level` = highest level among the features present; `Reasons` = the features at that level. For gold levels 12–19, when the rules do not reach the gold level, the gold level is used (`Source = label`) and explained from the guidelines. |
| `explain` | The features in the sentence that the guidelines map to the **gold** level (vocabulary fallbacks for levels 12–19). |

Both modes add `Codes` (grouped into Orthography / word count / Morphology / syntactic / vocab /
content) and `Arabic_Descriptions`, plus one column per code category.

**3. Evaluate reasons against human annotations.** The gold reasoning annotations are the BAREC-X
dataset, [Noorrabie/test](https://huggingface.co/datasets/Noorrabie/test) on the Hugging Face Hub.
`evaluate.py` reads its parquet files directly, as well as CSVs in the format `predict.py` writes
(`ID` plus the six category columns with comma-separated codes).

To reproduce the BAREC-X rule-based results, build the input from the dataset, then run steps 1–3:

```bash
huggingface-cli download Noorrabie/test --repo-type dataset --local-dir data/barecx
python -c "import pandas as pd; pd.read_parquet('data/barecx/annotations.parquet').rename(columns={'id': 'ID', 'sentence': 'Sentence', 'readability_level': 'Readability_Level_19'})[['ID', 'Sentence', 'Readability_Level_19']].to_csv('data/barecx/input.csv', index=False)"

python scripts/extract_features.py --input data/barecx/input.csv --out-dir outputs/barecx
python scripts/predict.py --features outputs/barecx/full_features.csv --out outputs/barecx/label_reasons.csv --mode explain
python scripts/evaluate.py --gold data/barecx/annotations.parquet --pred outputs/barecx/label_reasons.csv --criteria RC --out outputs/barecx/eval_RC.csv
python scripts/evaluate.py --gold data/barecx/annotations.parquet --pred outputs/barecx/label_reasons.csv --criteria C  --out outputs/barecx/eval_C.csv
```

This reproduces `rule_based.parquet` exactly (micro F1: 0.666 RC, 0.780 C).

| Criteria | What is compared |
|---|---|
| `RC` | The reasoning codes themselves (e.g. `S10-2` vs. `S10-3`) |
| `C` | Only which categories were used (any code in a category counts as that category) |

For each category and for all categories combined, it reports precision, recall, F1, Jaccard,
minimal match (at least one shared code) and exact match, as macro (mean over sentences) and micro
(pooled TP/FP/FN) averages. Per-sentence scores go to `--out` and the summary to `*_summary.csv`.

## How the rules work

- `resources/feature_levels.csv` assigns each feature a BAREC level.
- A feature counts when it is present in the sentence (non-zero count or true flag).
- The number of unique words and the maximum syllable count per word use range tables
  (`RANGES_UNIQUE_WORDS`, `RANGES_SYLLABLES` in `rules.py`).
- The gold level selects the feature set: levels 1–11 use the morphological/syntactic features
  (`RELEVANT_COLUMNS_11`); levels 12–19 use `RELEVANT_COLUMNS_19`.


## Acknowledgements

This project builds on [CAMeL Tools](https://github.com/CAMeL-Lab/camel_tools),
[CAMeL Parser](https://github.com/CAMeL-Lab/camel_parser), [conllx_df](https://github.com/CAMeL-Lab/conllx_df),
[CAMeL Morph](https://github.com/CAMeL-Lab/camel_morph) and the
[BAREC corpus](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Corpus-v1.0).
