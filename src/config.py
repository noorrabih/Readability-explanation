"""Paths to external resources. Each one can be overridden with an environment variable."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIR = REPO_ROOT / "third_party"

# Make the vendored camel_parser / conllx_df importable as top-level packages.
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

# CAMeL Morph MSA database used by the morphological analyzer (see README).
CAMEL_MORPH_DB = Path(os.environ.get(
    "CAMEL_MORPH_DB", REPO_ROOT / "resources" / "camel_morph_msa_v1.0.db"))

# CAMeL Parser models (downloaded by scripts/download_models.py).
PARSER_MODEL_DIR = Path(os.environ.get(
    "CAMEL_PARSER_MODELS", REPO_ROOT / "resources" / "parser_models"))
PARSER_MODEL_NAME = "CAMeLBERT-CATiB-biaffine.model"

CLITIC_FEATS_CSV = THIRD_PARTY_DIR / "camel_parser" / "data" / "clitic_feats.csv"

# Feature -> BAREC level / reasoning code / Arabic description.
FEATURE_LEVELS_CSV = REPO_ROOT / "resources" / "feature_levels.csv"
