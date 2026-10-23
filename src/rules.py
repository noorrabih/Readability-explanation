"""
Rule-based BAREC readability level prediction with interpretable reasons.

Each linguistic feature is mapped to a BAREC level in resources/feature_levels.csv.
A sentence's predicted level is the highest level among the features present in it,
and the features that reach that level are returned as the reasons (plus their
reasoning codes and Arabic descriptions).

Levels 1-11 are explained by morphological/syntactic features; levels 12-19 by a
small set of features (exceptions, sentence length, symbols).
"""
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import FEATURE_LEVELS_CSV

RELEVANT_COLUMNS_11 = [
    'hal_interrog', 'madd_alif', 'verb', 'adj', 'akho_abo', 'ordinal_number',
    'noun_adj', 'noon_niswa', 'interrogative_tools', 'idafa_hakikiya',
    'multiple_verbs', 'exception',
    'demonstrative_pronoun_singular', 'preposition',
    'demonstrative_pronoun_plural_dual', 'negation_particle',
    'relative_pronoun_singular', 'relative_pronoun_dual_plural',
    'syllables', 'num_1_10_digits', 'num_11_100_digits',
    'num_101_1000_digits', 'num_1001_1e6_digits', 'num_over_1e6_digits',
    'unique_words', 'has_number_words',
    'imperfective_singular', 'prc_Al_det', 'suf_1s_pron', 'prc_waw',
    'verb_present_plural', 'prc_prep', 'suf_pron', 'dual_noun_adj',
    'plural_fem_noun_adj', 'verb_past_s_p', 'plural_masc',
    'verb_past_present_dual', 'verb_command', 'suf_dual_pron',
    'broken_plural', 'waw_alqassam', 'verb_command_plural', 'amma_lakin',
    'verb_command_dual', 'interrogative_alif', 'ba_alqassam',
    'passive_voice', 'ta_alqassam', 'pronoun', 'proper_noun', 'no_obj',
    'obj', 'jar_majroor', 'verbal_present_sentence_with_an_almasdariya',
    'verbal_sentence_with_two_objects', 'vocative', 'kana_wa_akhawataha',
    'advanced_khabar', 'nominal_with_tpc', 'idafa_lafzia',
    'nominal_sentence', 'inna_wa_akhawataha',
    'weak_final', 'other_proclitics', 'single_word', 'parenthetical_clause',
]

RELEVANT_COLUMNS_19 = ['exception', 'unique_words', 'symbol']

RANGES_UNIQUE_WORDS = {
    range(1, 2): 1, range(2, 3): 2, range(3, 5): 3, range(5, 7): 4,
    range(7, 9): 5, range(9, 10): 6, range(10, 11): 7, range(11, 12): 8,
    range(12, 13): 9, range(13, 16): 10, range(16, 21): 11, range(21, 1000): 12
}
RANGES_SYLLABLES = {
    range(1, 3): 1, range(3, 4): 2, range(4, 5): 5, range(5, 6): 6, range(6, 7): 7
}
UNIQUE_WORDS_CAP = 12
SYLLABLES_CAP = 8

CATEGORY_NAMES = {
    'O': 'Orthography',
    'W': 'word count',
    'M': 'Morphology',
    'S': 'syntactic',
    'V': 'vocab',
    'C': 'content',
}


def _load_column_map(path, value_col: str) -> Dict[str, Any]:
    df = pd.read_csv(path)
    df = df[df['Feature'].notna() & df[value_col].notna()]
    return dict(zip(df['Feature'], df[value_col]))


feature_barec = _load_column_map(FEATURE_LEVELS_CSV, 'BAREC Level')
feature_code = _load_column_map(FEATURE_LEVELS_CSV, 'Code')
feature_desc = _load_column_map(FEATURE_LEVELS_CSV, 'Arabic Description')


def map_features_to_code(features: List[str], level: int,
                         feature_code_map: Dict[str, str] = feature_code) -> List[str]:
    """Codes for `features` at `level`. Range features (syllables, word count) have
    level-specific codes, written as templates like 'W{level}-1' in feature_levels.csv."""
    codes = []
    for feat in features:
        code = feature_code_map.get(feat)
        codes.append(code.format(level=level) if isinstance(code, str) else code)
    return codes


def get_arabic_descriptions(reasons: List[str], desc_map: Dict[str, str] = feature_desc) -> List[str]:
    return [desc_map.get(r, "") for r in reasons]


def _value_to_level(feature: str, value: Any, feature_barec: Dict[str, int]) -> int:
    """Map a single feature value to its BAREC level using special rules where needed."""
    # Treat None/NaN as 0
    if pd.isna(value):
        value = 0

    # Only count signals that are "present":
    # numerical: nonzero; boolean: True; strings: non-empty; others: truthy
    if isinstance(value, (int, float)):
        is_present = (value != 0)
    elif isinstance(value, str):
        is_present = (value.strip() != "")
    else:
        is_present = bool(value)

    if not is_present:
        return 0

    if feature == 'unique_words':
        v = int(value)
        if v >= 21:
            return UNIQUE_WORDS_CAP
        return next(level for rng, level in RANGES_UNIQUE_WORDS.items() if v in rng)

    if feature == 'syllables':
        v = int(value)
        if v >= 8:
            return SYLLABLES_CAP
        # find in defined ranges, fallback 0 if out of all ranges
        return next((level for rng, level in RANGES_SYLLABLES.items() if v in rng), 0)

    return int(feature_barec.get(feature, 0))


def group_codes_by_category(codes, category_names=CATEGORY_NAMES, keep_unknown=False):
    """
    Group codes by their starting letter using `category_names`.
    Special case: the exact code "O" -> 'Orthography'.
    Preserves input order and duplicates. Skips empty values and the 'Code' header.
    """
    grouped = defaultdict(list)
    unknown = []

    for raw in codes:
        code = str(raw).strip()
        if not code or code.lower() in ('code', 'none'):
            continue

        if code == 'O':
            grouped['Orthography'].append(code)
            continue

        first = code[0]
        if first in category_names:
            grouped[category_names[first]].append(code)
        elif keep_unknown:
            unknown.append(code)

    if keep_unknown and unknown:
        grouped['unknown'] = unknown

    return dict(grouped)


def _columns_for_level(level: int) -> List[str]:
    return RELEVANT_COLUMNS_19 if level > 11 else RELEVANT_COLUMNS_11


def predict_level_and_reasons(instance: Dict[str, Any], label: int) -> Tuple[int, List[str]]:
    """
    Compute the predicted BAREC level for an instance and the features that support it.

    The gold label only selects which feature set is used: levels 1-11 use
    RELEVANT_COLUMNS_11, levels 12-19 use RELEVANT_COLUMNS_19.

    Returns:
        (max_level, reasons): reasons are the features whose level == max_level.
    """
    per_feature_levels: Dict[str, int] = {}
    max_level = 0

    for feat in _columns_for_level(label):
        # If the feature is missing in instance, treat as absent (0)
        lvl = _value_to_level(feat, instance.get(feat, 0), feature_barec)
        per_feature_levels[feat] = lvl
        max_level = max(max_level, lvl)

    reasons = [f for f, lvl in per_feature_levels.items() if lvl == max_level and max_level > 0]
    return max_level, reasons


def predict_final_level_and_reasons(instance: Dict[str, Any], label: int) -> Tuple[int, List[str], Dict, str]:
    """
    Predict a level and its reasons, falling back to the gold label for levels 12-19
    when the rules do not reach it.

    Returns:
        (final_level, reasons, codes grouped by category, source) where source is
        "rules" or "label".
    """
    pred, reasons = predict_level_and_reasons(instance, label)
    source = "rules"

    if label > 11 and pred != label:
        pred, source = label, "label"
        # Explain with the 12-19 features that the guidelines map to this label
        reasons = [feat for feat in RELEVANT_COLUMNS_19
                   if feat != 'exception' and feature_barec.get(feat) == label]

    codes = group_codes_by_category(map_features_to_code(reasons, pred))
    return pred, reasons, codes, source


def reasons_for_level(level: int, instance: Dict[str, Any]) -> List[str]:
    """
    Return the features in `instance` whose computed BAREC level == `level`.
    For levels 12-19 with no matching feature, fall back to the vocabulary features
    that the guidelines map to this level ('vocab_X.Y' / 'samerMSA_Z.0').
    """
    matches = [feat for feat in _columns_for_level(level)
               if _value_to_level(feat, instance.get(feat, 0), feature_barec) == level]
    if matches:
        return matches

    return [
        f for f, lvl in feature_barec.items()
        if lvl == float(level) and level > 11
        and (str(f).startswith("vocab_") or str(f).startswith("samerMSA_"))
    ]


def reasons_and_codes_for_level(level: int, instance: Dict[str, Any]) -> Tuple[List[str], Dict]:
    feats = reasons_for_level(level, instance)
    codes = group_codes_by_category([c for c in map_features_to_code(feats, level) if c])
    return feats, codes
