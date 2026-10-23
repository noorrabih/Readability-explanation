"""
Arabic Sentence Parsing and CoNLL-X Export for Syntactic Analysis

Batches a dataset of Arabic sentences and parses it with the CAMeL Parser:

1. Splits input data into CSV batches of 100 sentences.
2. Parses each batch into dependency trees (CATiB).
3. Saves the parsed output in CoNLL-X format with sentence IDs for traceability.

Input:  DataFrame with 'ID' and 'Sentence' columns.
Output: CoNLL-X files (one per batch) for syntactic feature extraction.
"""
import os
from functools import lru_cache

import pandas as pd
from camel_tools.utils.charmap import CharMapper

from .config import CLITIC_FEATS_CSV, PARSER_MODEL_DIR, PARSER_MODEL_NAME
from src.classes import TextParams
from src.conll_output import text_tuples_to_string
from src.data_preparation import get_tagset, parse_text
from src.initialize_disambiguator.disambiguator_interface import get_disambiguator

arclean = CharMapper.builtin_mapper("arclean")


def batch_data(data_df, batch_output_dir, batch_size=100):
    """
    Splits the input DataFrame into batches and saves each batch as a CSV file.
    Args:
        data_df: Input DataFrame with 'ID' and 'Sentence' columns.
        batch_output_dir: Directory to save the batch CSV files.
    """
    os.makedirs(batch_output_dir, exist_ok=True)
    for batch_number, start_idx in enumerate(range(0, len(data_df), batch_size), start=1):
        batch = data_df.iloc[start_idx:start_idx + batch_size][["ID", "Sentence"]]
        batch_file_path = os.path.join(batch_output_dir, f"batch_{batch_number}.csv")
        batch.to_csv(batch_file_path, index=False)


@lru_cache(maxsize=None)
def get_parser_components():
    """Load the disambiguator, tagset and clitic features once (shared with parser_features)."""
    model_file = PARSER_MODEL_DIR / PARSER_MODEL_NAME
    if not model_file.exists():
        raise FileNotFoundError(
            f"Parser model not found at {model_file}. Run `python scripts/download_models.py` "
            "or set CAMEL_PARSER_MODELS.")
    clitic_feats_df = pd.read_csv(CLITIC_FEATS_CSV).astype(str).astype(object)
    tagset = get_tagset("catib")
    disambiguator = get_disambiguator("bert", "calima-msa-s31")
    return model_file, disambiguator, clitic_feats_df, tagset


def parse_data(input_folder, output_folder):
    """
    Parses all CSV files in the input folder and saves the parsed output in CoNLL-X format.
    Args:
        input_folder: Directory containing the CSV files to parse.
        output_folder: Directory to save the parsed CoNLL-X files.
    """
    os.makedirs(output_folder, exist_ok=True)
    # Loop over CSV batches and parse them
    for batch_file in os.listdir(input_folder):
        if os.path.exists(os.path.join(output_folder, batch_file.replace(".csv", ".conllx"))):
            print(f"Skipping {batch_file} as it has already been parsed.")
        else:
            if batch_file.endswith(".csv") :
                # Load CSV
                batch_path = os.path.join(input_folder, batch_file)
                batch_df = pd.read_csv(batch_path)

                # Ensure it has 'id' and 'sentence' columns
                if 'ID' not in batch_df.columns or 'Sentence' not in batch_df.columns:
                    raise ValueError(f"CSV {batch_file} must contain 'ID' and 'Sentence' columns.")

                # Extract sentences and IDs
                sentences = batch_df['Sentence'].tolist()
                sentence_ids = batch_df['ID'].tolist()

                # Pass sentences and parameters to TextParams
                model_file, disambiguator, clitic_feats_df, tagset = get_parser_components()
                file_type_params = TextParams(sentences, model_file, arclean, disambiguator, clitic_feats_df, tagset, "")
                parsed_text_tuples = parse_text("text", file_type_params)

                # Convert to CONLLX format string
                trees_string = text_tuples_to_string(parsed_text_tuples, file_type='conll', sentences=sentences)

                # join trees_string to one big string
                trees_string = "\n".join(trees_string)
                # Save to a .conllx file
                output_path = os.path.join(output_folder, batch_file.replace(".csv", ".conllx"))
                with open(output_path, 'w', encoding='utf-8') as conllx_file:
                    # Append sentence IDs as comments for traceability
                    for i, tree in enumerate(trees_string.split('\n\n')):
                        conllx_file.write(f"# id = {sentence_ids[i]}\n")
                        conllx_file.write(tree + "\n\n")
