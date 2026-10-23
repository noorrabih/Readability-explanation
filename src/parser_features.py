"""
Arabic Syntax Tree Feature Extraction from CoNLL-X Files

This script processes dependency trees in .conllx format to extract syntactic features
from Arabic sentences. It uses token-level dependency relations and POS tags to identify
specific grammatical constructions.

Key Features:
- Handles batch processing of multiple .conllx files.
- Extracts sentence-level syntactic features via dependency parsing.
"""
import glob
from typing import List

import pandas as pd

from .tree_parsing import arclean, get_parser_components
from camel_parser.src.classes import TextParams
from camel_parser.src.conll_output import text_tuples_to_string
from camel_parser.src.data_preparation import parse_text
from conllx_df.src.conllx_df import ConllxDf
from conllx_df.src.conll_utils import get_token_details, add_parent_details, add_direction


def preprocess_conllx_text(lines):
    """
    Preprocesses lines to remove newlines from multiline # text entries.

    Args:
        lines (list): Lines of the .conllx file.

    Returns:
        List[str]: Preprocessed lines.
    """
    processed_lines = []
    text_buffer = None  # Buffer to store multiline # text content

    for line in lines:
        if line.startswith("# text ="):
            if text_buffer:  # Save the previous # text if it exists
                processed_lines.append(text_buffer.strip() + "\n")
                text_buffer = None

            # Start buffering this # text
            text_buffer = line.strip()

        elif text_buffer and not line.startswith("#"):  # Part of a multiline # text
            text_buffer += " " + line.strip()
        else:
            if text_buffer:  # Save the buffered # text
                processed_lines.append(text_buffer.strip() + "\n")
                text_buffer = None
            processed_lines.append(line)  # Regular line

    # If there's still a buffered # text at the end
    if text_buffer:
        processed_lines.append(text_buffer.strip() + "\n")

    return processed_lines


kana_set = ["كان", "صار", "أصبح","زال", "ليس", "أمسى", "بات", "ظل", "أضحى", "برح", "فتئ", "دام", "انفك"]
inna_w_akhawataha = ['إن', 'أن', 'كأن', 'لكن', 'لعل', 'ليت']

def extract_features(sentences: List[pd.DataFrame]) -> pd.DataFrame:
    for sen_df in sentences:
        # Step 1: Build auxiliary structures
        tokens = {int(row["ID"]): get_token_details(sen_df, int(row["ID"])) for _, row in sen_df.iterrows()}
        children_map = {token_id: [] for token_id in tokens}

        for token in tokens.values():
            add_parent_details(sen_df, token)
            add_direction(token)
            if token.head in children_map:
                children_map[token.head].append(token)

        # Initialize feature flags
        vrbs_without_obj = False
        vrbs_with_obj = False
        prt_with_obj = False
        vrb_with_mod_and_prt_vrb = False
        prt_an_with_vrb_imp = False
        check_obj = True
        vrb_with_two_objs = False
        vocative_case = False
        kana_wa_akhawataha = False
        advanced_khabar = False
        nominal_with_tpc = False
        idafa_lafzia = False
        idafa_hakikiya = False
        inna_with_prd = False
        is_nominal_sentence = False

        # Check for "TPC" in the sentence
        has_tpc = any(token.deprel == "TPC" for token in tokens.values())

        for token in tokens.values():
            # 1. جملة فعلية بدون مفعول به # if theres one maf3ool bihi -> False, vrbs_without_obj
            # has object , vrbs_with_obj
            if len(tokens) > 1:
                if check_obj:  # Skip if already found an obj
                    if token.pos == "VRB":
                        has_obj = any(child.deprel == "OBJ" for child in children_map[token.token_id])
                        vrbs_without_obj = not has_obj
                        check_obj = False
                        vrbs_with_obj = has_obj

            # 2. جار+مجرور
            if not prt_with_obj:  # Skip if already found
                if token.pos == "PRT" and "pos=prep" in sen_df.loc[sen_df['ID'] == token.token_id, 'FEATS'].values[0] :
                    has_obj = any(child.deprel == "OBJ" for child in children_map[token.token_id])
                    prt_with_obj = has_obj

            # 3. جمل فعلية معطوفة
            if not vrb_with_mod_and_prt_vrb: # Skip if already found
                if token.pos == "VRB":
                    for child in children_map[token.token_id]:
                        if child.deprel == "MOD" and child.pos == "PRT":
                            if "pos=conj|" in sen_df.loc[sen_df['ID'] == child.token_id, 'FEATS'].values[0]:
                                if any(grandchild.pos == "VRB" and grandchild.deprel == "OBJ" for grandchild in children_map[child.token_id]):
                                    vrb_with_mod_and_prt_vrb = True

            # 4. جمل فعلية (مضارعة) مع أن المصدرية
            if not prt_an_with_vrb_imp: # Skip if already found
                if token.pos == "PRT" and token.lemma == "أن" :
                    if any(child.pos == "VRB" and child.deprel == "OBJ" and "asp=i" in sen_df.loc[sen_df['ID'] == child.token_id, 'FEATS'].values[0]
                        for child in children_map[token.token_id]):
                        prt_an_with_vrb_imp = True

            # 5. جملة فعلية تتعدى إلى مفعولين
            if not vrb_with_two_objs: # Skip if already found
                if token.pos == "VRB":
                    obj_children = [child for child in children_map[token.token_id] if child.deprel == "OBJ"]
                    if len(obj_children) == 2:
                        vrb_with_two_objs = True
            # 6. المنادى
            if not vocative_case: # Skip if already found
                if token.pos == "PRT" and "pos=part_voc" in sen_df.loc[sen_df['ID'] == token.token_id, 'FEATS'].values[0]:
                    if any(child.deprel == "OBJ" for child in children_map[token.token_id]):
                        vocative_case = True

            # inna wa akhawataha
            if not inna_with_prd and token.lemma in inna_w_akhawataha and ("pos=verb_pseudo" in sen_df.loc[sen_df['ID'] == token.token_id, 'FEATS'].values[0]):
                if any(child.deprel == "PRD" for child in children_map[token.token_id]):
                    inna_with_prd = True

            # كان وأخواتها
            if not kana_wa_akhawataha: # Skip if already found
                if token.pos == "VRB" and token.lemma in kana_set:
                    if any(child.deprel == "PRD" for child in children_map[token.token_id]):
                        kana_wa_akhawataha = True

            # الجملة الاسمية
            if not has_tpc and token.pos == "PRT" and token.lemma not in inna_w_akhawataha:
                has_sbj_child = any(
                    child.deprel == "SBJ" and child.pos != "VRB" for child in children_map[token.token_id]
                )
                if has_sbj_child:
                    is_nominal_sentence = True
                    # break  # Exit early once the condition is satisfied

            # خبر مقدم / مبتدأ مؤخر
            if is_nominal_sentence:
                for child in children_map[token.token_id]:
                    if child.deprel == "SBJ" and child.pos != "VRB" and sen_df[sen_df['ID'] == token.token_id].index[0]  < sen_df[sen_df['ID'] == child.token_id].index[0]:
                        advanced_khabar = True
                        # break  # Exit early once the condition is satisfied


            # جملة أسمية خبرها جملة أسمية (فيها مبتدآن)
            if not nominal_with_tpc: # Skip if already found
                if token.pos != "VRB":
                    if any(child.deprel == "TPC" for child in children_map[token.token_id]):
                        nominal_with_tpc = True

            # إضافة خيالية (لفظية)
            if not idafa_lafzia: # Skip if already found
                if token.pos == "NOM" and "pos=adj" in sen_df.loc[sen_df['ID'] == token.token_id, 'FEATS'].values[0]:
                    if any(child.deprel == "IDF" for child in children_map[token.token_id]):
                        idafa_lafzia = True
            # إضافة حقيقية
            if not idafa_hakikiya: # Skip if already found

                if token.pos == "NOM" and ("pos=adj" not in sen_df.loc[sen_df['ID'] == token.token_id, 'FEATS'].values[0]):
                    if any(child.deprel == "IDF" for child in children_map[token.token_id]):
                        idafa_hakikiya = True

        # Step 3: Append results
        results = {
            "no_obj": vrbs_without_obj,
            "obj" : vrbs_with_obj,
            "jar_majroor": prt_with_obj,
            "coordinated_verbs": vrb_with_mod_and_prt_vrb,
            "verbal_present_sentence_with_an_almasdariya": prt_an_with_vrb_imp,
            "verbal_sentence_with_two_objects" : vrb_with_two_objs,
            "vocative": vocative_case,
            "kana_wa_akhawataha": kana_wa_akhawataha,
            "advanced_khabar": advanced_khabar,
            "nominal_with_tpc": nominal_with_tpc,
            "idafa_lafzia": idafa_lafzia,
            "idafa_hakikiya": idafa_hakikiya,
            "nominal_sentence": is_nominal_sentence,
            "inna_wa_akhawataha": inna_with_prd,

        }

    return results

def process_tree(tree_text, ID):
    """
    Extracts features from a single tree.
    Args:
        tree_text (str): The tree in CoNLL-X format.
        ID (str): ID of the tree.
    Returns:
        dict: The extracted features plus the 'ID'.
    """
    conll_df = ConllxDf(data=tree_text).df
    return {**extract_features([conll_df]), 'ID': ID}

def process_conllx_file(input_file):
    """
    Reads a .conllx file with multiple trees and processes each tree individually.

    Args:
        input_file (str): Path to the input .conllx file.
    Returns:
        List[dict]: One feature dict per tree.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = preprocess_conllx_text(lines)
    # Split the file into trees based on "# id"
    trees = []
    current_tree = []

    for line in lines:
        if line.startswith("# id") and current_tree:
            trees.append(current_tree)
            current_tree = []
        current_tree.append(line)
    if current_tree:
        trees.append(current_tree)

    rows = []
    for tree in trees:
        if tree[0].startswith("# id"):
            ID = tree[0].split('=')[-1].strip()
        rows.append(process_tree("".join(tree), ID))
    return rows

def process_conllx_directory(conllx_directory, output_path):
    """
    Processes all .conllx files in a directory and extracts features.
    Args:
        conllx_directory (str): Path to the directory containing .conllx files.
        output_path (str): Path to the output CSV file.
    Returns:
        pd.DataFrame: The extracted features, one row per sentence.
    """
    rows = []
    for conllx_file in sorted(glob.glob(f"{conllx_directory}/*.conllx")):
        print(f"Processing file {conllx_file}")
        rows.extend(process_conllx_file(conllx_file))

    features_df = pd.DataFrame(rows)
    features_df.to_csv(output_path, index=False)
    return features_df

def extract_parser_features_from_sentence(sentence: str) -> pd.DataFrame:
    """
    Extracts syntactic features from a single Arabic sentence using the CAMeLBERT parser.
    Args:
        sentence (str): The Arabic sentence to process.
    Returns:
        dict: A dictionary containing extracted features.
    """
    # Step 1: Preprocess and parse the sentence
    cleaned_sentence = arclean(sentence)
    model_file, disambiguator, clitic_feats_df, tagset = get_parser_components()
    params = TextParams(
        [cleaned_sentence],
        model_file,
        arclean,
        disambiguator,
        clitic_feats_df,
        tagset,
        ""
    )
    parsed = parse_text("text", params)
    trees_string = text_tuples_to_string(parsed, file_type="conll", sentences=[cleaned_sentence])
    trees_string = trees_string[:-1]  # Only one sentence

    tree_text = "".join(str(line).rstrip("\n") + "\n" for line in trees_string)
    conll_df = ConllxDf(data=tree_text).df
    return extract_features([conll_df])
