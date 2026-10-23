"""
Arabic Feature Extraction Script for Readability Analysis

This script extracts morphological features from Modern Standard Arabic sentences
using CAMeL Tools and a pretrained CAMeLBERT model. It processes sentences in batches, disambiguates
them, and identifies both word-level and sentence-level features. The output is saved as a CSV for use in readability prediction or linguistic analysis.

"""


import re
import string
from functools import lru_cache

import pandas as pd
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.charsets import (AR_DIAC_CHARSET, AR_LETTERS_CHARSET,
                                        UNICODE_PUNCT_CHARSET)
from camel_tools.utils.transliterate import Transliterator
from tqdm import tqdm

from .config import CAMEL_MORPH_DB

# Initialize tqdm for pandas
tqdm.pandas()

# --- Number utilities (Arabic/Indic digits & words) ---
ARABIC_INDIC_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
# Thousand separators commonly seen in Arabic text
ARABIC_SEPARATORS = ['\u066C', '،', ',']  # '٬' (U+066C), Arabic comma, Western comma

# A contiguous numeric token that may include Arabic/Western thousand separators or decimals
NUMBER_TOKEN_RE = re.compile(
    r'[0-9\u0660-\u0669]+(?:[.,\u066B' + ''.join(ARABIC_SEPARATORS) + r'][0-9\u0660-\u0669]{2,3})*'
)

def _normalize_digits(s: str) -> str:
    """Convert Arabic-Indic digits to Western and remove common thousand separators."""
    s = s.translate(ARABIC_INDIC_DIGITS)
    # Remove directional marks/spaces if present
    s = s.replace('\u200f', '').replace('\u200e', '').strip()
    # Remove thousand separators (Arabic and Western)
    for sep in ARABIC_SEPARATORS:
        s = s.replace(sep, '')
    # Arabic decimal separator (U+066B) -> dot
    s = s.replace('\u066B', '.')
    return s

def _int_from_token(tok: str):
    """Try to convert a numeric token to int (ignore fractional part). Return None on failure."""
    try:
        s = _normalize_digits(tok)
        if '.' in s:
            s = s.split('.', 1)[0]  # keep integer part only
        if not s:
            return None
        return int(s)
    except Exception:
        return None


def count_unique_words(raw_sentence):
    arabic_punctuation = "؟،؛«»ـ…"
    all_punctuation = string.punctuation + arabic_punctuation

    translator = str.maketrans('', '', all_punctuation)
    cleaned_text = raw_sentence.translate(translator)

    words = cleaned_text.split()
    unique_words = set(words)

    return len(unique_words)



WEAK_FINAL_RE = re.compile(r'[يىا]$')

# جمل اعتراضية: content inside (...) or between inner commas that contains Arabic words
PAREN_CLAUSE_RE = re.compile(r'\([^()]*[\u0600-\u06FF]{2,}[^()]*\)')

qassam_lex = ['حَقّ', 'رَبّ' , 'شَمْس', 'قَمَر', 'تِين', 'اللّٰه', 'حَياة','مَوْت', 'دِين', 'قُرْآن', 'كَعْبَة', 'سَماء', 'عَصْر', 'فَجْر', 'لَيْل', 'أَرْض', 'كَعْبَة', 'جَبَل', 'مَلْأَك', 'جَبْرَئِيل', 'ميكائيل', 'مُحَمَّد', 'مُوسَى', 'عِيسَى', 'إِبْرَاهِيم', 'نُوح', 'يُونُس', 'يُوسُف', 'يَعْقُوب', 'إِسْمَاعِيل', 'إِسْحَاق', 'إِدْرِيس', 'نَبِيّ'  ]

tanween = ['ً', 'ٍ', 'ٌ']

# Define vowels
vowels = ['a', 'e', 'i', 'o', 'u', 'aa', 'ee', 'ii', 'oo', 'uu']
arabic_vowels = ['ي', 'و', 'ا', 'ى', 'ؤ', 'ئ', 'إ', 'أ', 'آ', 'ة']
AR_LETTERS_PATTERN = "[" + "".join(AR_LETTERS_CHARSET) + "]"

db = MorphologyDB(str(CAMEL_MORPH_DB), "a")

# Initialize the analyzer
analyzer = Analyzer(db, 'ADD_PROP', cache_size=100000)

# Load the pretrained BERT model
bert = BERTUnfactoredDisambiguator.pretrained(model_name='msa', pretrained_cache=False)

# Set the analyzer
bert._analyzer = analyzer


@lru_cache(maxsize=1024)  # Caches results of disambiguation
def disambiguate_sentence(sentence_text):
    # Tokenize the sentence
    sentence = simple_word_tokenize(sentence_text)

    # Tag and disambiguate the sentence
    disambig = bert.tag_sentence(sentence)

    return disambig

def extract_word_features(word_analysis):
    """
    Extract morphological features from a single word analysis.
    Args:
        word_analysis (dict): A dictionary containing morphological analysis of a word.
    Returns:
        features (dict): A dictionary containing extracted features for the word.
    """
    # Unpack values from analysis with default values
    pos = word_analysis.get('pos', '')
    num = word_analysis.get('num', '')
    asp = word_analysis.get('asp', '')
    gen = word_analysis.get('gen', '')
    form_gen = word_analysis.get('form_gen', '')
    form_num = word_analysis.get('form_num', '')
    enc0 = word_analysis.get('enc0', '')
    prc0 = word_analysis.get('prc0', '')
    prc1 = word_analysis.get('prc1', '')
    prc2 = word_analysis.get('prc2', '')
    prc3 = word_analysis.get('prc3', '')
    bw = word_analysis.get('bw', '')
    lex = word_analysis.get('lex', '')
    vox = word_analysis.get('vox', '')
    diac = word_analysis.get('diac', '')
    per = word_analysis.get('per', '')


    # مفرد مضارع
    def imperfective_singular():
        return num == 's' and asp == 'i' and pos == 'verb'

    # سوابق: ال التعريف
    def prc_Al_det():
        return prc0 == 'Al_det'

    #  حرف  العطف  واو (سابقة)
    def prc_waw():
        return prc2 == 'wa_conj'

    # لواحق: ضمير المتكلم المفرد المتصل
    def suf_1s_pron():
        suf = True if (enc0 in ['1s_pron', '1s_poss', '1s_dobj']) else False
        return suf

    #  الفعل المضارع الجمع
    def verb_present_plural():
        return pos == 'verb' and asp == 'i' and num == 'p'

    # سوابق حروف جر متصلة (ب+ ل+ ك+)
    def prc_prep():
        return prc1 in ('bi_prep', 'li_prep', 'ka_prep')

    # لواحق: ضمير  متصل مفرد أو جمع
    def suf_pron():
        # Define the list of pronoun types
        prefixes = [
            "1p", "2ms", "2fs", "2mp", "2fp", "2p", "3ms", "3fs", "3mp", "3fp", "3p"
        ]
        suffixes = [
            "dobj", "poss", "pron"
        ]

        # Combine prefixes and suffixes to form the full list of pronouns
        valid_enc0_values = [f"{prefix}_{suffix}" for prefix in prefixes for suffix in suffixes]

        return enc0 in valid_enc0_values

    # المثنى (في الأسماء والصفات)
    def dual_noun_adj():
        return num == 'd' and pos in ['noun', 'adj', 'noun_quant', 'adj_comp']

    # جمع المؤنث السالم في الأسماء فقط والصفات للكلمات الشائعة (سيارات، معلّمات)
    def plural_fem_noun_adj():
        return form_num == 'p' and form_gen == 'f' # and pos in ['noun', 'adj']

    # الفعل الماضي المفرد والجمع
    def verb_past_s_p():
        return pos == 'verb' and asp == 'p' and num in ['s', 'p']

    # جمع مذكر سالم
    def plural_masc():
        return form_gen == 'm' and form_num == 'p' and pos in ['noun', 'adj']

    #  الفعل الماضي المثنى  والمضارع المثنى
    def verb_past_present_dual():
        return pos == 'verb' and asp in ['p', 'i'] and num == 'd'

    # فعل الأمر المفرد
    def verb_command():
        return pos == 'verb' and asp == 'c' and num == 's'

    #  لواحق: ضمير المثنى المتصل
    def suf_dual_pron():
        prefixes = ["2d", "3d"]
        suffixes = ["dobj", "poss", "pron"]
        valid_enc0_values = [f"{prefix}_{suffix}" for prefix in prefixes for suffix in suffixes]
        return enc0 in valid_enc0_values

    # جمع التكسير غير الشائع: أقدام، خضروات
    def broken_plural():
        return pos in ['noun', 'adj'] and form_num == 's' and num == 'p'

    # واو القسم (والله)
    def waw_alqassam():
        return prc2 in [ 'wa_prep'] and lex in qassam_lex

    # فعل الأمر الجمع
    def verb_command_plural():
        return asp == 'c' and num == 'p' and pos == 'verb'

    lex_list = ['ثُمَّ', 'حَتَّى', 'أَوْ', 'أَمْ', 'لٰكِنَّ', 'لٰكِنْ', 'أَمّا'] # لٰكِنْ

    #أدوات ربط
    def amma_lakin():
        return lex in lex_list

    # فعل الأمر للمثنى
    def verb_command_dual():
        return asp == 'c' and num == 'd' and pos == 'verb'

    # أداة الاستفهام أ مثل: أسمعتَ.
    def interrogative_alif():
        return prc3 in ['>a_ques', 'A_ques']

    # باء القسم
    def ba_alqassam():
        return prc1 in ['bi_prep'] and lex in qassam_lex

    # المبني للمجهول
    def passive_voice():
        return vox == 'p' and pos == 'verb'

    # تاء القسم
    def ta_alqassam():
        return prc1 in ['ta_prep'] and lex in qassam_lex

    # ضمير منفصل
    def pronoun():
        return pos == 'pron' # , 'pron_exclam', 'pron_interrog', 'pron_rel'

    # اسم علم
    def proper_noun():
        return pos == 'noun_prop'

    # فعل
    def verb():
        return pos == 'verb'

    #صفة
    def adj():
        return pos == 'adj'

    # • أبو – أخو
    def akho_abo():
        akho =  (lex == 'أَخ' and pos == 'noun' and diac == 'أَخُو')
        abo = (lex == 'أَب' and pos == 'noun' and diac == 'أَبُو')
        return akho or abo

    #  العدد الترتيبي
    def ordinal_number():
        return num == 'adj_num'

    # الرموز (ش.م.)
    def symbol():
        return pos == 'abbrev'

    def interrogative_tools():
        return pos in ('adv_interrog', 'pron_interrog')

    # سوابق أخرى: سين الاستقبال، واو الاستئناف/العطف، فاء العطف
    def other_proclitics():
        prcs = [prc0, prc1, prc2, prc3]
        for prc in prcs:
            if not prc:
                continue
            if prc.startswith('sa_'):
                return True
            if prc.startswith('wa_') and prc != 'wa_prep':
                return True
            if prc.startswith('fa_'):
                return True
        return False

    # أفعال/أسماء معتلة الآخر: stem ends with ي/ى/ا, no suffix
    def weak_final():
        if pos not in ('noun', 'verb'):
            return False
        if enc0 and enc0 != '0':  # has a clitic suffix ('0' means none)
            return False
        stem = word_analysis.get('stem', '')
        stem_nodiac = ''.join(ch for ch in stem if ch not in AR_DIAC_CHARSET)
        return bool(WEAK_FINAL_RE.search(stem_nodiac))

    # Return the extracted features as a dictionary
    return {
        'verb': verb(),
        'adj': adj(),
        'akho_abo': akho_abo(),
        'ordinal_number': ordinal_number(),
        'symbol': symbol(),
        'interrogative_tools': interrogative_tools(),
        'imperfective_singular': imperfective_singular(),
        'prc_Al_det': prc_Al_det(),
        'suf_1s_pron': suf_1s_pron(),
        'prc_waw': prc_waw(),
        'verb_present_plural': verb_present_plural(),
        'prc_prep': prc_prep(),
        'suf_pron': suf_pron(),
        'dual_noun_adj': dual_noun_adj(),
        'plural_fem_noun_adj': plural_fem_noun_adj(),
        'verb_past_s_p': verb_past_s_p(),
        'plural_masc': plural_masc(),
        'verb_past_present_dual': verb_past_present_dual(),
        'verb_command': verb_command(),
        'suf_dual_pron': suf_dual_pron(),
        'broken_plural': broken_plural(),
        'waw_alqassam': waw_alqassam(),
        'verb_command_plural': verb_command_plural(),
        'amma_lakin' : amma_lakin(),
        'verb_command_dual': verb_command_dual(),
        'interrogative_alif': interrogative_alif(),
        'ba_alqassam': ba_alqassam(),
        'passive_voice': passive_voice(),
        'ta_alqassam' : ta_alqassam(),
        'pronoun': pronoun(),
        'proper_noun': proper_noun(),
        'weak_final': weak_final(),
        'other_proclitics': other_proclitics()
    }


def pronoun(word_analysis):
    return word_analysis.get('pos') == 'pron'

def imperfective_singular(word_analysis):
    return word_analysis.get('num') == 's' and word_analysis.get('asp') == 'i' and word_analysis.get('pos') == 'verb'

def proper_noun(word_analysis):
    return word_analysis.get('pos') == 'noun_prop'

def count_syllables(caphi, diac, prc0, prc2):
    syllable_count = 0
    if caphi is not None:
        caphi_parts = caphi.split('_')  # Split caphi into individual components
        # Iterate over caphi parts to count syllables
        for i, part in enumerate(caphi_parts):
            if part in vowels:
                if i == len(caphi_parts) - 1:  # If it's the last part (vowel)
                    if diac and diac[-1] in AR_DIAC_CHARSET:  # Check if the last diac char is in AR_DIAC_CHARSET
                        continue  # Don't count this syllable
                syllable_count += 1

        if len(caphi_parts) > 2:
            # tanween
            if caphi_parts[-1] == 'n' and caphi_parts[-2] in vowels and diac[-1] in tanween :
                syllable_count -= 1
        # exclude al
        if prc0 == 'Al_det':
            syllable_count -= 1
        if prc2 == 'wa_conj':
            syllable_count -= 1

    else: # If caphi is None, count syllables based on vowels
        for i in range(len(diac)):
                if diac[i] in arabic_vowels:
                    syllable_count += 1

    return syllable_count

def analyze_sentence(sentence_analysis, raw_sentence):
    """
    Analyze a sentence to extract various linguistic features.
    Args:
        sentence_analysis (list): List of dictionaries containing morphological analysis of each word.
        raw_sentence (str): The original sentence as a string.
    Returns:
        results (dict): A dictionary containing various linguistic features extracted from the sentence.
    """
    syllable_word = ''
    tokenized_raw_sentence = simple_word_tokenize(raw_sentence)
    # جمل اعتراضية: (...) or inner comma-bracketed clause with Arabic words
    def parenthetical_clause():
        if PAREN_CLAUSE_RE.search(raw_sentence):
            return True
        # comma case: split on , or ، — any inner segment (not first/last) with Arabic words
        parts = re.split(r'[,،]', raw_sentence)
        if len(parts) >= 3:
            for middle in parts[1:-1]:
                if re.search(r'[\u0600-\u06FF]{2,}', middle.strip()):
                    return True
        return False

    results = {
        'single_word': len(tokenized_raw_sentence) == 1,
        'parenthetical_clause': parenthetical_clause(),
        'noon_niswa': False,
        'hal_interrog': False,
        'noun_adj': False,
        'pronoun_proper_noun': False,
        'multiple_verbs': False,
        'exception': False,
        'exception_lex': False,
        'demonstrative_pronoun_singular': False,
        'preposition': False,
        'demonstrative_pronoun_plural_dual': False,
        'negation_particle': False,
        'relative_pronoun_singular': False,
        'relative_pronoun_dual_plural': False,
        'syllables': 0,
        'syllable_word': '',
        'num_1_10_digits': 0,
        'num_11_100_digits': 0,
        'num_101_1000_digits': 0,
        'num_1001_1e6_digits': 0,
        'num_over_1e6_digits': 0,
        'has_number_words': False,
        'madd_alif': 0,
    }

    verb_count = 0
    lexs = []
    question_mark = False
    interrog_word = False
    tanween = ['ً', 'ٍ', 'ٌ']
    for i, (word, raw_word) in enumerate(zip(sentence_analysis,tokenized_raw_sentence)):
        caphi = word.get('caphi')
        diac = word.get('diac')
        pos = word.get('pos')
        lex = word.get('lex')
        num = word.get('num')
        prc0 = word.get('prc0', '')
        prc2 = word.get('prc2', '')
        bw = word.get('bw', '')

        # أفعال/أسماء معتلة الآخر

        # نون النسوة في الأسماء والأفعال (انتظرنَ دورهنّ)
        word_nodiac = ''.join(ch for ch in raw_word if ch not in AR_DIAC_CHARSET)
        if pos in ('verb', 'noun') and num == 'p' and word_nodiac and word_nodiac.endswith('ن') and '3FP' in bw:
            results['noon_niswa'] = True

        # أداة الاستفهام هل
        if lex == 'هَلْ' and pos == 'part_interrog':
            results['hal_interrog'] = True
        # <ضمير> + <فعل مضارع>

        # <اسم> + <صفة>
        if i < len(sentence_analysis) - 1 and pos == 'noun' and sentence_analysis[i + 1].get('pos') == 'adj':
            results['noun_adj'] = True

        # <ضمير> + <اسم علم>
        if i < len(sentence_analysis) - 1 and pronoun(word) and proper_noun(sentence_analysis[i + 1]):
            results['pronoun_proper_noun'] = True

        # جمل فعلية معطوفة

        # فعلين أو أكثر في الجملة الواحدة
        if pos == 'verb' and lex not in lexs : # and diac != 'صَحَّ'
            verb_count += 1
            lexs.append(lex)

        # استثناء
        if pos == 'part_restrict' :
            results['exception'] = True


        # ضمير منفصل مفرد

        # أسماء الإشارة المفردة
        if pos == 'pron_dem' and num == 's':
            results['demonstrative_pronoun_singular'] = True

        # حروف الجر المنفصلة والمتصلة
        if pos == 'prep' or word.get('prc1') in ['bi_prep', 'li_prep', 'ka_prep', 'wa_prep', 'ta_prep'] :
            results['preposition'] = True

        #  اسم اشارة مثنى، جمع"
        if pos == 'pron_dem' and num in ('p', 'd'):
            results['demonstrative_pronoun_plural_dual'] = True

        # أحرف النفي
        if pos == 'part_neg' : # or lex in ['لا', 'لَيْسَ', 'غَيْر', 'لَمْ', 'لمّا', 'لَنْ', 'ما', 'لات']
            results['negation_particle'] = True

        # أسماء الوصل المفردة
        if pos == 'pron_rel' and num == 's':
            results['relative_pronoun_singular'] = True

        # أسماء الوصل المثنى والجمع
        if pos == 'pron_rel' and num in ['d', 'p']:
            results['relative_pronoun_dual_plural'] = True

        # Count syllables
        syllable =  count_syllables(caphi, diac, prc0, prc2)
        if syllable > results['syllables']:
            results['syllables'] = syllable
            syllable_word = raw_word

        if pos == 'noun_num':
            results['has_number_words'] = True

    # Handle post-loop conditions:
    results['madd_alif'] = sum(1 for tok in tokenized_raw_sentence if 'آ' in tok)
    results['multiple_verbs'] = verb_count >= 2
    results['syllable_word'] = syllable_word

        # --- Count digit-based numbers across required ranges in the raw sentence ---
    num_1_10 = num_11_100 = num_101_1000 = num_1001_1e6 = num_over_1e6 = 0
    for m in NUMBER_TOKEN_RE.finditer(raw_sentence):
        n = _int_from_token(m.group(0))
        if n is None or n <= 0:
            continue
        if 1 <= n <= 10:
            num_1_10 += 1
        elif 11 <= n <= 100:
            num_11_100 += 1
        elif 101 <= n <= 1000:
            num_101_1000 += 1
        elif 1001 <= n <= 1_000_000:
            num_1001_1e6 += 1
        elif n > 1_000_000:
            num_over_1e6 += 1

    results['unique_words'] = count_unique_words(raw_sentence)
    results['num_1_10_digits'] = num_1_10
    results['num_11_100_digits'] = num_11_100
    results['num_101_1000_digits'] = num_101_1000
    results['num_1001_1e6_digits'] = num_1001_1e6
    results['num_over_1e6_digits'] = num_over_1e6

    return results


# Define feature extraction function
def extract_features_from_row(row):
    """Extract features from a single row of the DataFrame.
    Args:
        row (pd.Series): A single row of the DataFrame containing 'Sentence' and 'ID'.
    Returns:
        features (dict): A dictionary containing extracted features for the row.
        This includes both sentence-level and aggregated word-level features.
    """

    raw_sentence = row['Sentence']
    sentence = raw_sentence
    # Perform sentence-level analysis
    if len(sentence.split(" ")) == 1:
        sentence = '"' + sentence + '"'
    sentence_analysis = disambiguate_sentence(sentence)
    sentence_features = analyze_sentence(sentence_analysis, sentence)
    # Count tokens on the original sentence, not the quote-wrapped one
    sentence_features['single_word'] = len(simple_word_tokenize(raw_sentence)) == 1

    # Extract word-level features and aggregate
    word_features_list = []
    for word_analysis in sentence_analysis:
        word_features = extract_word_features(word_analysis)
        word_features_list.append(word_features)

    # Aggregate word-level features
    aggregated_word_features = {
        'verb': sum(wf.get("verb", 0) for wf in word_features_list),
        'adj': sum(wf.get("adj", 0) for wf in word_features_list),
        'akho_abo': sum(wf.get("akho_abo", 0) for wf in word_features_list),
        'ordinal_number': sum(wf.get("ordinal_number", 0) for wf in word_features_list),
        'symbol': sum(wf.get("symbol", 0) for wf in word_features_list),
        'interrogative_tools': sum(wf.get("interrogative_tools", 0) for wf in word_features_list),
        'imperfective_singular': sum(wf.get("imperfective_singular", 0) for wf in word_features_list),
        'prc_Al_det': sum(wf.get("prc_Al_det", 0) for wf in word_features_list),
        'suf_1s_pron': sum(wf.get("suf_1s_pron", 0) for wf in word_features_list),
        'prc_waw': sum(wf.get("prc_waw", 0) for wf in word_features_list),
        'verb_present_plural': sum(wf.get("verb_present_plural", 0) for wf in word_features_list),
        'prc_prep': sum(wf.get("prc_prep", 0) for wf in word_features_list),
        'suf_pron': sum(wf.get("suf_pron", 0) for wf in word_features_list),
        'dual_noun_adj': sum(wf.get("dual_noun_adj", 0) for wf in word_features_list),
        'plural_fem_noun_adj': sum(wf.get("plural_fem_noun_adj", 0) for wf in word_features_list),
        'verb_past_s_p': sum(wf.get("verb_past_s_p", 0) for wf in word_features_list),
        'plural_masc': sum(wf.get("plural_masc", 0) for wf in word_features_list),
        'verb_past_present_dual': sum(wf.get("verb_past_present_dual", 0) for wf in word_features_list),
        'verb_command': sum(wf.get("verb_command", 0) for wf in word_features_list),
        'suf_dual_pron': sum(wf.get("suf_dual_pron", 0) for wf in word_features_list),
        'broken_plural': sum(wf.get("broken_plural", 0) for wf in word_features_list),
        'waw_alqassam': sum(wf.get("waw_alqassam", 0) for wf in word_features_list),
        'verb_command_plural': sum(wf.get("verb_command_plural", 0) for wf in word_features_list),
        'amma_lakin': sum(wf.get("amma_lakin", 0) for wf in word_features_list),
        'verb_command_dual': sum(wf.get("verb_command_dual", 0) for wf in word_features_list),
        'interrogative_alif': sum(wf.get("interrogative_alif", 0) for wf in word_features_list),
        'ba_alqassam': sum(wf.get("ba_alqassam", 0) for wf in word_features_list),
        'passive_voice': sum(wf.get("passive_voice", 0) for wf in word_features_list),
        'ta_alqassam': sum(wf.get("ta_alqassam", 0) for wf in word_features_list),
        'pronoun': sum(wf.get("pronoun", 0) for wf in word_features_list),
        'proper_noun': sum(wf.get("proper_noun", 0) for wf in word_features_list),
        'weak_final': sum(wf.get("weak_final", 0) for wf in word_features_list),
        'other_proclitics': sum(wf.get("other_proclitics", 0) for wf in word_features_list)
    }

    # Combine sentence and word-level features and id
    features = {'ID': row['ID'], **sentence_features, **aggregated_word_features}

    return features


def extract_features_from_sentence(sentence):
    """Extract features from a single sentence(string).
        Args:
            sentence (str): A single Arabic sentence.
        Returns:
            features (dict): A dictionary containing extracted features.
            pairs (list): A list of tuples containing lemma and pos pairs. (required for further feature extraction)
    """
    raw_sentence = sentence
    # Perform sentence-level analysis
    if len(sentence.split(" ")) == 1:
        sentence = '"' + sentence + '"'
    sentence_analysis = disambiguate_sentence(sentence)
    # from each sentence_analysis, extract lemma and pos pairs
    sentence_features = analyze_sentence(sentence_analysis, sentence)
    # Count tokens on the original sentence, not the quote-wrapped one
    sentence_features['single_word'] = len(simple_word_tokenize(raw_sentence)) == 1

    # Extract word-level features and aggregate
    word_features_list = []
    pairs = []

    for word_analysis in sentence_analysis:
        word_features = extract_word_features(word_analysis)
        word_features_list.append(word_features)
        if 'lex' in word_analysis and 'pos' in word_analysis:
            pairs.append((word_analysis['lex'], word_analysis['pos']))

    # Aggregate word-level features
    aggregated_word_features = {
        'verb': sum(wf.get("verb", 0) for wf in word_features_list),
        'adj': sum(wf.get("adj", 0) for wf in word_features_list),
        'akho_abo': sum(wf.get("akho_abo", 0) for wf in word_features_list),
        'ordinal_number': sum(wf.get("ordinal_number", 0) for wf in word_features_list),
        'symbol': sum(wf.get("symbol", 0) for wf in word_features_list),
        'interrogative_tools': sum(wf.get("interrogative_tools", 0) for wf in word_features_list),
        'imperfective_singular': sum(wf.get("imperfective_singular", 0) for wf in word_features_list),
        'prc_Al_det': sum(wf.get("prc_Al_det", 0) for wf in word_features_list),
        'suf_1s_pron': sum(wf.get("suf_1s_pron", 0) for wf in word_features_list),
        'prc_waw': sum(wf.get("prc_waw", 0) for wf in word_features_list),
        'verb_present_plural': sum(wf.get("verb_present_plural", 0) for wf in word_features_list),
        'prc_prep': sum(wf.get("prc_prep", 0) for wf in word_features_list),
        'suf_pron': sum(wf.get("suf_pron", 0) for wf in word_features_list),
        'dual_noun_adj': sum(wf.get("dual_noun_adj", 0) for wf in word_features_list),
        'plural_fem_noun_adj': sum(wf.get("plural_fem_noun_adj", 0) for wf in word_features_list),
        'verb_past_s_p': sum(wf.get("verb_past_s_p", 0) for wf in word_features_list),
        'plural_masc': sum(wf.get("plural_masc", 0) for wf in word_features_list),
        'verb_past_present_dual': sum(wf.get("verb_past_present_dual", 0) for wf in word_features_list),
        'verb_command': sum(wf.get("verb_command", 0) for wf in word_features_list),
        'suf_dual_pron': sum(wf.get("suf_dual_pron", 0) for wf in word_features_list),
        'broken_plural': sum(wf.get("broken_plural", 0) for wf in word_features_list),
        'waw_alqassam': sum(wf.get("waw_alqassam", 0) for wf in word_features_list),
        'verb_command_plural': sum(wf.get("verb_command_plural", 0) for wf in word_features_list),
        'amma_lakin': sum(wf.get("amma_lakin", 0) for wf in word_features_list),
        'verb_command_dual': sum(wf.get("verb_command_dual", 0) for wf in word_features_list),
        'interrogative_alif': sum(wf.get("interrogative_alif", 0) for wf in word_features_list),
        'ba_alqassam': sum(wf.get("ba_alqassam", 0) for wf in word_features_list),
        'passive_voice': sum(wf.get("passive_voice", 0) for wf in word_features_list),
        'ta_alqassam': sum(wf.get("ta_alqassam", 0) for wf in word_features_list),
        'pronoun': sum(wf.get("pronoun", 0) for wf in word_features_list),
        'proper_noun': sum(wf.get("proper_noun", 0) for wf in word_features_list),
        'weak_final': sum(wf.get("weak_final", 0) for wf in word_features_list),
        'other_proclitics': sum(wf.get("other_proclitics", 0) for wf in word_features_list)
    }

    # Combine sentence and word-level features and id
    features = { **sentence_features, **aggregated_word_features}

    return features, pairs

def load_existing_ids(output_file):
    """Load existing IDs from the output file to avoid duplication."""
    try:
        existing_data = pd.read_csv(output_file, usecols=['ID'])
        existing_ids = set(existing_data['ID'].tolist())
        return existing_ids
    except FileNotFoundError:
        # If the file does not exist, return an empty set
        return set()

# Main reusable function
def extract_disambig_feats(df):
    """
    Extract disambiguation features for a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with at least 'Sentence' and 'ID' columns.

    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    features = df.progress_apply(extract_features_from_row, axis=1)
    features_df = pd.DataFrame(features.tolist())
    return features_df
