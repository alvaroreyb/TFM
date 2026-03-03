"""
lexical_analysis.py

Reads all .txt files from INPUT_DIR (recursive) and computes corpus-level lexical statistics, exporting a single-row CSV:
- total_documents, unique_words, total_words, total_sentences
- mean_sentence_length_words, mean_word_length_chars
- non_stop_total, unique_non_stop
- non_stop_no_num_total, unique_non_stop_no_num
- top_non_stop_words
- top_abbreviations
- negations_per_1k_words
- misspelling_rate_pct + misspelled_words:
  Uses wordfreq zipf_frequency as a general-language signal plus a domain-frequency safeguard:
  A word is treated as correct if:
    - zipf_frequency(word, "es") >= WORDFREQ_ZIPF_THRESHOLD, OR
    - corpus frequency >= DOMAIN_FREQ_MIN (domain terminology), OR
    - word in USER_DICTIONARY_PATH
  Only alphabetic, lowercase tokens, length >= MISSPELL_MIN_LEN, non-stopwords, non-abbreviations are considered.
- lexical_diversity (as requested): unique_words / len(filtered_words_nonstop_nonnum)

Dependencies:
- nltk
- wordfreq

Edit INPUT_DIR / OUTPUT_CSV / USER_DICTIONARY_PATH before running.
"""

import os
import re
import csv
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from wordfreq import zipf_frequency


INPUT_DIR = ""
OUTPUT_CSV = ""
USER_DICTIONARY_PATH = r""

TOP_N_WORDS = 30
TOP_N_ABBREVIATIONS = 30
MAX_MISSPELLED_WORDS_IN_CSV = 250

MISSPELL_MIN_LEN = 4
WORDFREQ_ZIPF_THRESHOLD = 2
DOMAIN_FREQ_MIN = 3

COMMON_NEGATION_CUES = [
    "no", "nunca", "nadie", "nada", "ninguno", "ninguna", "jamás", "ni", "tampoco", "sin",
    "siquiera", "ni siquiera", "no hay", "no tiene", "no existe"
]

NOT_ABBREVIATIONS = [
    "abi", "ago", "agua", "al", "algo", "alta", "alto", "anal", "and", "ano", "ante", "arco", "area",
    "as", "asas", "asi", "asma", "así", "ayer", "año", "años", "aún", "baja", "bajo", "base", "bazo",
    "baño", "bien", "boca", "bote", "buen", "cabe", "cada", "cama", "capa", "cara", "card", "care",
    "casi", "caso", "cava", "cede", "cel", "cell", "cena", "cita", "cito", "cola", "coli", "como",
    "comp", "con", "cono", "cook", "cree", "creo", "cruz", "cual", "cuya", "cuyo", "cuña", "da", "dada",
    "dado", "dar", "data", "days", "de", "debe", "dedo", "deja", "dejo", "del", "dia", "dias", "dic",
    "dice", "diez", "dos", "doy", "dres", "duda", "dura", "duro", "día", "días", "eco", "edad", "eje",
    "ejes", "el", "ella", "ello", "emma", "en", "es", "esa", "ese", "est", "esta", "este", "está", "esté",
    "exon", "exón", "fase", "feb", "fin", "foco", "for", "fosa", "frio", "fría", "frío", "fue", "fuga",
    "gain", "gas", "gen", "gene", "gran", "guía", "ha", "hace", "haga", "hago", "halo", "han", "hay",
    "hija", "hijo", "hoja", "hora", "hoy", "hoz", "háb", "ia", "ii", "iia", "iib", "iic", "iii", "iiia",
    "iiib", "iiic", "in", "ir", "iv", "iva", "ivb", "ivc", "juan", "jul", "jun", "ki", "kit", "la", "lado",
    "lago", "las", "le", "lee", "leve", "liga", "lo", "los", "loss", "lung", "luz", "mal", "mala", "mama",
    "mar", "mas", "masa", "may", "mayo", "me", "mes", "mesa", "mide", "modo", "moec", "must", "muy", "más",
    "nada", "neo", "ni", "nice", "no", "nos", "nota", "nov", "nula", "oct", "of", "ojal", "ok", "on",
    "once", "or", "oral", "osea", "oseo", "otra", "otro", "pala", "para", "paso", "pedi", "peor", "pero",
    "pese", "peso", "pico", "pido", "pies", "pio", "plan", "poco", "polo", "por", "port", "que", "raiz",
    "rama", "raíz", "real", "rita", "roce", "roja", "roux", "sabe", "saco", "sale", "san", "sana", "se",
    "sea", "seca", "seco", "seno", "sept", "ser", "será", "si", "sido", "sin", "sino", "site", "situ", "solo",
    "son", "stop", "su", "sus", "sí", "sólo", "test", "tipo", "toda", "todo", "toma", "top", "tos", "tras",
    "tres", "trás", "tubo", "tuvo", "type", "tía", "tío", "un", "una", "unas", "uno", "unos", "urea", "usa",
    "usar", "uso", "va", "van", "ve", "ven", "vena", "veo", "ver", "vez", "via", "vial", "vida", "vii", "vino",
    "viva", "vive", "vía", "vías", "wild", "with", "ya", "you", "zona", "área", "él", "éste", "ósea", "óseo"
]

SPANISH_CHARS = "a-zA-ZñÑáéíóúÁÉÍÓÚüÜ"
ABBREVIATION_PATTERN = re.compile(
    r"(?<![\w-])([" + SPANISH_CHARS + r"]{1,2}(?:\.[" + SPANISH_CHARS + r"]{1,2})*[" + SPANISH_CHARS + r"]{1,2})(?![\w-])"
)
WORD_PATTERN = re.compile(rf"[{SPANISH_CHARS}0-9]+(?:[.,][{SPANISH_CHARS}0-9]+)*")
NUMERIC_PATTERN = re.compile(r"^\d+(?:[.,]\d+)*$")


def analyze_and_save(input_dir: str, out_csv: str) -> None:
    """
    *Compute corpus-level lexical metrics from .txt files and write a one-row CSV.*
    Arguments: input_dir, out_csv
    Return: None
    """
    for pkg, res in (("punkt", "tokenizers/punkt"), ("stopwords", "corpora/stopwords")):
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(pkg, quiet=True)

    def strip_accents(text: str) -> str:
        """
        *Removes diacritics from a string.*
        Arguments: text
        Return: text without accents
        """
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    sw = set(stopwords.words("spanish"))
    not_abbrev = set(w.lower() for w in NOT_ABBREVIATIONS)

    user_dict = set()
    if USER_DICTIONARY_PATH and os.path.isfile(USER_DICTIONARY_PATH):
        with open(USER_DICTIONARY_PATH, "r", encoding="utf-8", errors="ignore") as f:
            user_dict = set(line.strip().lower() for line in f if line.strip())

    texts: List[str] = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                path = os.path.join(root, fn)
                for enc in ("utf-8", "utf-8-sig", "latin-1"):
                    try:
                        with open(path, "r", encoding=enc, errors="strict") as f:
                            texts.append(f.read())
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        texts.append(f.read())

    total_documents = len(texts)
    full_text = "\n".join(texts)
    full_text_lower = full_text.lower()

    sentences: List[str] = []
    for t in texts:
        sentences.extend([s.strip() for s in sent_tokenize(t, language="spanish") if s.strip()])
    total_sentences = len(sentences)

    tokens_raw = WORD_PATTERN.findall(full_text)
    tokens_norm = [t.lower() for t in tokens_raw]
    total_words = len(tokens_norm)
    total_unique_words = len(set(tokens_norm))

    token_counter = Counter()
    for doc in texts:
        for w in word_tokenize(doc, language="spanish"):
            if w.isalpha():
                token_counter[w.lower()] += 1

    mean_word_length = (sum(len(t) for t in tokens_norm) / total_words) if total_words else 0.0
    sent_lengths = [len(WORD_PATTERN.findall(s)) for s in sentences]
    mean_sentence_length = (sum(sent_lengths) / len(sent_lengths)) if sent_lengths else 0.0

    non_stop = [t for t in tokens_norm if t.isalpha() and t not in sw]
    non_stop_total = len(non_stop)
    unique_non_stop = len(set(non_stop))

    filtered_words_nonstop_nonnum = [
        t for t in tokens_norm
        if (t not in sw) and (not NUMERIC_PATTERN.match(t)) and any(ch.isalpha() for ch in t)
    ]
    non_stop_no_num_total = len(filtered_words_nonstop_nonnum)
    unique_non_stop_no_num = len(set(filtered_words_nonstop_nonnum))

    top_non_stop_words = Counter(filtered_words_nonstop_nonnum).most_common(TOP_N_WORDS)

    abbr_counter = Counter()
    abbr_norm_set = set()
    for m in ABBREVIATION_PATTERN.finditer(full_text):
        abbr = m.group(1)
        abbr_norm = abbr.lower()
        if len(abbr_norm) == 1:
            continue
        if abbr_norm in not_abbrev:
            continue
        abbr_counter[abbr] += 1
        abbr_norm_set.add(abbr_norm)
        abbr_norm_set.add(abbr_norm.replace(".", ""))
    top_abbreviations = abbr_counter.most_common(TOP_N_ABBREVIATIONS)

    neg_count = 0
    for cue in COMMON_NEGATION_CUES:
        cue_esc = re.escape(cue.lower())
        cue_pat = r"\b" + cue_esc.replace(r"\ ", r"\s+") + r"\b"
        neg_count += len(re.findall(cue_pat, full_text_lower, flags=re.IGNORECASE))
    negations_per_1k_words = (neg_count / total_words * 1000.0) if total_words else 0.0

    misspell_candidates_total = 0
    miss_counts = Counter()

    for doc in texts:
        doc_tokens = word_tokenize(doc, language="spanish")
        for w in doc_tokens:
            if not w.isalpha():
                continue

            wl = w.lower()
            if len(wl) < MISSPELL_MIN_LEN:
                continue
            if wl in sw:
                continue
            if wl in user_dict:
                continue
            if wl in abbr_norm_set:
                continue

            misspell_candidates_total += 1

            if token_counter[wl] >= DOMAIN_FREQ_MIN:
                continue

            if zipf_frequency(wl, "es") >= WORDFREQ_ZIPF_THRESHOLD:
                continue

            wna = strip_accents(wl)
            if wna != wl and zipf_frequency(wna, "es") >= WORDFREQ_ZIPF_THRESHOLD:
                continue

            miss_counts[wl] += 1

    misspelled_total = sum(miss_counts.values())
    misspelling_rate_pct = (misspelled_total / misspell_candidates_total) * 100.0 if misspell_candidates_total else 0.0
    misspelled_words: List[Tuple[str, int]] = miss_counts.most_common(MAX_MISSPELLED_WORDS_IN_CSV)

    lexical_diversity = (
        total_unique_words / len(filtered_words_nonstop_nonnum)
        if len(filtered_words_nonstop_nonnum) > 0
        else 0.0
    )

    row: Dict[str, object] = {
        "total_documents": total_documents,
        "unique_words": total_unique_words,
        "total_words": total_words,
        "total_sentences": total_sentences,
        "mean_sentence_length_words": round(mean_sentence_length, 6),
        "mean_word_length_chars": round(mean_word_length, 6),
        "non_stop_total": non_stop_total,
        "unique_non_stop": unique_non_stop,
        "non_stop_no_num_total": non_stop_no_num_total,
        "unique_non_stop_no_num": unique_non_stop_no_num,
        "top_non_stop_words": "; ".join([f"{w}:{c}" for w, c in top_non_stop_words]),
        "top_abbreviations": "; ".join([f"{w}:{c}" for w, c in top_abbreviations]),
        "negations_per_1k_words": round(negations_per_1k_words, 6),
        "misspelling_rate_pct": round(float(misspelling_rate_pct), 6),
        "misspelled_words": "; ".join([f"{w}:{c}" for w, c in misspelled_words]) if misspelled_words else "",
        "misspell_min_len": MISSPELL_MIN_LEN,
        "misspell_candidates_total": misspell_candidates_total,
        "wordfreq_zipf_threshold": WORDFREQ_ZIPF_THRESHOLD,
        "domain_freq_min": DOMAIN_FREQ_MIN,
        "lexical_diversity": round(float(lexical_diversity), 6),
    }

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(f"Documents: {total_documents}")
    print(f"Total words: {total_words} | Unique words: {total_unique_words}")
    print(f"Total sentences: {total_sentences}")
    print(f"Misspell candidates total: {misspell_candidates_total}")
    print(f"Misspelling rate (%): {misspelling_rate_pct}")
    print(f"Output CSV: {out_csv}")


if __name__ == "__main__":
    analyze_and_save(INPUT_DIR, OUTPUT_CSV)
