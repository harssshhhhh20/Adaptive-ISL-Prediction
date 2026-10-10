"""Gloss normalisation for ISL-CSLTR.

The corpus ships one gloss string per sentence (``ISL Corpus sign glosses.csv``). It contains
annotation inconsistencies that would otherwise inflate WER with label noise. Every rule below
is a deliberate, documented choice; ``normalise(..., raw=True)`` keeps only the minimal cleaning
(case, punctuation, whitespace) so results can be re-scored under the raw labels as a check.
"""
import csv
import re
from pathlib import Path

RAW_META = Path(__file__).resolve().parents[1] / "data" / "raw_meta"

# Merge spellings of the same sign. Negation is written DONOT / DONT / DO NOT / NOT across
# sentences; ISL uses one negation sign, so all four map to NOT.
_PHRASE_RULES = [
    (r"\bDO NOT\b", "NOT"),
    (r"\bDONOT\b", "NOT"),
    (r"\bDONT\b", "NOT"),
    (r"\bCONGRATULATIIONS\b", "CONGRATULATIONS"),
    (r"\(AGE\)", "AGE"),
    (r"\bXXXXXXXX\b", "NAME"),
]


# Frame-folder spellings that differ from the gloss table (table spelling -> folder spelling).
_FOLDER_SPELLING = {"which college school are you from": "which collegeschool are you from"}


def sentence_key(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def normalise(gloss, raw=False):
    g = gloss.upper()
    if not raw:
        for pat, rep in _PHRASE_RULES:
            g = re.sub(pat, rep, g)
    g = re.sub(r"[^A-Z0-9 ]", " ", g)
    return g.split()


FRAMES = RAW_META.parent / "frames"
# Sentences listed in the gloss table that have no frame sequences in the ISL-CSLTR release
# (verified when the frames were downloaded); pinned so results do not depend on data/frames.
NO_FRAMES = {"he would be coming today", "i dont agree", "no need to worry dont worry",
             "now onwards he will never hurt you"}


def load_gloss_table(raw=False, with_data_only=True):
    """Sentence -> gloss list. By default only the sentences that have frame sequences in the
    release (97 of 101); the other 4 would add output classes that can never be observed."""
    with open(RAW_META / "ISL Corpus sign glosses.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    table = {sentence_key(r["Sentence"]): normalise(r["SIGN GLOSSES"], raw) for r in rows}
    table = {_FOLDER_SPELLING.get(k, k): v for k, v in table.items()}
    if with_data_only:
        table = {k: v for k, v in table.items() if k not in NO_FRAMES}
    return table
