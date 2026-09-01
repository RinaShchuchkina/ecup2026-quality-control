import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
_WS_RE = re.compile(r"\s\s+")

RULES = [
    ("r_bad_phrase", r"биологически активн"),
    ("r_bad_word", r"\bбад\b"),
    ("r_not_bad", r"не является (бад|биологически)"),
    ("r_sgr", r"\bсгр\b|свидетельство о гос"),
    ("r_sport", r"спортивн\w+ питан|протеин|\bbcaa\b|гейнер|креатин|аминокислот|изолят|казеин|предтрен"),
    ("r_pharm", r"не является лекарств"),
    ("r_no_fuel", r"баллон[^.]{0,40}не входит|без (газового )?баллона|без топлива|не заправлен|топливо не входит"),
    ("r_selfignite", r"т[её]рка для поджига|спичка внутри|самоподжиг|со спичкой"),
    ("r_pyro", r"пиротехнич|салют|фейерверк|бенгальск|дымовая шашка|цветной дым"),
    ("r_pneumo", r"пневмат|сжатый воздух|не пиротехника|не является пиротехник"),
    ("r_electric", r"\busb\b|электроимпульс|электронн\w+ зажигалк|дугов\w+ зажигалк"),
    ("r_fuel_kit", r"с газом|заправлен|в комплекте с угл[её]м|уголь в комплекте|с сухим горючим"),
    ("r_gas_refill", r"газ (бутан|для заправки)|мапп|\bmapp\b"),
    ("r_flam_mark", r"огнеопасн|легковоспламен"),
]
_RULES_COMPILED = [(name, re.compile(pat)) for name, pat in RULES]


def word_ngrams(doc: str):
    tokens = _TOKEN_RE.findall(doc.lower())
    out = list(tokens)
    out.extend(" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1))
    return out


def word_unigrams(doc: str):
    return _TOKEN_RE.findall(doc.lower())


def char_wb_ngrams(doc: str):
    doc = _WS_RE.sub(" ", doc.lower())
    ngrams = []
    for w in doc.split():
        w = " " + w + " "
        w_len = len(w)
        for n in range(3, 6):
            offset = 0
            ngrams.append(w[offset:offset + n])
            while offset + n < w_len:
                offset += 1
                ngrams.append(w[offset:offset + n])
            if offset == 0:
                break
    return ngrams


def rule_flags(doc: str) -> np.ndarray:
    low = doc.lower()
    return np.array([1.0 if rx.search(low) else 0.0 for _, rx in _RULES_COMPILED])


def transform_tfidf(docs, vocab: dict, idf: np.ndarray, analyzer) -> csr_matrix:
    indptr, indices, data = [0], [], []
    for doc in docs:
        cnt = Counter(t for t in analyzer(doc) if t in vocab)
        if cnt:
            cols, vals = zip(*((vocab[t], 1.0 + np.log(c)) for t, c in cnt.items()))
            vals = np.asarray(vals) * idf[list(cols)]
            norm = np.sqrt((vals ** 2).sum())
            if norm > 0:
                vals = vals / norm
            indices.extend(cols)
            data.extend(vals)
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), shape=(len(docs), len(idf)))
