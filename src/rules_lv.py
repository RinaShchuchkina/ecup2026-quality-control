import html
import re

POSITIVE = [
    ("pull_mech", r"(?=.*хлопушк)(?:.*((потян|дёрн|дерн)[а-я]*те? за (кольцо|нить|шнур|вер[ёе]вочк|вер[ёе]вк)|вытяжн\w+ (кольц|нит|шнур)))"),
    ("forester_roll", r"forester[^.]{0,60}супер-?ролл|супер-?ролл[^.]{0,60}forester"),
    ("disposable_grill", r"одноразов\w+ мангал[^.]{0,60}с угл|мангал одноразов\w+[^.]{0,80}с угл"),
]

NEGATIVE = [
    ("eternal_match", r"вечная спичка|вечные спички"),
    ("no_sparks", r"без искр( и дыма)?|не содержит пиротехн"),
    ("science_kit", r"набор (для )?(опытов|экспериментов)"),
]

_PULL_ACTION = re.compile(
    r"((потян|дёрн|дерн)[а-я]*те? за "
    r"(кольцо|нить|шнур|вер[ёе]вочк|вер[ёе]вк)"
    r"|вытяжн\w+ (кольц|нит|шнур))"
)
_POS = [(n, re.compile(p)) for n, p in POSITIVE[1:]]
_NEG = [(n, re.compile(p)) for n, p in NEGATIVE]

_ELECTRIC_PRANK = re.compile(
    r"прикол[^.]{0,25}(?:шокер|зажигалк)|"
    r"шокер[^.]{0,25}зажигалк|"
    r"электрическ\w*[^.]{0,20}зажигалк"
)
_DRY_FUEL = re.compile(r"сух(?:ое|ой)\s+(?:горюч|спирт)")
_DRY_IGNITION = re.compile(
    r"с\s+поджиг|спичк[^.]{0,35}(?:в\s+комплект|внутр|прилаг)|"
    r"терк[^.]{0,35}(?:поджиг|спич)"
)
_PYRO_GRENADE = re.compile(r"граната\s+страйк")
_PYRO_CHARGE = re.compile(r"пироэлемент|петард|корсар")
_LONG_MATCHES = re.compile(
    r"^спички\b.*(?:длительн\w*\s+горен|турист|охотнич|ветро)"
)
_GAS_BUNDLE = re.compile(
    r"горелк[^+]{0,100}\+\s*(?:\d+\s*)?(?:цангов\w*\s*)?"
    r"бал+он\w*(?:\s+с)?\s+газ"
)


def _clean_for_natural(value: str) -> str:
    value = html.unescape(value).replace("ё", "е")
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(value.split())


def _has_pull_mechanism(text_lower: str) -> bool:
    return any("хлопушк" in line and _PULL_ACTION.search(line)
               for line in text_lower.splitlines())


def route(text_lower: str, name_lower=None):
    pos = "pull_mech" if _has_pull_mechanism(text_lower) else next(
        (n for n, rx in _POS if rx.search(text_lower)), None
    )
    neg = next((n for n, rx in _NEG if rx.search(text_lower)), None)
    clean_text = _clean_for_natural(text_lower)
    clean_name = _clean_for_natural(name_lower if name_lower is not None else text_lower)
    raw_name = name_lower if name_lower is not None else text_lower
    if pos is None:
        if _DRY_FUEL.search(clean_text) and _DRY_IGNITION.search(clean_text):
            pos = "dry_fuel_with_ignition"
        elif _PYRO_GRENADE.search(clean_text) and _PYRO_CHARGE.search(clean_text):
            pos = "explicit_pyro_grenade"
        elif _LONG_MATCHES.search(clean_name):
            pos = "long_burning_matches"
        elif _GAS_BUNDLE.search(raw_name):
            pos = "explicit_gas_bundle"
    if neg is None and _ELECTRIC_PRANK.search(clean_text):
        neg = "electric_prank"
    if neg and not pos:
        return 0, neg
    if pos and not neg:
        return 1, pos
    return None, None
