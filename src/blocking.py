"""
Blocking / candidate generation for Source 1 -> Source 2 / Source 3 entity resolution.

A blocking strategy maps each record to zero or more block keys. Two records are
candidates for matching if they share at least one key in a given strategy, or -
for a "union" strategy - if they share a key in *any* of several underlying
strategies.

This module keeps the "single-key" (one key per record, e.g. exact normalized name)
and "multi-key" (many keys per record, e.g. one per significant name token) cases
under one small framework so new strategies are one function + one registry entry.
"""

import re
from collections import defaultdict

from preprocessing import normalize_name, normalize_address

LEGAL_SUFFIX_STOPWORDS = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "lp", "pvt", "private", "ltda",
    "gmbh", "plc", "group", "holdings", "enterprises", "enterprise",
    "international", "intl", "services", "service", "solutions",
    "the", "and", "of",
}

ADDRESS_STOPWORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "drive", "dr",
    "lane", "ln", "boulevard", "blvd", "near", "opp", "opposite",
    "no", "floor", "flr", "suite", "ste", "block", "sector", "phase",
    "colony", "nagar", "marg", "circle", "cross", "layout", "extension",
    "ext", "post", "office", "po", "pin", "zip", "code",
    "india", "usa", "united", "states", "america",
}

MIN_TOKEN_LEN = 2
MIN_DIGIT_TOKEN_LEN = 3


def significant_name_tokens(name_norm):
    """Tokens of a normalized business name, minus legal-suffix noise words."""
    if not name_norm:
        return []
    tokens = [
        t for t in name_norm.split()
        if t not in LEGAL_SUFFIX_STOPWORDS and len(t) >= MIN_TOKEN_LEN
    ]
    if tokens:
        return tokens
    # Fall back to the raw tokens if stripping stopwords left nothing
    # (e.g. the name IS just "Inc" or a single short word).
    return name_norm.split()


def significant_address_tokens(address_norm):
    """Non-numeric address tokens, minus generic address noise words."""
    if not address_norm:
        return []
    return [
        t for t in address_norm.split()
        if t not in ADDRESS_STOPWORDS and not t.isdigit() and len(t) >= MIN_TOKEN_LEN
    ]


def address_digit_tokens(address_norm, min_len=MIN_DIGIT_TOKEN_LEN):
    """Digit runs (street numbers, PIN/ZIP codes) long enough to be distinctive."""
    if not address_norm:
        return []
    return [t for t in re.findall(r"\d+", address_norm) if len(t) >= min_len]


def name_token_bigrams(name_tokens):
    """
    Sorted-pair combinations of significant name tokens.

    Requiring two shared tokens (instead of one) is far more selective than
    single-token blocking - it survives word reordering and one noisy/missing
    token, while avoiding the huge blocks a single common word creates.
    Falls back to single tokens when there aren't two to pair.
    """
    if len(name_tokens) < 2:
        return list(name_tokens)
    pairs = []
    for i in range(len(name_tokens)):
        for j in range(i + 1, len(name_tokens)):
            a, b = name_tokens[i], name_tokens[j]
            pairs.append(f"{a}_{b}" if a <= b else f"{b}_{a}")
    return pairs


def prepare_record(business_name, business_address, country):
    """
    Compute every normalized field and block-key a strategy might need for one
    record. Returns a dict; callers add ``entity_id`` themselves.
    """
    name_norm = normalize_name(business_name)
    address_norm = normalize_address(business_address)
    country_norm = (country or "").strip().lower()

    name_tokens = significant_name_tokens(name_norm)
    address_tokens = significant_address_tokens(address_norm)
    digit_tokens = address_digit_tokens(address_norm)

    name_first = name_tokens[0] if name_tokens else ""
    address_first = address_tokens[0] if address_tokens else ""

    return {
        "name_norm": name_norm,
        "address_norm": address_norm,
        "country_norm": country_norm,
        "name_tokens": name_tokens,
        "address_tokens": address_tokens,
        "digit_tokens": digit_tokens,
        # single-key strategies
        "key_exact_name": name_norm,
        "key_name_first_token": name_first,
        "key_address_first_token": address_first,
        "key_country_name_first": (
            f"{country_norm}|{name_first}" if country_norm and name_first else ""
        ),
        "key_country_address_first": (
            f"{country_norm}|{address_first}" if country_norm and address_first else ""
        ),
        # multi-key strategies
        "keys_name_tokens": name_tokens,
        "keys_name_token_pairs": name_token_bigrams(name_tokens),
        "keys_address_digits": digit_tokens,
        "keys_name_first_address_digit": (
            [f"{name_first}|{d}" for d in digit_tokens] if name_first else []
        ),
    }


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def build_index(records, key_field, multi=False, max_block_frac=None):
    """
    records: iterable of dicts, each with ``entity_id`` plus the fields from
             ``prepare_record``.
    key_field: which field to block on (single string or list-of-strings field).
    max_block_frac: if set, block purging - any key whose block would contain
        more than ``max_block_frac`` of all records is dropped entirely. Those
        keys are too generic (common word, common digit run) to discriminate
        and just inflate candidate volume without adding recall.
    Returns dict[key] -> list[entity_id].
    """
    records = list(records)
    index = defaultdict(list)
    if multi:
        for rec in records:
            eid = rec["entity_id"]
            for key in rec[key_field]:
                if key:
                    index[key].append(eid)
    else:
        for rec in records:
            key = rec[key_field]
            if key:
                index[key].append(rec["entity_id"])

    if max_block_frac is not None:
        cap = max(1, int(len(records) * max_block_frac))
        index = {k: v for k, v in index.items() if len(v) <= cap}

    return dict(index)


# Registry: strategy name -> (key_field, multi, max_block_frac)
# max_block_frac purges any block key that would match more than that fraction
# of the indexed corpus - e.g. 0.0005 drops keys shared by >0.05% of records.
STRATEGIES = {
    "exact_name": ("key_exact_name", False, None),
    "name_first_token": ("key_name_first_token", False, None),
    "name_first_token_purged": ("key_name_first_token", False, 0.0005),
    "name_tokens": ("keys_name_tokens", True, None),
    "name_tokens_purged": ("keys_name_tokens", True, 0.0005),
    "name_token_pairs": ("keys_name_token_pairs", True, None),
    "address_first_token": ("key_address_first_token", False, None),
    "address_first_token_purged": ("key_address_first_token", False, 0.0005),
    "address_digits": ("keys_address_digits", True, None),
    "country_name_first": ("key_country_name_first", False, None),
    "country_name_first_purged": ("key_country_name_first", False, 0.0005),
    "country_address_first": ("key_country_address_first", False, None),
    "country_address_first_purged": ("key_country_address_first", False, 0.0005),
    "name_first_address_digit": ("keys_name_first_address_digit", True, None),
}

# The strategies whose union is used by the production candidate generators.
# Chosen from experiment_results (see notebooks/04_blocking_experiments.ipynb):
# bigram name-token blocking survives word reordering/typos without the
# volume blowup of single-token blocking, address-digit blocking catches
# cases where the name is too noisy but the street number/PIN still lines
# up, purged country+address-first-token is a cheap low-volume top-up, and
# purged name-first-token recovers the remaining cases where only one
# significant name token survived (so no bigram could form).
#
# Measured on a 25%-background sample (25k S1 queries, ~1.3M S2 + 1.35M S3
# background, all true matches guaranteed present): 94.4% candidate recall
# at ~1,945 candidates/S1 (~7,600/S1 extrapolated to the full corpus).
DEFAULT_UNION_STRATEGIES = (
    "name_token_pairs",
    "address_digits",
    "country_address_first_purged",
    "name_first_token_purged",
)


def build_indices(records, strategies=STRATEGIES):
    """Build every requested index in one pass over ``records``."""
    records = list(records)
    return {
        name: build_index(records, key_field, multi, max_block_frac)
        for name, (key_field, multi, max_block_frac) in strategies.items()
    }


def candidates_for_record(record, indices, strategy_names):
    """Union of candidate entity_ids across the given strategies for one query record."""
    out = set()
    for name in strategy_names:
        key_field, multi, _ = STRATEGIES[name]
        index = indices[name]
        if multi:
            for key in record[key_field]:
                if key:
                    out.update(index.get(key, ()))
        else:
            key = record[key_field]
            if key:
                out.update(index.get(key, ()))
    return out


# ---------------------------------------------------------------------------
# DataFrame-facing helpers
# ---------------------------------------------------------------------------

def _rows_to_records(df):
    records = []
    for row in df.itertuples(index=False):
        rec = prepare_record(row.business_name, row.business_address, row.country)
        rec["entity_id"] = row.entity_id
        records.append(rec)
    return records


def _generate_candidates(source1_df, other_df, strategy_names=DEFAULT_UNION_STRATEGIES):
    """
    Shared implementation for generate_candidates_s2 / generate_candidates_s3.

    Returns dict[s1_entity_id] -> sorted list of candidate entity_ids from
    ``other_df`` (Source 2 or Source 3).
    """
    other_records = _rows_to_records(other_df)
    needed_strategies = set(strategy_names)
    needed_keys = {name: STRATEGIES[name] for name in needed_strategies}
    indices = {
        name: build_index(other_records, key_field, multi, max_block_frac)
        for name, (key_field, multi, max_block_frac) in needed_keys.items()
    }

    result = {}
    for row in source1_df.itertuples(index=False):
        rec = prepare_record(row.business_name, row.business_address, row.country)
        candidates = candidates_for_record(rec, indices, strategy_names)
        result[row.entity_id] = sorted(candidates)
    return result


def generate_candidates_s2(source1, source2, strategy_names=DEFAULT_UNION_STRATEGIES):
    """
    source1, source2: DataFrames with columns entity_id, business_name,
    business_address, country (Source 1 and Source 2 respectively).

    Returns dict[s1_entity_id] -> sorted list of candidate S2 entity_ids.
    """
    return _generate_candidates(source1, source2, strategy_names)


def generate_candidates_s3(source1, source3, strategy_names=DEFAULT_UNION_STRATEGIES):
    """
    source1, source3: DataFrames with columns entity_id, business_name,
    business_address, country (Source 1 and Source 3 respectively).

    Returns dict[s1_entity_id] -> sorted list of candidate S3 entity_ids.
    """
    return _generate_candidates(source1, source3, strategy_names)
