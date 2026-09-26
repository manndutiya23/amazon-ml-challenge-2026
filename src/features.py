import re

from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio


def jaccard_tokens(a, b):
    a_tokens = set(a.split()) if a else set()
    b_tokens = set(b.split()) if b else set()

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def char_jaccard(a, b, n=3):
    if not a or not b:
        return 0.0

    if len(a) < n or len(b) < n:
        return 1.0 if a == b else 0.0

    a_grams = set(
        a[i:i + n]
        for i in range(len(a) - n + 1)
    )

    b_grams = set(
        b[i:i + n]
        for i in range(len(b) - n + 1)
    )

    if not a_grams or not b_grams:
        return 0.0

    return len(a_grams & b_grams) / len(a_grams | b_grams)


def digit_tokens(text):
    return set(re.findall(r"\d+", text or ""))


def length_difference(a, b):
    return abs(len(a or "") - len(b or ""))


def length_ratio(a, b):
    len_a = len(a or "")
    len_b = len(b or "")

    if len_a == 0 and len_b == 0:
        return 1.0

    if len_a == 0 or len_b == 0:
        return 0.0

    return min(len_a, len_b) / max(len_a, len_b)


def feature_pair(row1, row2):
    """
    Generate matching features for one candidate pair.

    Expected fields:
        name_norm
        address_norm
        country
    """

    name1 = row1.get("name_norm", "")
    name2 = row2.get("name_norm", "")

    addr1 = row1.get("address_norm", "")
    addr2 = row2.get("address_norm", "")

    country1 = str(row1.get("country", "") or "").strip().lower()
    country2 = str(row2.get("country", "") or "").strip().lower()

    digits1 = digit_tokens(addr1)
    digits2 = digit_tokens(addr2)

    if digits1 or digits2:
        digit_overlap = (
            len(digits1 & digits2)
            / len(digits1 | digits2)
        )
    else:
        digit_overlap = 0.0

    return {
        # -------------------------
        # Name features
        # -------------------------
        "name_exact": int(
            bool(name1) and name1 == name2
        ),

        "name_ratio": ratio(
            name1, name2
        ) / 100.0,

        "name_token_set": token_set_ratio(
            name1, name2
        ) / 100.0,

        "name_token_sort": token_sort_ratio(
            name1, name2
        ) / 100.0,

        "name_jaccard": jaccard_tokens(
            name1, name2
        ),

        "name_char_jaccard": char_jaccard(
            name1, name2
        ),

        "name_length_difference": length_difference(
            name1, name2
        ),

        "name_length_ratio": length_ratio(
            name1, name2
        ),

        # -------------------------
        # Address features
        # -------------------------
        "address_exact": int(
            bool(addr1) and addr1 == addr2
        ),

        "address_ratio": ratio(
            addr1, addr2
        ) / 100.0,

        "address_token_set": token_set_ratio(
            addr1, addr2
        ) / 100.0,

        "address_token_sort": token_sort_ratio(
            addr1, addr2
        ) / 100.0,

        "address_jaccard": jaccard_tokens(
            addr1, addr2
        ),

        "address_char_jaccard": char_jaccard(
            addr1, addr2
        ),

        "address_length_difference": length_difference(
            addr1, addr2
        ),

        "address_length_ratio": length_ratio(
            addr1, addr2
        ),

        # -------------------------
        # Address number signal
        # -------------------------
        "digit_overlap": digit_overlap,

        # -------------------------
        # Country
        # -------------------------
        "country_same": int(
            bool(country1)
            and bool(country2)
            and country1 == country2
        ),

        # -------------------------
        # Strong combined signal
        # -------------------------
        "both_exact": int(
            bool(name1)
            and bool(addr1)
            and name1 == name2
            and addr1 == addr2
        ),
    }