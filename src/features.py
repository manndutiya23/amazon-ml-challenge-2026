from collections import Counter
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

    a_grams = set(a[i:i+n] for i in range(len(a) - n + 1))
    b_grams = set(b[i:i+n] for i in range(len(b) - n + 1))

    if not a_grams or not b_grams:
        return 0.0

    return len(a_grams & b_grams) / len(a_grams | b_grams)


def digit_tokens(text):
    return set(re.findall(r"\d+", text or ""))


def feature_pair(row1, row2):
    name1 = row1["name_norm"]
    name2 = row2["name_norm"]

    addr1 = row1["address_norm"]
    addr2 = row2["address_norm"]

    digits1 = digit_tokens(addr1)
    digits2 = digit_tokens(addr2)

    return {
        "name_ratio": ratio(name1, name2) / 100.0,
        "name_token_set": token_set_ratio(name1, name2) / 100.0,
        "name_token_sort": token_sort_ratio(name1, name2) / 100.0,
        "name_jaccard": jaccard_tokens(name1, name2),
        "name_char_jaccard": char_jaccard(name1, name2),

        "address_ratio": ratio(addr1, addr2) / 100.0,
        "address_token_set": token_set_ratio(addr1, addr2) / 100.0,
        "address_token_sort": token_sort_ratio(addr1, addr2) / 100.0,
        "address_jaccard": jaccard_tokens(addr1, addr2),
        "address_char_jaccard": char_jaccard(addr1, addr2),

        "digit_overlap": (
            len(digits1 & digits2) / len(digits1 | digits2)
            if digits1 or digits2 else 0.0
        ),

        "country_same": int(
            row1.get("country", "") != ""
            and row1.get("country", "") == row2.get("country", "")
        ),

        "name_exact": int(name1 != "" and name1 == name2),
        "address_exact": int(addr1 != "" and addr1 == addr2),
    }