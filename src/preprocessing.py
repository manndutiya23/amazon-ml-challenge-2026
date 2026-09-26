import re
import unicodedata


def normalize_text(text):
    """
    General-purpose normalization for business names and addresses.
    Keeps letters/numbers, removes punctuation, collapses whitespace.
    """
    if text is None:
        return ""

    text = str(text).lower().strip()

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Replace common separators with spaces
    text = re.sub(r"[/\-_,.&]+", " ", text)

    # Remove remaining punctuation
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(name):
    return normalize_text(name)


def normalize_address(address):
    return normalize_text(address)


def tokenize(text):
    if not text:
        return []

    return text.split()


def compact(text):
    """Remove spaces for character-level comparison."""
    return re.sub(r"\s+", "", text)