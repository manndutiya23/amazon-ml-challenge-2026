import pandas as pd
from collections import defaultdict, Counter
from blocking import prepare_record


PURGE_FRAC = 0.0005

STRATEGIES = (
    "name_token_pairs",
    "address_digits",
    "country_address_first",
    "name_first_token",
)


def build_streaming_index(path, chunksize=100_000, nrows=None):
    # First pass: count rows + determine which purge keys are too common
    counts = {
        "country_address_first": Counter(),
        "name_first_token": Counter(),
    }

    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=chunksize,
        nrows=nrows,
        keep_default_na=False,
    ):
        total_rows += len(chunk)

        for row in chunk.itertuples(index=False):
            rec = prepare_record(
                row.business_name,
                row.business_address,
                row.country,
            )

            key = rec["key_country_address_first"]
            if key:
                counts["country_address_first"][key] += 1

            key = rec["key_name_first_token"]
            if key:
                counts["name_first_token"][key] += 1

    cap = max(1, int(total_rows * PURGE_FRAC))

    purged = {
        name: {key for key, count in counter.items() if count > cap}
        for name, counter in counts.items()
    }

    print(f"Rows: {total_rows:,}")
    print(f"Purge cap: {cap:,}")
    print(
        "Purged country_address_first keys:",
        len(purged["country_address_first"]),
    )
    print(
        "Purged name_first_token keys:",
        len(purged["name_first_token"]),
    )

    # Second pass: build compact indexes, skipping purged keys
    indexes = {
        "name_token_pairs": defaultdict(list),
        "address_digits": defaultdict(list),
        "country_address_first": defaultdict(list),
        "name_first_token": defaultdict(list),
    }

    rows_seen = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=chunksize,
        nrows=nrows,
        keep_default_na=False,
    ):
        for row in chunk.itertuples(index=False):
            rec = prepare_record(
                row.business_name,
                row.business_address,
                row.country,
            )

            eid = row.entity_id

            for key in rec["keys_name_token_pairs"]:
                indexes["name_token_pairs"][key].append(eid)

            for key in rec["keys_address_digits"]:
                indexes["address_digits"][key].append(eid)

            key = rec["key_country_address_first"]
            if key and key not in purged["country_address_first"]:
                indexes["country_address_first"][key].append(eid)

            key = rec["key_name_first_token"]
            if key and key not in purged["name_first_token"]:
                indexes["name_first_token"][key].append(eid)

        rows_seen += len(chunk)
        print(f"Indexed {rows_seen:,} rows...", flush=True)

    return indexes


def query_index(source1_df, indexes):
    result = {}

    for row in source1_df.itertuples(index=False):
        rec = prepare_record(
            row.business_name,
            row.business_address,
            row.country,
        )

        candidates = set()

        for key in rec["keys_name_token_pairs"]:
            candidates.update(indexes["name_token_pairs"].get(key, []))

        for key in rec["keys_address_digits"]:
            candidates.update(indexes["address_digits"].get(key, []))

        key = rec["key_country_address_first"]
        if key:
            candidates.update(
                indexes["country_address_first"].get(key, [])
            )

        key = rec["key_name_first_token"]
        if key:
            candidates.update(
                indexes["name_first_token"].get(key, [])
            )

        result[row.entity_id] = sorted(candidates)

    return result


def save_index(indexes, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(dict(indexes), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
