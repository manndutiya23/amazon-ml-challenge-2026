import pandas as pd


def load_ground_truth(
    ground_truth_path,
    source2_ids,
    source3_ids,
    chunksize=100_000,
):
    """
    Load ground truth and classify Source 1 entities into:

    safe_ground_truth:
        Every GT match for the S1 entity exists in the supplied
        training Source 2 / Source 3 records.

    unsafe_s1_ids:
        At least one GT match references an unavailable S2/S3 record.

    Empty GT lists are considered safe singletons.
    """

    valid_ids = set(source2_ids)
    valid_ids.update(source3_ids)

    safe_ground_truth = {}
    unsafe_s1_ids = set()

    for chunk in pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str,
        chunksize=chunksize,
        keep_default_na=False,
    ):
        for _, row in chunk.iterrows():

            s1_id = row["source1_entity_id"]
            raw_matches = row["matched_entity_ids"]

            if not raw_matches:
                safe_ground_truth[s1_id] = []
                continue

            matches = [
                x.strip()
                for x in raw_matches.split(",")
                if x.strip()
            ]

            unavailable = [
                x for x in matches
                if x not in valid_ids
            ]

            if unavailable:
                unsafe_s1_ids.add(s1_id)
                safe_ground_truth.pop(s1_id, None)
                continue

            matches = list(dict.fromkeys(matches))

            safe_ground_truth[s1_id] = matches

    return safe_ground_truth, unsafe_s1_ids