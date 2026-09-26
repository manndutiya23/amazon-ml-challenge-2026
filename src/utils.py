from pathlib import Path
import pandas as pd


def load_ground_truth(
    ground_truth_path,
    source2_ids=None,
    source3_ids=None,
    chunksize=100_000,
):
    """
    Load ground truth while keeping only match IDs that exist
    in the supplied Source 2 / Source 3 datasets.

    Ground-truth references to records outside the supplied
    datasets are ignored because those records cannot be used
    for pairwise training.
    """

    valid_ids = None

    if source2_ids is not None or source3_ids is not None:
        valid_ids = set()

        if source2_ids is not None:
            valid_ids.update(source2_ids)

        if source3_ids is not None:
            valid_ids.update(source3_ids)

    ground_truth = {}

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
                ground_truth[s1_id] = []
                continue

            matches = [
                x.strip()
                for x in raw_matches.split(",")
                if x.strip()
            ]

            if valid_ids is not None:
                matches = [
                    x for x in matches
                    if x in valid_ids
                ]

            # Remove duplicates while preserving order
            matches = list(dict.fromkeys(matches))

            ground_truth[s1_id] = matches

    return ground_truth