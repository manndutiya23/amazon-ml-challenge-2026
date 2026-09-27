"""Bounded training from an explicitly validated challenge dataset."""

from __future__ import annotations

import argparse
import csv
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from features import feature_pair
from blocking import prepare_record
from predict_streaming import DiskBlockIndex
from train import FEATURE_COLUMNS, train_model
from validate_dataset import validate_dataset


def _sample_ground_truth(path: Path, max_s1: int, seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    sample: Dict[str, List[str]] = {}
    seen = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            seen += 1
            s1_id = row["source1_entity_id"]
            matches = [part.strip() for part in row["matched_entity_ids"].split(",") if part.strip()]
            if len(sample) < max_s1:
                sample[s1_id] = matches
            else:
                replacement = rng.randrange(seen)
                if replacement < max_s1:
                    old_id = next(iter(sample))
                    del sample[old_id]
                    sample[s1_id] = matches
    return sample


def _load_sample_s1(path: Path, ids: Set[str], chunk_size: int) -> Dict[str, Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    import pandas as pd
    for frame in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size):
        for row in frame.itertuples(index=False):
            if row.entity_id in ids:
                records[row.entity_id] = {
                    "entity_id": row.entity_id,
                    "business_name": row.business_name,
                    "business_address": row.business_address,
                    "country": row.country,
                }
    return records


def train_correct(
    data_root: Path,
    output_model: Path,
    max_s1: int,
    negatives_per_s1: int,
    index_chunk_size: int,
    seed: int,
    max_candidates_per_s1: int,
) -> None:
    validate_dataset(data_root)
    data_root = Path(data_root).resolve()
    sample_gt = _sample_ground_truth(data_root / "train" / "train_ground_truth.tsv", max_s1, seed)
    source1 = _load_sample_s1(data_root / "train" / "train_source1.tsv", set(sample_gt), index_chunk_size)
    index_path = output_model.with_suffix(".training_index.sqlite")
    index = DiskBlockIndex.build(
        [data_root / "train" / "train_source2.tsv", data_root / "train" / "train_source3.tsv"],
        index_path,
        index_chunk_size,
        max_candidates_per_s1,
    )
    positives = []
    negatives = []
    rng = random.Random(seed)
    for s1_id, truth in sample_gt.items():
        if s1_id not in source1:
            raise RuntimeError(f"Sampled ground-truth S1 is absent from train_source1: {s1_id}")
        target_rows = index.target_rows(truth)
        positives.extend(
            feature_pair(source1[s1_id], target_rows[target_id])
            for target_id in truth
            if target_id in target_rows
        )
        candidates, _ = index.candidates(
            prepare_record(
                source1[s1_id]["business_name"],
                source1[s1_id]["business_address"],
                source1[s1_id]["country"],
            )
        )
        negative_ids = [candidate for candidate in candidates if candidate not in set(truth)]
        rng.shuffle(negative_ids)
        negative_ids = negative_ids[:negatives_per_s1]
        negative_rows = index.target_rows(negative_ids)
        negatives.extend(
            feature_pair(source1[s1_id], negative_rows[target_id])
            for target_id in negative_ids
            if target_id in negative_rows
        )
    index.close()
    if not positives or not negatives:
        raise RuntimeError("Training sample produced no positive or negative examples")
    rows = positives + negatives
    X = np.asarray([[row[column] for column in FEATURE_COLUMNS] for row in rows], dtype=np.float32)
    y = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int8)
    model = train_model(X, y)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    with output_model.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved model: {output_model}")
    print(f"Training rows: {len(rows):,} ({len(positives):,} positive, {len(negatives):,} negative)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, default=Path("models/model_correct_dataset.pkl"))
    parser.add_argument("--max-s1", type=int, default=25_000)
    parser.add_argument("--negatives-per-s1", type=int, default=10)
    parser.add_argument("--index-chunk-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-candidates-per-s1", type=int, default=10_000)
    args = parser.parse_args()
    train_correct(
        args.data_root, args.output_model, args.max_s1, args.negatives_per_s1,
        args.index_chunk_size, args.seed, args.max_candidates_per_s1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())