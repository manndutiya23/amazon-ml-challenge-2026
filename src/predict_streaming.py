"""Bounded, disk-indexed inference for the challenge-scale dataset."""

from __future__ import annotations

import argparse
import csv
from itertools import islice
import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from blocking import DEFAULT_UNION_STRATEGIES, STRATEGIES, prepare_record
from features import feature_pair
from train import FEATURE_COLUMNS
from validate_dataset import validate_dataset


INDEX_STRATEGIES = tuple(DEFAULT_UNION_STRATEGIES)
PURGE_FRAC = 0.0005


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0.0


def _mem(stage: str, started: float, s1_rows: int = 0, candidates: int = 0) -> None:
    print(
        f"[MEM] stage={stage} elapsed={time.time() - started:.1f}s "
        f"RSS={_rss_mb():.1f}MB S1={s1_rows} candidates={candidates}",
        flush=True,
    )


def _chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )


class DiskBlockIndex:
    """SQLite-backed target records and block memberships.

    Candidate memberships stay on disk. Only one S1 record and one bounded
    candidate/feature batch are materialized in Python at a time.
    """

    def __init__(self, db_path: Path, max_candidates: int = 10_000):
        self.db_path = Path(db_path)
        self.max_candidates = max_candidates
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS targets (
                entity_id TEXT PRIMARY KEY,
                business_name TEXT NOT NULL,
                business_address TEXT NOT NULL,
                country TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS blocks (
                strategy TEXT NOT NULL,
                block_key TEXT NOT NULL,
                entity_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS block_counts (
                strategy TEXT NOT NULL,
                block_key TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                PRIMARY KEY(strategy, block_key)
            );
            """
        )
        self.connection.execute("DELETE FROM targets")
        self.connection.execute("DELETE FROM blocks")
        self.connection.execute("DELETE FROM block_counts")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def _insert_frame(self, frame: pd.DataFrame) -> None:
        targets: List[Tuple[str, str, str, str]] = []
        blocks: List[Tuple[str, str, str]] = []
        for row in frame.itertuples(index=False):
            entity_id = str(row.entity_id)
            name = str(row.business_name or "")
            address = str(row.business_address or "")
            country = str(row.country or "")
            targets.append((entity_id, name, address, country))
            record = prepare_record(name, address, country)
            for strategy in INDEX_STRATEGIES:
                key_field, multi, _ = STRATEGIES[strategy]
                values = record[key_field] if multi else [record[key_field]]
                for key in values:
                    if key:
                        blocks.append((strategy, str(key), entity_id))
        self.connection.executemany("INSERT OR IGNORE INTO targets VALUES (?, ?, ?, ?)", targets)
        self.connection.executemany("INSERT INTO blocks VALUES (?, ?, ?)", blocks)

    @classmethod
    def build(cls, paths: Sequence[Path], db_path: Path, chunk_size: int, max_candidates: int) -> "DiskBlockIndex":
        index = cls(db_path, max_candidates=max_candidates)
        for path in paths:
            for frame in _chunks(path, chunk_size):
                index._insert_frame(frame)
                index.connection.commit()
        index.connection.execute("DELETE FROM block_counts")
        index.connection.execute(
            "INSERT INTO block_counts SELECT strategy, block_key, COUNT(*) FROM blocks GROUP BY strategy, block_key"
        )
        for strategy in ("country_address_first_purged", "name_first_token_purged"):
            index.connection.execute(
                "DELETE FROM blocks WHERE strategy = ? AND EXISTS ("
                "SELECT 1 FROM block_counts c WHERE c.strategy = ? AND c.block_key = blocks.block_key AND c.row_count > ?"
                ")",
                (strategy, strategy, max(1, int(index.count_targets() * PURGE_FRAC))),
            )
        index.connection.execute("CREATE INDEX IF NOT EXISTS blocks_lookup ON blocks(strategy, block_key)")
        index.connection.execute("CREATE INDEX IF NOT EXISTS blocks_entity ON blocks(entity_id)")
        index.connection.commit()
        return index

    def count_targets(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM targets").fetchone()[0])

    def candidates(self, record: Dict[str, object]) -> Tuple[List[str], bool]:
        values: List[str] = []
        for strategy in INDEX_STRATEGIES:
            key_field, multi, _ = STRATEGIES[strategy]
            keys = record[key_field] if multi else [record[key_field]]
            for key in keys:
                if key:
                    values.extend(
                        row[0]
                        for row in self.connection.execute(
                            "SELECT entity_id FROM blocks WHERE strategy = ? AND block_key = ?",
                            (strategy, str(key)),
                        )
                    )
        candidates = sorted(set(values))
        truncated = len(candidates) > self.max_candidates
        return candidates[: self.max_candidates], truncated

    def target_rows(self, entity_ids: Sequence[str]) -> Dict[str, Dict[str, str]]:
        if not entity_ids:
            return {}
        result: Dict[str, Dict[str, str]] = {}
        for start in range(0, len(entity_ids), 500):
            batch = entity_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT entity_id, business_name, business_address, country FROM targets WHERE entity_id IN ({placeholders})",
                list(batch),
            )
            result.update({
                row[0]: {
                    "entity_id": row[0],
                    "business_name": row[1],
                    "business_address": row[2],
                    "country": row[3],
                }
                for row in rows
            }
            )
        return result


def _score_candidates(
    model: object,
    source1: Dict[str, str],
    candidate_ids: Sequence[str],
    index: DiskBlockIndex,
    score_batch_size: int,
    threshold: float,
) -> List[str]:
    matches: List[str] = []
    for start in range(0, len(candidate_ids), score_batch_size):
        batch_ids = candidate_ids[start : start + score_batch_size]
        target_rows = index.target_rows(batch_ids)
        if len(target_rows) != len(batch_ids):
            missing = sorted(set(batch_ids) - set(target_rows))[:5]
            raise RuntimeError(f"Candidate IDs missing from target index: {missing}")
        feature_rows = [
            feature_pair(source1, target_rows[entity_id])
            for entity_id in batch_ids
            if entity_id in target_rows
        ]
        if not feature_rows:
            continue
        matrix = np.asarray(
            [[row[column] for column in FEATURE_COLUMNS] for row in feature_rows],
            dtype=np.float32,
        )
        probabilities = model.predict_proba(matrix)[:, 1]
        matches.extend(
            entity_id
            for entity_id, probability in zip(batch_ids, probabilities)
            if probability >= threshold
        )
        del target_rows, feature_rows, matrix, probabilities
    return matches


def _validate_outputs(
    s1_path: Path,
    matching_path: Path,
    candidate_path: Path,
    expected_rows: int,
) -> None:
    """Validate output shape and alignment without retaining the output files."""
    source_ids: Iterator[str] = islice((
        str(row.entity_id)
        for frame in _chunks(s1_path, 50_000)
        for row in frame.itertuples(index=False)
    ), expected_rows)
    with matching_path.open("r", encoding="utf-8", newline="") as matching_handle, candidate_path.open("r", encoding="utf-8", newline="") as candidate_handle:
        matching_reader = csv.reader(matching_handle, delimiter="\t")
        candidate_reader = csv.reader(candidate_handle, delimiter="\t")
        if next(matching_reader, None) != ["source1_entity_id", "matched_entity_ids"]:
            raise RuntimeError("matching_results.tsv has an invalid header")
        if next(candidate_reader, None) != ["source1_entity_id", "candidate_entity_ids"]:
            raise RuntimeError("candidate_pairs.tsv has an invalid header")
        rows = 0
        for matching_row, candidate_row in zip(matching_reader, candidate_reader):
            if len(matching_row) != 2 or len(candidate_row) != 2:
                raise RuntimeError(f"Malformed output row at index {rows + 1}")
            expected_s1 = next(source_ids, None)
            if expected_s1 is None or matching_row[0] != expected_s1 or candidate_row[0] != expected_s1:
                raise RuntimeError(f"Output S1 row misalignment at index {rows + 1}")
            candidate_ids = [value for value in candidate_row[1].split(",") if value]
            matched_ids = [value for value in matching_row[1].split(",") if value]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise RuntimeError(f"Duplicate candidate ID for {expected_s1}")
            if len(matched_ids) != len(set(matched_ids)):
                raise RuntimeError(f"Duplicate matched ID for {expected_s1}")
            if any(not value.startswith(("S2-", "S3-")) for value in candidate_ids + matched_ids):
                raise RuntimeError(f"Invalid target ID prefix for {expected_s1}")
            if not set(matched_ids).issubset(candidate_ids):
                raise RuntimeError(f"Predicted match is absent from candidates for {expected_s1}")
            rows += 1
        if next(matching_reader, None) is not None or next(candidate_reader, None) is not None:
            raise RuntimeError("Output files have different row counts")
        if next(source_ids, None) is not None or rows != expected_rows:
            raise RuntimeError("Output row count does not match processed S1 count")


def run_inference(
    data_root: Path,
    model_path: Path,
    output_dir: Path,
    threshold: float,
    max_s1: Optional[int],
    s1_chunk_size: int,
    score_batch_size: int,
    index_chunk_size: int,
    max_candidates: int,
) -> Dict[str, float]:
    started = time.time()
    validate_dataset(data_root)
    data_root = Path(data_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index = DiskBlockIndex.build(
        [data_root / "test" / "test_source2.tsv", data_root / "test" / "test_source3.tsv"],
        output_dir / "test_target_index.sqlite",
        index_chunk_size,
        max_candidates,
    )
    _mem("after_index", started)
    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    matching_path = output_dir / ("test_matching_results.tsv" if max_s1 else "matching_results.tsv")
    candidate_path = output_dir / ("test_candidate_pairs.tsv" if max_s1 else "candidate_pairs.tsv")
    s1_path = data_root / "test" / "test_source1.tsv"
    processed = candidates_seen = truncated_rows = 0
    with matching_path.open("w", encoding="utf-8", newline="") as matching_handle, candidate_path.open("w", encoding="utf-8", newline="") as candidate_handle:
        matching_writer = csv.writer(matching_handle, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_handle, delimiter="\t", lineterminator="\n")
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for frame in _chunks(s1_path, s1_chunk_size):
            for row in frame.itertuples(index=False):
                if max_s1 is not None and processed >= max_s1:
                    break
                s1_id = str(row.entity_id)
                source1 = {
                    "entity_id": s1_id,
                    "business_name": str(row.business_name or ""),
                    "business_address": str(row.business_address or ""),
                    "country": str(row.country or ""),
                }
                candidates, was_truncated = index.candidates(prepare_record(source1["business_name"], source1["business_address"], source1["country"]))
                matches = _score_candidates(model, source1, candidates, index, score_batch_size, threshold)
                if not set(matches).issubset(candidates):
                    raise RuntimeError(f"Scored match is absent from candidates for {s1_id}")
                candidate_writer.writerow([s1_id, ",".join(candidates)])
                matching_writer.writerow([s1_id, ",".join(dict.fromkeys(matches))])
                processed += 1
                candidates_seen += len(candidates)
                truncated_rows += int(was_truncated)
            matching_handle.flush()
            candidate_handle.flush()
            _mem("after_s1_chunk", started, processed, candidates_seen)
            if max_s1 is not None and processed >= max_s1:
                break
    index.close()
    _validate_outputs(s1_path, matching_path, candidate_path, processed)
    summary = {
        "s1_rows": float(processed),
        "candidate_pairs": float(candidates_seen),
        "truncated_s1_rows": float(truncated_rows),
        "elapsed_seconds": time.time() - started,
        "peak_rss_mb": _rss_mb(),
    }
    print(f"[SUMMARY] {summary}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/dataset"))
    parser.add_argument("--model", type=Path, default=Path("models/model_correct_dataset.pkl"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--max-s1", type=int, default=None)
    parser.add_argument("--s1-chunk-size", type=int, default=1_000)
    parser.add_argument("--score-batch-size", type=int, default=500)
    parser.add_argument("--index-chunk-size", type=int, default=50_000)
    parser.add_argument("--max-candidates-per-s1", type=int, default=10_000)
    args = parser.parse_args()
    if args.max_s1 is not None and args.max_s1 <= 0:
        parser.error("--max-s1 must be positive")
    if args.max_candidates_per_s1 <= 0:
        parser.error("--max-candidates-per-s1 must be positive")
    run_inference(
        args.data_root, args.model, args.output_dir, args.threshold, args.max_s1,
        args.s1_chunk_size, args.score_batch_size, args.index_chunk_size,
        args.max_candidates_per_s1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())