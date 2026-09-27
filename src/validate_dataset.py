"""Validate the challenge dataset before training or inference."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

EXPECTED_COUNTS = {
    "test/test_source1.tsv": 1_732_544,
    "test/test_source2.tsv": 4_887_273,
    "test/test_source3.tsv": 5_082_316,
    "train/train_source1.tsv": 2_206_821,
    "train/train_source2.tsv": 5_034_616,
    "train/train_source3.tsv": 5_285_603,
    "train/train_ground_truth.tsv": 2_206_821,
}

# The challenge data is expected to contain these countries. Extra countries
# are allowed because the challenge can add a country in a later split.
EXPECTED_COUNTRIES = {"india", "us", "france"}


class DatasetValidationError(RuntimeError):
    """Raised when a supplied data root is not the expected challenge dataset."""


def _read_rows(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise DatasetValidationError(f"{path}: missing header")
        for row in reader:
            yield row


def _check_file(
    path: Path,
    expected_rows: int,
    columns: List[str],
    id_connection: sqlite3.Connection,
    id_group: str,
) -> Tuple[int, Set[str]]:
    if not path.is_file():
        raise DatasetValidationError(f"Missing required dataset file: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        actual_columns = reader.fieldnames or []
        missing = [column for column in columns if column not in actual_columns]
        if missing:
            raise DatasetValidationError(
                f"{path}: missing required columns: {', '.join(missing)}"
            )

        count = 0
        countries: Set[str] = set()
        for row in reader:
            count += 1
            if "entity_id" in columns:
                entity_id = (row.get("entity_id") or "").strip()
                if not entity_id:
                    raise DatasetValidationError(f"{path}: blank entity_id at row {count + 1}")
                try:
                    id_connection.execute(
                        "INSERT INTO entity_ids(group_name, entity_id) VALUES (?, ?)",
                        (id_group, entity_id),
                    )
                except sqlite3.IntegrityError:
                    raise DatasetValidationError(f"{path}: duplicate entity_id {entity_id}")
                countries.add((row.get("country") or "").strip().lower())

        if count != expected_rows:
            raise DatasetValidationError(
                f"{path}: expected {expected_rows:,} data rows, found {count:,}"
            )
        return count, countries


def _check_ground_truth(data_root: Path, id_connection: sqlite3.Connection) -> None:
    path = data_root / "train" / "train_ground_truth.tsv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or any(column not in reader.fieldnames for column in GROUND_TRUTH_COLUMNS):
            raise DatasetValidationError(f"{path}: required columns are {GROUND_TRUTH_COLUMNS}")
        for row_number, row in enumerate(reader, start=2):
            s1_id = (row.get("source1_entity_id") or "").strip()
            if not s1_id.startswith("S1-"):
                raise DatasetValidationError(f"{path}: invalid or unknown S1 ID {s1_id!r} at row {row_number}")
            s1_exists = id_connection.execute(
                "SELECT 1 FROM entity_ids WHERE group_name = ? AND entity_id = ?",
                ("train_s1", s1_id),
            ).fetchone()
            if s1_exists is None:
                raise DatasetValidationError(f"{path}: unknown S1 ID {s1_id!r} at row {row_number}")
            try:
                id_connection.execute(
                    "INSERT INTO ground_truth_ids(entity_id) VALUES (?)", (s1_id,)
                )
            except sqlite3.IntegrityError:
                raise DatasetValidationError(f"{path}: duplicate source1_entity_id {s1_id}")
            raw_matches = (row.get("matched_entity_ids") or "").strip()
            for match_id in (part.strip() for part in raw_matches.split(",") if part.strip()):
                if match_id.startswith("S2-"):
                    group = "train_s2"
                elif match_id.startswith("S3-"):
                    group = "train_s3"
                else:
                    group = ""
                valid = bool(group) and id_connection.execute(
                    "SELECT 1 FROM entity_ids WHERE group_name = ? AND entity_id = ?",
                    (group, match_id),
                ).fetchone()
                if not valid:
                    raise DatasetValidationError(
                        f"{path}: ground-truth target {match_id!r} is not in train S2/S3"
                    )

    gt_count = id_connection.execute("SELECT COUNT(*) FROM ground_truth_ids").fetchone()[0]
    train_s1_count = id_connection.execute(
        "SELECT COUNT(*) FROM entity_ids WHERE group_name = 'train_s1'"
    ).fetchone()[0]
    if gt_count != train_s1_count:
        raise DatasetValidationError("ground truth S1 IDs do not exactly match train_source1.tsv")


def validate_dataset(data_root: Path) -> Dict[str, int]:
    """Validate and return the authoritative row counts under ``data_root``."""
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise DatasetValidationError(f"Dataset root does not exist: {data_root}")

    countries: Set[str] = set()
    counts: Dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="dataset_validation_") as temp_dir:
        id_connection = sqlite3.connect(str(Path(temp_dir) / "ids.sqlite"))
        id_connection.execute(
            "CREATE TABLE entity_ids(group_name TEXT NOT NULL, entity_id TEXT NOT NULL, PRIMARY KEY(group_name, entity_id))"
        )
        id_connection.execute(
            "CREATE TABLE ground_truth_ids(entity_id TEXT PRIMARY KEY)"
        )
        source_specs = (
            ("test/test_source1.tsv", "S1-", "test_s1"),
            ("test/test_source2.tsv", "S2-", "test_s2"),
            ("test/test_source3.tsv", "S3-", "test_s3"),
            ("train/train_source1.tsv", "S1-", "train_s1"),
            ("train/train_source2.tsv", "S2-", "train_s2"),
            ("train/train_source3.tsv", "S3-", "train_s3"),
        )
        for relative_path, prefix, group in source_specs:
            path = data_root / relative_path
            count, file_countries = _check_file(
                path, EXPECTED_COUNTS[relative_path], SOURCE_COLUMNS, id_connection, group
            )
            counts[relative_path] = count
            countries.update(file_countries)
            for entity_id in id_connection.execute(
                "SELECT entity_id FROM entity_ids WHERE group_name = ?", (group,)
            ):
                if not entity_id[0].startswith(prefix):
                    raise DatasetValidationError(
                        f"{path}: entity_id {entity_id[0]!r} does not start with {prefix}"
                    )
            id_connection.commit()

        gt_path = data_root / "train" / "train_ground_truth.tsv"
        with gt_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or any(column not in reader.fieldnames for column in GROUND_TRUTH_COLUMNS):
                raise DatasetValidationError(f"{gt_path}: required columns are {GROUND_TRUTH_COLUMNS}")
            gt_rows = sum(1 for _ in reader)
        if gt_rows != EXPECTED_COUNTS["train/train_ground_truth.tsv"]:
            raise DatasetValidationError(
                f"{gt_path}: expected {EXPECTED_COUNTS['train/train_ground_truth.tsv']:,} data rows, found {gt_rows:,}"
            )
        counts["train/train_ground_truth.tsv"] = gt_rows
        _check_ground_truth(data_root, id_connection)
        id_connection.close()

    missing_countries = EXPECTED_COUNTRIES - countries
    if missing_countries:
        raise DatasetValidationError(
            "Dataset is missing expected challenge countries: "
            + ", ".join(sorted(missing_countries))
        )

    print(f"VALID DATASET: {data_root}")
    for relative_path, count in counts.items():
        print(f"  {relative_path}: {count:,} rows")
    print(f"  countries: {', '.join(sorted(countries - {''}))}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        validate_dataset(args.data_root)
    except (DatasetValidationError, OSError, UnicodeError) as exc:
        print(f"DATASET VALIDATION FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
