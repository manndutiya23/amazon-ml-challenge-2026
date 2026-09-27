# Streaming Inference Design

This pipeline is separate from the legacy inference files and never falls back
to `data/raw`.

## Dataset authority

Every training or inference run requires `--data-root`. The validator checks
the expected train/test files, row counts, columns, ID prefixes, uniqueness,
countries, and ground-truth referential integrity. A missing or mismatched
dataset causes a non-zero exit before indexing or model work begins.

## Indexing

`predict_streaming.py` stores target records and block memberships in a SQLite
database on disk. The existing `prepare_record()` and production four-strategy
union are reused. Oversized country/address-first and name-first blocks use the
existing 0.05% purge rule. The two unpurged strategies remain subject to an
explicit per-S1 cap.

The default cap is 10,000 candidates per S1. When a candidate set exceeds it,
the sorted candidate IDs are deterministically truncated to the first 10,000;
the run reports how many S1 rows were truncated. This is an intentional,
documented recall trade-off, and the retained candidates are exactly the ones
scored and written to `candidate_pairs.tsv`.

## Bounded inference

Test S1 is read with pandas chunks. For each S1 row, candidates are queried,
scored in NumPy batches, and written immediately. No global candidate mapping,
feature list, metadata list, match mapping, or full feature DataFrame exists.
RSS, elapsed time, processed S1 rows, and processed candidates are printed at
chunk boundaries. `--max-s1 10000` uses the identical architecture and writes
`test_matching_results.tsv` and `test_candidate_pairs.tsv`.

## Bounded training

`train_correct.py` validates the supplied dataset, reservoir-samples a bounded
number of training S1/ground-truth rows, builds the same disk-backed blocker,
uses ground-truth links as positives, and samples a bounded number of blocked
hard negatives per S1. It writes a new model, by default
`models/model_correct_dataset.pkl`, and never overwrites `baseline_model.pkl`.

The training feature matrix is bounded by the sampled S1 count and negative
limit. It is not a full all-pairs training matrix.

## Commands

```bash
python src/validate_dataset.py --data-root data/dataset
python src/train_correct.py --data-root data/dataset
python src/predict_streaming.py --data-root data/dataset --model models/model_correct_dataset.pkl --max-s1 10000
python src/predict_streaming.py --data-root data/dataset --model models/model_correct_dataset.pkl
```

The full inference command is intentionally not run automatically during
development.