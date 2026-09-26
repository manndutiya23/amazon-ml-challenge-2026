# Blocking Interface Contract (for the matching model)

`src/blocking.py` is the candidate-generation stage. This is the contract
the matching model (`src/features.py`, `src/train.py`, `src/predict.py`)
can rely on.

## Public API

```python
from blocking import generate_candidates_s2, generate_candidates_s3

s2_candidates = generate_candidates_s2(source1_df, source2_df)
s3_candidates = generate_candidates_s3(source1_df, source3_df)
```

### Input

Both `source1_df` and `source2_df` / `source3_df` are pandas DataFrames with
exactly the raw challenge columns, as read straight from the `.tsv` files
(`dtype=str`, no pre-normalization needed - `blocking.py` normalizes
internally):

| column | type | notes |
|---|---|---|
| `entity_id` | str | `S1-...` in source1, `S2-...`/`S3-...` in source2/source3 |
| `business_name` | str | may be empty |
| `business_address` | str | may be empty |
| `country` | str | may be empty; treat as an open string set (test adds `France`) |

No other columns are read. Row order doesn't matter. Duplicate
`entity_id`s in the input aren't deduplicated by `blocking.py` - dedupe
upstream if that's possible in your data.

### Output

`dict[s1_entity_id] -> sorted list[str]` of candidate entity_ids from that
source (S2 ids from `generate_candidates_s2`, S3 ids from
`generate_candidates_s3`).

- **Every** `entity_id` in `source1_df` gets a key in the output dict, even
  if the value is `[]` (no candidates found - most often true singletons,
  occasionally a genuine blocking miss).
- **Candidate IDs are deduplicated** - the value is built from a `set`
  union across strategies, then sorted, so no id ever repeats within one
  S1's list.
- To get the final `candidate_pairs.tsv` per the challenge's submission
  format, merge the two dicts per S1 id:
  `candidate_entity_ids = s2_candidates[s1_id] + s3_candidates[s1_id]`
  (each half is already dedup'd; a given id can't appear in both dicts
  since S2/S3 id spaces don't overlap).

### Strategy selection

Both functions take an optional `strategy_names` argument (defaults to
`blocking.DEFAULT_UNION_STRATEGIES`) - pass a different tuple of names from
`blocking.STRATEGIES` to use a different combination, e.g. for an
experiment or to drop down to a cheaper/looser strategy.

```python
# Recall ceiling (expensive - all strategies unioned):
generate_candidates_s2(s1, s2, strategy_names=tuple(blocking.STRATEGIES.keys()))
```

## What's inside `DEFAULT_UNION_STRATEGIES`

A **union** (OR) of four independent blocking strategies - a record is a
candidate if it shares a block key with the query under *any one* of these:

| strategy | key | purged? |
|---|---|---|
| `name_token_pairs` | sorted pair of two significant name tokens (survives reordering/typos in the other tokens) | no |
| `address_digits` | street-number / PIN-code digit runs, len >= 3 | no |
| `country_address_first_purged` | `country + first significant address token` | yes, 0.05% |
| `name_first_token_purged` | first significant name token (catches names with only one significant token, where no bigram can form) | yes, 0.05% |

"Significant" name tokens exclude a hardcoded legal-suffix stopword list
(`Inc`, `Corp`, `Pvt`, `Ltd`, `LLC`, ... - see `LEGAL_SUFFIX_STOPWORDS`) and
tokens shorter than 2 characters. "Significant" address tokens similarly
exclude generic address words (`Street`, `Road`, `Near`, `Sector`, ... - see
`ADDRESS_STOPWORDS`) and pure digit tokens.

## How purging works

After building a raw index for a strategy (`{key: [entity_id, ...]}`),
purging drops any key whose block exceeds `max_block_frac` of the indexed
corpus (`STRATEGIES[name][2]` in the registry, e.g. `0.0005` = 0.05%).
Those keys are common words/tokens that appear in too many records to be
discriminating - keeping them only adds candidate volume, not recall. This
is why `name_token_pairs` and `address_digits` are *not* purged (a shared
2-token pair or a distinctive digit run rarely creates an oversized block
on its own) while the single-token strategies are.

## Diagnostics

For candidate counts/volume diagnostics (e.g. to check average
candidates/S1 on your own data slice), the returned dict is enough:

```python
import statistics

sizes = [len(s2_candidates[s1_id]) + len(s3_candidates[s1_id]) for s1_id in s2_candidates]
print("avg candidates/S1:", statistics.mean(sizes))
print("p50/p95/max:", statistics.median(sizes), sorted(sizes)[int(len(sizes)*0.95)], max(sizes))
```

For per-strategy breakdowns (which strategy contributed which candidates,
recall per strategy), use the lower-level functions directly - see
`notebooks/04_blocking_experiments.ipynb` for the pattern:

```python
records = blocking._rows_to_records(source2_df)
index = blocking.build_index(records, "keys_name_token_pairs", multi=True)
# blocking.candidates_for_record(one_prepared_s1_record, {"name_token_pairs": index}, ["name_token_pairs"])
```

## Measured recall/volume ceiling

**94.4% candidate recall at ~7,600 candidates/S1** (extrapolated to the
full training corpus from a 25%-background sample; see
`docs/blocking_experiment_report.md` for the full experiment and
`docs/blocking_miss_taxonomy.md` for what the remaining ~5.6% of misses
look like).

**Do not treat 94.4% as the final ceiling to design around yet.** The plan
is: build the end-to-end pipeline (blocking -> pair features -> classifier
-> threshold -> F0.5 validation) first, then decide from *model* error
analysis whether the blocking misses are actually costing final F0.5 before
spending more blocking-recall budget - F0.5 penalizes false positives 2x
over false negatives, so a wider candidate set is only worth it if the
classifier can actually keep precision high on the extra candidates.
