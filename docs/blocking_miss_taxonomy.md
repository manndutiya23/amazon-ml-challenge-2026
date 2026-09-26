# Miss Taxonomy - Round 2 Blocking (`DEFAULT_UNION_STRATEGIES`)

Categorizes the true S1->S2/S3 links that `blocking.DEFAULT_UNION_STRATEGIES`
misses, on the same sampled corpus used in `blocking_experiment_report.md`
(25,000 S1 queries, 86,734 true links). Purpose: know *why* the ~5.6% miss
rate happens before spending more blocking-recall budget on it - per the
project's rule, no more blocking expansion until model/error-analysis
evidence says it's worth it.

Reproduce: `data/processed/blocking_miss_taxonomy.json` has the full
category counts + one example per category; regenerate with
`notebooks/04_blocking_experiments.ipynb`'s classification cell (or the
one-off script this was built from).

## Results

Total true links checked: 86,734
Missed by `DEFAULT_UNION_STRATEGIES`: 4,850 (5.59%) - matches the 94.4%
recall reported in `blocking_experiment_report.md`.

| Category | Count | % of misses |
|---|---:|---:|
| Transliteration / script variation | 2,397 | 49.4% |
| Name variation not captured | 803 | 16.6% |
| Very short / common business name (after stopword stripping) | 749 | 15.4% |
| Address variation not captured | 629 | 13.0% |
| Missing address (one side has no address at all) | 272 | 5.6% |
| Country mismatch / missing country | 0 | 0.0% |
| Other | 0 | 0.0% |

(Classification priority: missing address > country mismatch >
transliteration/script > very-short-name > name-variation > address-variation
> other. Categories are mutually exclusive by this priority order.)

## What each bucket looks like

**Transliteration / script variation (49.4%, the dominant bucket)** - one
side is in a non-Latin script (mostly Devanagari for Indian records), the
other is romanized:

```
S1:    innovative consultants
       15 16 indrapuri madhubancolony karvenagar pune maharashtra | india
S2:    इन व ट व क सल ट ट स
       h no b3 15 16 indrapuri madhubancolony karvenagar pune poona मह र ष ट र | india
```

Note the address tokens *mostly* still agree in Latin script even here
(`indrapuri`, `madhubancolony`, `karvenagar`, `pune`) - a same-script
address signal is often present even when the name isn't, which is a
plausible future feature for the *matching model* (not blocking, since we'd
still need the record to be a candidate first - and it often already is via
address tokens/digits, just not via `name_token_pairs`).

**Important constraint**: closing this gap with real transliteration
(e.g. a script-conversion library or model) is *not* an "external database
lookup" in the prohibited sense if it's a generic, training-data-free
transformation rather than an entity lookup - but it's still meaningfully
out of scope for a quick blocking fix. Flagging for Mann/model discussion
rather than solving here.

**Name variation not captured (16.6%)** - genuinely different tokenizations
of the same name, most commonly a name glued into a single word (often a
domain-style DBA name):

```
S1:    reet exports india private limited
S2:    reetexportsindia com
```

No token overlap at all once `reetexportsindia` is one token - a
character-n-gram or edit-distance signal would catch this, not word-token
blocking.

**Very short / common business name (15.4%)** - after stripping legal-suffix
stopwords, one side collapses to 0-1 significant tokens, so no bigram can
form (this is *why* `name_first_token_purged` exists as a single-token
fallback in the union - but purging still drops it if that lone token is
too common):

```
S1:    perfect tech pvt ltd         -> tokens after stopwords: [perfect, tech]
S3:    perfect pvt ltd services     -> tokens after stopwords: [perfect]   ("services" is itself a stopword)
```

**Address variation not captured (13.0%)** - partial name overlap (often
just one shared token, not the two `name_token_pairs` needs) combined with
addresses that don't share a long-enough digit run either:

```
S1:    bharat infra private limited      | ward no 17 23 ... hoshangabad hoshangabad madhya pradesh
S3:    bharat private limited center     | ward no 17 mumbai mp
```

Digit tokens here (`17`, `23`, `7`, `2`) are all shorter than the
`MIN_DIGIT_TOKEN_LEN = 3` cutoff, so `address_digits` can't see them either.

**Missing address (5.6%)** - one side has an empty `business_address`,
removing the address-based signals entirely; blocking then depends
entirely on the name signal, which in these cases was also too weak (often
paired with a name typo, e.g. `e z clean` vs `e z cclon`).

## Implication for the matching model handoff

Per the plan in `docs/blocking_interface.md`: don't chase these buckets in
blocking yet. Two are structurally different problems the matching model
(not blocking) is better placed to reason about once a record *is* a
candidate via some other signal (address digits, partial name):
transliteration and address variation. "Very short name" and "missing
address" are cases where blocking has almost no signal to work with at all
- more relevant to decide via F0.5 error analysis whether they're worth
targeted handling (e.g. a stricter/looser threshold when address is empty)
rather than blocking changes.
