# Blocking / Candidate Generation - Experiment Report

Owner: Laksh
Code: `src/blocking.py`
Reproduce: `notebooks/04_blocking_experiments.ipynb`

## Goal

Blocking sets the recall ceiling for the whole pipeline: any true S1->S2/S3
match that never appears in the candidate set can never be recovered by the
matching model. At the same time, the challenge explicitly rewards a
*smaller* candidate set per S1 entity, so the metric that matters is the
**recall / candidate-volume trade-off**, not recall alone.

## Method

Building blocking indices over the full training corpus (S1: 2.2M, S2:
5.03M, S3: 5.29M rows) for every candidate strategy, every iteration, is slow
and memory-heavy on a dev machine. Instead:

1. Sample 25,000 S1 query entities from `train_ground_truth.tsv` (preserves
   the natural ~94% matched / ~6% singleton mix).
2. Collect the full set of their true S2/S3 matches and **force-include**
   those records in the sampled corpus, so recall is never lowered by
   corpus-sampling artifacts (recall only depends on whether the true match
   shares a block key with the query - not on how much other, unrelated
   data surrounds it).
3. Add a random ~25% background sample of the rest of S2/S3 (~1.29M S2 +
   1.35M S3 rows) so candidate-volume numbers reflect realistic collision
   behavior, then extrapolate volume to full scale by ~3.9x.

This gave 25,000 queries / 86,734 true positive links to evaluate against.

**Candidate recall** = fraction of true (S1, S2/S3) links whose target id
appears in the generated candidate set for that S1 (pair-level / micro
average - matches how the challenge scores `candidate_pairs.tsv`).

## Round 1 - single/multi-key blocking (no purging)

| Strategy | Recall | Candidates/S1 (sample) | Full-scale est. |
|---|---:|---:|---:|
| Exact normalized name | 22.0% | 3.1 | 12 |
| Name first token | 77.1% | 1,773.6 | 6,900 |
| Name tokens (every significant token, OR) | 85.0% | 16,104.5 | 62,800 |
| Address first token | 61.8% | 3,187.7 | 12,400 |
| Address digit tokens (street#/PIN) | 57.3% | 1,294.8 | 5,000 |
| Country + name first token | 77.1% | 1,456.6 | 5,700 |
| Country + address first token | 61.8% | 2,854.5 | 11,100 |
| Name first token AND address digit | 44.7% | 2.5 | 10 |
| **Union: name tokens + address digits + country/address** | **97.1%** | **20,205.6** | **78,800** |

**Verdict:** the best union reached 97.1% recall, but at ~78,800
candidates/S1 at full scale (~175 billion S1xcandidate pairs total across
2.2M S1 entities) - not usable. Root cause: single-token and first-token
blocking keys collide heavily on common words (generic business terms,
common city names), producing enormous blocks that add volume without
adding recall.

## Round 2 - block purging + bigram blocking

Two fixes, both standard entity-resolution techniques for this exact
failure mode:

- **Block purging**: after building an index, drop any key whose block
  would contain more than 0.05% of the indexed corpus. Those keys are too
  generic to discriminate and are pure candidate-volume noise.
- **Bigram (2-token) name blocking**: require two shared significant name
  tokens instead of one. Far more selective (survives word reordering and
  one noisy/missing token, without the single-common-word blowup).

| Strategy | Recall | Candidates/S1 (sample) | Full-scale est. |
|---|---:|---:|---:|
| Name first token (purged) | 47.1% | 107.3 | 418 |
| Name tokens (purged) | 56.5% | 281.1 | 1,096 |
| Name token bigrams | 71.1% | 374.6 | 1,461 |
| Address first token (purged) | 47.4% | 181.2 | 706 |
| Country + name first token (purged) | 48.1% | 121.5 | 474 |
| Country + address first token (purged) | 48.2% | 182.5 | 712 |
| Union: name bigrams + address digits | 86.7% | 1,667.7 | 6,504 |
| Union: bigrams + digits + country/address (purged) | 92.5% | 1,847.5 | 7,205 |
| **Union: bigrams + digits + country/address + name-first (all purged)** | **94.4%** | **1,944.8** | **7,589** |

**~10x fewer candidates for only 2.7pp less recall** than round 1's best
union.

## Recommendation (shipped as `blocking.DEFAULT_UNION_STRATEGIES`)

Union of:
1. `name_token_pairs` - bigrams of significant name tokens
2. `address_digits` - street-number / PIN-code digit runs (len >= 3)
3. `country_address_first_purged` - country + first significant address
   token, purged of oversized blocks
4. `name_first_token_purged` - first significant name token, purged of
   oversized blocks (recovers cases with only one significant name token,
   where no bigram can form)

**Measured: 94.4% candidate recall at ~1,945 candidates/S1 on the sample
corpus (~7,600/S1 extrapolated to the full training corpus).**

## What's still missing (~5.6% of true links)

Cases where the name is too garbled to share even one significant token
pair *and* the address digit / first-token / country signals also disagree.
Likely candidates for closing this gap in a follow-up:

- A phonetic key (soundex/metaphone) on the first significant name token,
  to catch transliteration variants that don't share literal tokens.
- Fuzzy/edit-distance blocking (e.g. a low-dimensional MinHash/LSH pass) for
  near-duplicate names that a bigram just barely misses.

Both would add candidates, so the right move is to decide this once we know
how much recall headroom the matching model actually needs under F0.5
(which weights precision 2x over recall) - pushing blocking recall further
than the matching model can use precisely is wasted candidate volume.

## Files

- `src/blocking.py` - normalization, strategy registry, `generate_candidates_s2`
  / `generate_candidates_s3` production API.
- `notebooks/04_blocking_experiments.ipynb` - reproduces this experiment
  end-to-end (sampling, index-building, evaluation).
- `data/processed/blocking_experiment_results.json`,
  `blocking_experiment_results_v2.json` - raw per-strategy numbers.
