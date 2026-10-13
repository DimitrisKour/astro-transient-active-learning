# Historical acquisition-policy comparison

The launcher trains only entropy_diversity and entropy_balanced_explore.
For the existing historical suite, reuse its random/entropy results and its
validated seed-specific bootstrap to compare all four policies. Each new
policy has its own run directory; earlier selections and models stay intact.
Defaults are experimental starting
settings, not parameters demonstrated optimal on this dataset.

## Shared experimental conditions

- The original historical training objects are partitioned into L0 and U0,
  with split seed 135, initial fraction 0.10, and a minimum of five L0 objects
  per class. Validation and test splits remain immutable.
- All strategies for a given model seed share the same L0 bootstrap. Model
  seeds 135, 246, and 357 use the same object partition.
- Each cycle acquires ceil(0.10 * historical_objects) objects. The complete
  pool schedule consumes the exact remainder in its final cycle: for the
  current registry, 473 objects in cycles 1–8 and 469 in cycle 9, giving all
  4,729 historical training objects at the endpoint.
- ViT training still restarts from random initialization each stage, with
  frozen published expert towers, maximum 50 epochs, patience 10, current
  training-manifest class weights, and per-sample historical weighting.
- Acquisition sees model probabilities/embeddings and already confirmed
  labels only. The hidden-label oracle is queried after selection. Object
  class counts deduplicate alert rows and map native labels explicitly.

## Three entropy methods

Let p_c(x) be the current classifier probability for class c, h(x) its
final-classifier input embedding, k the object query budget, and n_c the
number of confirmed labeled objects in class c before this cycle.

### Entropy

H(x) = -sum_c p_c(x) log p_c(x).

Select the k largest H. Every object is ranked independently. It prioritizes
uncertainty but imposes no batch diversity or class coverage condition.

### Entropy plus diversity

First form T from the min(|U|, ceil(m*k)) highest-entropy objects, with m=3.
Normalize each embedding to unit length; a zero vector remains the origin.
Start S with the highest-entropy row. Repeatedly append:

x* = argmax_(x in T\S) min_(s in S) ||h_normalized(x)-h_normalized(s)||^2.

Entropy order breaks distance ties. This encourages spread within the uncertain
shortlist. It uses batch diversity, not distance to the existing labeled set.
It is a specific shortlist/farthest-first heuristic, not BADGE. Selected
scores retain their entropy meaning; greedy distances are saved separately.

### Entropy plus soft class balancing and exploration

Define smoothed counts a_c=n_c+alpha, their class mean a_bar, and:

w_c = clip((a_bar/a_c)^beta, 1, cap)

s(x) = H(x) * [1 + strength * sum_c p_c(x)(w_c-1)].

Defaults: alpha=1, beta=0.5, strength=1, cap=3. The multiplier lies between
1 and 3 at these defaults. Probabilities for all classes contribute, not only
the argmax class. There are no true-label quotas. Equal counts, cap=1, or
strength=0 reduce the score to ordinary entropy.

First draw floor(epsilon*k+0.5) objects uniformly without replacement from
all currently eligible candidates, using epsilon=0.20. Fill the remaining
slots with the largest s among candidates not already drawn. Exploration
can reach objects the model confidently misclassifies; rarity bonuses alone
cannot reliably find such objects. Random decisions use a fresh generator
seeded by [model_seed, cycle, 173].

## Recorded diagnostics and resume behavior

- `selection.json`: IDs, original inference indices, scores, and selector
  metadata. Diversity saves greedy distances and shortlist settings. Balancing
  saves ordered classes, confirmed object counts, class weights, and explicit
  exploration/targeted IDs.
- `candidate_scores.npz`: every candidate score. `inference.npz` preserves
  predictions and embeddings. They live on HDD.
- `experiment.json`: acquisition settings and, for balancing, the class-column
  order and native-to-canonical label map. Changing a policy rejects resume.
- A single-cycle launch validates every strategy state before advancing any.
  Repeating that same round skips completed strategies. A failure/time limit
  leaves successors blocked by afterok. A training stage interrupted midway
  may retrain; epoch/optimizer resume is not newly introduced by this patch.
- Split writers use unique temporary files so parallel model seeds can validate
  the same immutable split safely.

## Evaluation

Run `scripts/report_policy_compare.py` separately on validation and test.
It reports normalized area under the curve over the observed portion up to
50% labeled data, endpoint metrics, and the first/persistent label budgets
at 95% of the baseline-to-full-data improvement for each run.

The normalized early area is the trapezoidal integral divided by the observed
budget interval. Compare it only when runs cover the same interval; incomplete
curves are never extrapolated. Thresholds require a completed full-data cycle
and positive baseline-to-endpoint improvement. They use each run's endpoint,
so they must be read together with absolute endpoint quality; a weak endpoint
can make its own threshold easier to reach. Use paired seed comparisons and
per-class precision/recall, macro-F1, balanced accuracy and accuracy.

The final dataset is a full-data reference, not a proven optimal classifier.
All strategies acquire the same object set, but selection order changes
manifest order, and augmentation/training can still yield different models.
The report hashes sorted full-manifest rows to check data-content agreement.
If endpoint differences are large, inspect training before attributing early
advantages entirely to acquisition.

## Delta operation

For baseline reuse, select the existing suite explicitly:

```bash
export POLICY_SUITE_ID=astrominn_historical_replay_v1
```

Use a fixed code checkout for queued jobs. See
`docs/existing_replay_continuation.md` for the existing runs and final cycle.
No new dependencies are required in Delta's existing
al-pipeline environment. PYTHONPATH chooses the isolated source copy without
reinstalling the editable package or modifying the original environment.

Prepare without GPU training or job submission:

```bash
bash scripts/adaptive/submit_policy_compare.sh 135 --prepare-only
```

This probes projects/HDD writes, creates a registry with only storage roots
overridden, validates the existing bootstrap, audits split
disjointness/exclusions and file presence, and freezes
source hashes. It does not reserve quota. The current pool's historical input
files can stay on NVMe; all new heavy writes go to HDD.

When ready:

```bash
bash scripts/adaptive/submit_policy_compare.sh 135 01:00:00
# Later, after reviewing the first seed:
bash scripts/adaptive/submit_policy_compare.sh 246 01:00:00
bash scripts/adaptive/submit_policy_compare.sh 357 01:00:00
```

Each seed gets nine sequential GPU jobs. Each job runs the two new strategies
sequentially on one GPU. Random and entropy are never included in this chain. Each chain is submitted held and released only after
all submissions succeed. Failure before release cancels only its new jobs;
the partial submission record remains for inspection. The script never cancels
existing Experiment B jobs and refuses duplicate or progressed submissions.

One hour is a starting request for the two-policy comparison. Earlier
16–17 minute jobs contained three strategies and no diversity selection;
35 minutes has not been verified for these new jobs or their later cycles.
The command's second argument overrides wall time explicitly.

Runtime checks detect source changes after preparation. Freeze this code copy
while its jobs are queued/running. Changing the scientific settings requires a
new suite ID (e.g. POLICY_SUITE_ID=astrominn_policy_compare_v2). If a cycle fails,
inspect its log/state and retry the same cycle/settings; then update its immediate
successor's dependency to the new job ID. Do not blindly rerun the submission
script or delete its record.

After results are available:

```bash
python scripts/report_policy_compare.py \
  --suite-root /projects/bcrv/dkourtesis/applecider_adaptive/runs/astrominn_historical_replay_v1 \
  --split validation --metric f1_macro \
  --output /projects/bcrv/dkourtesis/applecider_adaptive/reports/policy_compare_validation
```

The script writes `summary.json` and `curves.csv`. Repeat with `--split test`
and a separate output directory for the held-out report. Tune policies using
validation results, not repeated test-guided changes.
