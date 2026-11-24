# OPD candidate-ranking diagnostics

Last updated: 2026-11-24

These optional diagnostics compare the teacher and student on the **current training batch**.
They reuse actor log-probability recomputation and teacher scoring already required by training.
They do not generate trajectories, issue additional scoring requests, load another checkpoint,
or maintain a fixed-prefix evaluation set.

## Enable

For the launchers under `run/`, add to the existing experiment environment or `train.config`:

```bash
OPD_DIAGNOSTICS_COLLECT_RANK_METRICS=True
```

The Hydra equivalent is `distillation.diagnostics.collect_rank_metrics=true`.
The switch defaults to false. Diagnostics must also be enabled through
`distillation.diagnostics.enabled=true` (already the default in the common launcher).
The shared controls remain:

```bash
OPD_DIAGNOSTICS_TOPK=16
OPD_DIAGNOSTICS_LOG_INTERVAL=1
OPD_DIAGNOSTICS_TOPK_LOG_INTERVAL=10
OPD_DIAGNOSTICS_POSITION_BIN_SIZE=512
OPD_DIAGNOSTICS_ASYNC_METRICS=True
```

Both logging intervals must divide the source training step. The existing top-k ladder is
used: for example, k=1,2,4,8,16 for a configured k=16. At least two head candidates are
requested in the existing forward pass to determine whether top-1 is unique.

Supported actor engines are FSDP/FSDP2, including the existing NPU path. Actor
`use_fused_kernels` and rollout-correction `bypass_mode` must be false because these
paths do not expose the required student top-k scores. Enabling an unsupported
combination raises a configuration error; the feature does not change that configuration.

## Candidate sets and metrics

Let S_k and T_k be the true student and teacher top-k token-ID lists at the same causal
prefix. Candidate scores are aligned **by token ID**, not by list index. No metric uses
the inference backend's `rank` field: targeted scoring may use that field for output slots.

| Basis in the metric name | Metrics | Interpretation |
|---|---|---|
| `lists` | `top1_agreement`, `top1_unique`, `topk_overlap` | Agreement of the returned top-1 IDs; whether both maxima are unique; size of S_k intersect T_k divided by k |
| `intersection` | `candidate_count`, `student_mass`, `teacher_mass`, `kendall_tau_b`, `spearman_rho` | Ranking and probability mass **only on the intersection** of the two candidate sets |
| `student_topk_exact` | `kendall_tau_b`, `spearman_rho`, `student_mass`, `teacher_mass`, `tv_residual_bucket`, `js_residual_bucket`, `mass_roundoff_corrected` | Exact scores from both models on **every candidate in S_k**; distribution distances additionally include one residual-probability bucket |

The current `only_stu_rethinking_style` training path already obtains exact teacher
scores for student candidates. Diagnostics retain those scores before loss clamping.
Other paths, including Relay and legacy OPD, still report list and intersection
statistics. Without that exact scoring result, `student_topk_exact` has zero valid
counts and no means. Scores are never reconstructed by filling missing candidates
with zero, a minimum log-probability, or a truncated-list approximation.
Relay obtains the diagnostic candidates from its existing teacher trigger-scoring
call and reuses that result when the scored trajectory is final. Its trigger still
reads only teacher top-1 and student top-5.

Kendall tau-b accounts for ties; Spearman uses average ranks for tied values. Fewer
than two shared candidates or a constant ranking makes the correlation undefined.
Such positions are counted as unavailable, not as zero correlation. Top-1 agreement
uses the first ID in each returned descending list; tied heads are explicitly
reported by `top1_unique` and excluded from the conditional scopes below.

For probability distances, the two distributions are the probabilities of the same
student candidate IDs followed by `1 - sum(candidate probabilities)`. Thus TV and JS
are **coarse-grained distances**, not full-vocabulary distances. JS uses natural logs
and lies in [0, ln(2)]. Missing tail mass is not discarded or renormalized away.
Small numerical mass overflow up to 0.001 is normalized back to one and reported by
`mass_roundoff_corrected`; larger overflow is invalid.

Scores use the existing training scoring temperatures. The feature does not force
teacher and student temperatures to match. Positive temperature scaling preserves
rankings, but a temperature difference contributes to probability-distance metrics.

## Sources, denominators, and positions

Each metric is reported under four scopes:

- `all`: all valid response positions, including inserted teacher tokens.
- `student_only`: response positions excluding inserted teacher spans. Student suffix
  tokens remain included even when the training loss mask excludes them.
- `all_top1_agree`: the subset where both top-1 scores are uniquely maximal and the
  top-1 token IDs agree.
- `student_only_top1_agree`: the same condition restricted to student-origin positions.

These are per-run conditional subsets, not matched-prefix subsets across different
training runs. Source masks preserve original response coordinates, including gaps
left by teacher spans and padding. Known nonfinite teacher-score events are excluded
from valid metric counts while remaining in the eligible denominator.

Global metric keys have the form:

```text
opd/rank/{scope}/{basis}/top{k}/{metric}/{stat}
```

`stat` is `sum`, `count`, `eligible_count`, `unavailable_count`, `valid_fraction`, or
`mean`. Means require a positive valid count; valid fractions require a positive
eligible count. Candidate lists with duplicate IDs, invalid scores, or inconsistent
descending order are excluded. Empty scopes retain counts but have no fabricated mean.

Position-bin keys have the form:

```text
opd/rank/{scope}/{basis}/top{k}/{metric}_bin/{start}_{end}/{sum|count|eligible_count}
```

These totals are additive across batches and runs. To aggregate, sum numerators and
valid counts before dividing; do not average bin means or correlate aggregate curves
as a substitute for token-level ranking. Scope sizes are also logged as
`opd/rank/{scope}/token_count`.

## Logging and computational cost

The existing asynchronous diagnostics snapshot, source-step barrier, metric router,
and Parquet schema are reused. In balanced mode, only configured-k student-only
headline metrics and their principal valid fractions are live, together with
`exact_scores_available`, `available_topk`, and `compute_seconds`.
Other k values, conditional scopes, counts, and position bins are archived.
Full mode continues to log all metrics. SwanLab/W&B logging and HF archival follow
the experiment's existing settings; this feature adds no separate upload process.

Candidate comparisons run in CPU chunks. The pairwise element budget is 2^20 and
the maximum chunk length is 4096 tokens; no full-batch [B,T,K,K] ranking tensor is
allocated. Ranking k must not exceed 1024, so even a single position fits that
pairwise budget; the usual k=16 is recommended for inexpensive monitoring.
Existing forward passes may return additional top-k entries, so host
transfer and CPU work can increase even though model-forward/request counts do not.
`opd/rank/compute_seconds` measures ranking computation, while existing diagnostics
and step timing metrics retain their original meanings.

After snapshotting, diagnostic-only exact scores are released and student candidate
tensors are restricted back to the configured loss k before actor training.
This prevents a wider diagnostic k from changing PPO tensor widths or the objective.

## Interpretation

Use intersection correlations together with overlap, candidate count, probability
mass, and valid fractions: high correlation on a small intersection is not strong
agreement of the complete candidate distributions. Small probability differences
can also cause rank reversals; inspect probability distances alongside correlations.

The metrics describe alignment on each method's own training trajectories. They do
not establish fixed-input temporal training stability, teacher correctness, or better
task performance. Those claims require corresponding evaluations.
