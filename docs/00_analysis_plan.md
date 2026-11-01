# Pre-registered Analysis Plan

**Status:** locked before any outcome data was inspected. Committed to git as the first commit in this repository; every analysis file is committed after it. Any deviation from this plan is listed in section 10 and in the memo, with a reason.

**Date written:** 2026-10-30
**Data:** Criteo Uplift Modeling dataset v2.1 (`criteo-research-uplift-v2.1.csv.gz`, Hugging Face mirror `criteo/criteo-uplift`; the original `go.criteo.net` link is dead).

---

## 1. Decision

Choose one policy for the ad campaign:

| Policy | Description |
|---|---|
| (a) Treat all | Show the ad to every eligible user |
| (b) Stop | Show the ad to nobody |
| (c) Target | Show the ad only to the top-ranked users by predicted uplift |

## 2. What is known about the data before analysis (from the dataset card, not from the data)

- About 14M rows, 12 dense anonymized features (`f0`–`f11`), `treatment`, `exposure`, `visit`, `conversion`.
- The documented treatment ratio is 0.85. **This is a claim to verify, not an assumption.**
- The file was **assembled from several incrementality tests** and **sub-sampled non-uniformly** for privacy, so "the original incrementality level cannot be deduced". Consequences, fixed now:
  - Absolute lift and the dollar figures are **illustrative of the method**, not a forecast of Criteo's real revenue.
  - Pooling several tests can create imbalance between arms if the tests had different treatment ratios and different base rates. The covariate-balance check (section 4) is the diagnostic for this.
- `treatment` is **assignment** (eligible to be targeted). `exposure` is whether the user was **actually shown** an ad. Ads only cost money when shown, so cost is tied to `exposure`, while the causal effect is estimated on `treatment` (intention-to-treat).

## 3. Metrics (fixed)

| Type | Metric | Estimand |
|---|---|---|
| **Primary** | Conversion rate | ITT average treatment effect: `p_conv(T=1) − p_conv(T=0)`, in percentage points. Relative lift reported alongside. |
| Secondary | Visit rate | Same estimand on `visit`. |
| Guardrail | Cost per incremental conversion (CPIC) | `C × (exposed users in treatment) / (ATE_conv × N_treatment)` |

The **only confirmatory test** is the primary metric. Everything else is reported with intervals but labeled supporting or exploratory.

## 4. Validity checks (run first)

1. **Sample ratio mismatch (SRM).** Chi-square goodness-of-fit of the arm counts against the intended 85/15 split. Flag if `p < 0.001`. With 14M users, tiny deviations become "significant", so I also report the observed share and its absolute deviation from 0.85.
2. **Covariate balance.** For each of the 12 features: standardized mean difference (SMD) between arms. Flag any `|SMD| > 0.1` (material); also report `|SMD| > 0.02` as a minor imbalance worth noting at this scale.
3. **Joint balance.** A classifier trained to predict `treatment` from the 12 features, scored by cross-validated AUC. A clean randomization gives AUC ≈ 0.50.
4. **Exposure integrity.** Control users should have `exposure = 0`. Report the exposure rate in both arms.

**If a check fails:** results are still reported, but the memo states the failure prominently. The recommendation is then conditioned on robustness checks that adjust for the features (regression adjustment), since the features are pre-treatment.

## 5. Inference (fixed)

- **Primary test:** two-sided two-sample test of proportions with unpooled (Wald) standard error, `α = 0.05`. At this sample size it is numerically equivalent to Welch's t-test on a 0/1 outcome.
- **Confidence intervals:** 95% Wald CI on the absolute difference; delta-method 95% CI on the relative lift.
- **Secondary (visit):** same test; no multiplicity correction, because it is not used to make the decision.

## 6. Power

- **MDE** for the realized design: `(z_{0.975} + z_{0.80}) × sqrt(p_c(1−p_c)(1/n_t + 1/n_c))`, with `p_c` = control conversion rate. Reported in absolute and relative terms, and for visits.
- **Follow-up test size.** The target effect is set in advance as the **economic break-even uplift**: `C × e / V`, where `e` is the exposure rate among treated users. This is the smallest effect worth acting on. Sample size is reported for 80% power, α = 0.05, for 85/15 and 50/50 allocation, and for the targeted segment.

## 7. Variance reduction (CUPED)

The dataset has no pre-experiment outcome, so classic CUPED (pre-period metric as covariate) is not possible. Plan:

- **CUPAC:** the CUPED covariate is a **cross-fitted prediction** of the outcome from the 12 pre-treatment features (2-fold, seed 42, model never sees `treatment`). `θ = Cov(Y, X) / Var(X)` is pooled across arms.
- **Check:** Lin (2013) regression adjustment (OLS with treatment, centered features, and interactions; HC2 standard errors).
- **Reported:** variance reduction (%), CI width before vs after, and the equivalent sample-size saving.

## 8. Heterogeneity and targeting

- **Split (seed 42, stratified on treatment × conversion):** 40% train, 20% validation, 40% test.
  - train: fit the uplift models.
  - validation: choose the targeting fraction `k` for policy (c).
  - test: all reported Qini, decile, and policy numbers. Touched once.
- **Model 1 (primary): T-learner.** Two `HistGradientBoostingClassifier` models (treated / control), each predicting conversion. Uplift score = `p̂₁(x) − p̂₀(x)`. Hyperparameters fixed in advance: `learning_rate=0.1, max_iter=300, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=42`.
- **Model 2 (if time): causal forest** (`econml` `CausalForestDML`) fit on a subsample of train. Exploratory comparison only.
- **Evaluation on test:** Qini curve, Qini coefficient (area between model and random targeting, normalized), uplift by score decile with 95% CIs, and incremental conversions per dollar at each targeting depth.
- **Segment cuts in SQL (DuckDB):** score deciles (primary view). Quartiles of individual features are **exploratory** and labeled so.

## 9. Economics and decision rule

### Cost and value assumptions (stated before seeing results)

The dataset has no cost or revenue. I assume:

| Parameter | Base case | Sensitivity range | Reasoning |
|---|---|---|---|
| `C` = cost per **exposed** user | **$0.10** | $0.01 – $2.00 | Display CPM of about $2–5, roughly 20–50 impressions per exposed user over a campaign |
| `V` = value per conversion | **$25** | $5 – $200 | About $80 average order value at about 30% contribution margin |

Cost per **targeted** user = `C × e(x)`, where `e(x)` is the exposure rate of treated users in that segment. Assigning a user to treatment does not guarantee they see an ad.

### Objective

Expected net incremental profit of a policy:

`Profit = V × (incremental conversions) − C × (expected exposures)`

- (b) Stop: profit = 0 by definition.
- (a) Treat all: `N × (V × ATE − C × e)`.
- (c) Target top `k`: same formula restricted to the top `k` by T-learner score. `k` is chosen on the validation set and evaluated on the test set.

### Rule (applied mechanically)

1. If the 95% lower bound of (a)'s profit and of (c)'s profit are both ≤ 0, recommend **(b) Stop**, or a follow-up test if the point estimates are positive.
2. **(c) beats (a)** only if the users that (c) excludes have negative net profit on the test set, with the 95% CI of that remainder's profit below zero. Otherwise prefer (a), because it is simpler and has no model risk.
3. If the comparison is inconclusive, recommend the policy with the higher point estimate as the interim policy, plus a follow-up A/B test of (c) vs (a), sized in section 6.
4. **Sensitivity:** rerun the rule over the full (C, V) grid and report the regions where each policy wins. Report the break-even ratio `C/V` for each policy.

## 10. Deviations log

Any change to this plan after seeing data is recorded here, with the reason.

| # | Change | Reason |
|---|---|---|
| 1 | **Clarification (2026-10-31, before any validity check was run):** joint-balance check (section 4.3) is scored with 5-fold stratified cross-validated AUC (seed 42) and is **flagged if mean AUC > 0.55**. | Section 4.3 gave the expected value (≈ 0.50) but no failure threshold or fold count. At 14M rows a significance test on AUC would flag trivially small departures, so a practical threshold is fixed in advance instead. |

## 11. Reproducibility

- Seeds: 42 everywhere.
- Pinned dependencies in `requirements.txt`.
- One command, `make all`, runs download → Parquet conversion → notebook execution → figures, results, and memo numbers.
