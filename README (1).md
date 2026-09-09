# Loan Default Risk Scoring API

A machine learning service that predicts the probability that a personal loan
application will default, using only information available at the moment the
application is evaluated. Built as a screening tool to help a risk team
prioritize manual review — not a final approval system.

## Table of Contents

- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Setup](#setup)
- [Retraining the Model](#retraining-the-model)
- [Running the API](#running-the-api)
- [Example Requests](#example-requests)
- [Running Tests](#running-tests)
- [Docker](#docker)
- [Leakage Analysis](#leakage-analysis)
- [Model Selection](#model-selection)
- [Evaluation](#evaluation)
- [Probability Calibration](#probability-calibration)
- [Threshold Selection](#threshold-selection)
- [Known Limitations](#known-limitations)
- [Before Production](#before-production)
- [Required Documentation Questions (Q1–Q8)](#required-documentation-questions-q1q8)

## Dataset

**Source:** [Lending Club Loan Data — Kaggle](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv),
originally issued by Lending Club, a peer-to-peer lending platform. Covers
loans issued 2007–2015 (~890,000 rows, 75+ columns).

**Sampling:** the raw file (2,260,668 × 145 after the platform's own
expansions) was reduced to a labeled set (1,303,638 × 146) after target
construction, then a **stratified random sample of 100,004 rows** (seed=42)
was drawn, preserving the original class balance. This size was chosen to
keep the full pipeline (EDA → feature engineering → tuning → evaluation)
tractable within the assessment time budget while remaining large enough for
stable validation/test metrics.

**Target construction:** loans in non-terminal states (`Current`, `Late`,
`In Grace Period`) were dropped — they haven't reached an outcome yet, and
including them would mix resolved and unresolved loans under one label. This
was confirmed with an empirical check: comparing the monthly distribution of
`issue_d` across loan_status values showed the expected right-censoring
pattern for recently-issued, still-open loans. Label 0 = `Fully Paid`
(79.93%); label 1 = `Charged Off` merged with `Default` (20.07% combined —
the 31 `Default` rows were merged in because they are a transitional
pre-charge-off state, not a distinct outcome). The "does not meet credit
policy" cohort (2,749 rows, only present 2007–2010) was excluded as a
separate underwriting regime — `policy_code` was always 1 in this data, so
it could not be used to validate the split.

## Project Pipeline

```
raw CSV (2,260,668 × 145)
 → target construction               → loan_data_labeled.csv (1,303,638 × 146)
 → stratified sample (seed=42)       → loan_data_sample_100k.csv (100,004 × 146)
 → leakage/scope removal              → loan_data_clean_features.csv (100,004 × 90)
 → 70/15/15 stratified split          → split_train / split_val / split_test
 → preprocessing (fit on train only)  → 236 columns after encoding, phase7_preprocessor.joblib
 → baseline models (LR + RF)
 → hyperparameter tuning (LR, RF, HistGradientBoosting) → phase9_tuned_logistic_regression.joblib
 → calibration (CalibratedClassifierCV, sigmoid, cv=5, fit on train only) → phase9_calibrated_logistic_regression.joblib
 → threshold selection → 0.10 (on calibrated probabilities)
 → FastAPI service (main.py)
 → final evaluation on held-out test set
```

## Setup

```bash
# clone the repository
git clone <repo-url>
cd api_service

# create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

Trained artifacts (`phase7_preprocessor.joblib`,
`phase9_calibrated_logistic_regression.joblib`) are included in the repo /
regenerable via the retraining steps below. Raw data is **not** committed —
download it from the [Kaggle dataset page](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)
and place it as described in the training notebook/scripts.

## Retraining the Model

1. Download the raw Lending Club CSV from Kaggle and place it at the path
   expected by the training scripts/notebooks (see inline comments there).
2. Run the pipeline stages in order: target construction → sampling →
   leakage/feature removal → train/val/test split → preprocessing fit →
   baseline models → hyperparameter tuning → calibration.
3. This regenerates `phase7_preprocessor.joblib` and
   `phase9_calibrated_logistic_regression.joblib`, which the API loads
   directly (no retraining happens at request time).

## Running the API

Locally (after `pip install -r requirements.txt`):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or via Docker — see [Docker](#docker) below.

Once running:
- `GET /health` — liveness/readiness check
- `GET /model-info` — model type, version, and metadata
- `GET /required-fields` — the exact input schema the model expects
- `POST /predict` — submit applicant/loan data, get back a probability and decision

## Example Requests

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model_loaded": true}
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{ "...": "see /required-fields for the full input schema" }'
```
```json
{
  "default_probability": 0.27,
  "risk_tier": "Medium",
  "threshold": 0.10
}
```

Malformed or out-of-range input returns a `4xx` response with a descriptive
error body instead of a stack trace or a silent bad prediction.

## Running Tests

```bash
pytest
```

Test coverage includes:
- **ML pipeline:** preprocessing output shape/dtype checks, edge-case inputs
  (missing fields, out-of-range values), and consistency of predictions
  between a freshly-fit and a saved-and-reloaded model.
- **API:** valid requests return well-formed `200` responses with the
  expected schema; invalid/malformed requests return appropriate `4xx`
  errors rather than crashing the service.

## Docker

```bash
cd api_service
docker build -t loan-risk-api .
docker run -p 8000:8000 loan-risk-api
```

Then verify with `curl http://localhost:8000/health` — a `200` response with
`"model_loaded": true` confirms the model and preprocessor loaded correctly
inside the container. This was tested end-to-end: `docker build` completes
cleanly, `docker run` starts the service, loads the model and preprocessor
(88 input columns after preprocessing), and `/health` returns
`{"status":"ok","model_loaded":true}`.

You'll see repeated `InconsistentVersionWarning` messages from scikit-learn
on startup — see [Known Limitations](#known-limitations) for why, and the
fix that's pinned in `requirements.txt` going forward.

## Leakage Analysis

The raw dataset mixes application-time fields with loan-performance fields
in the same table with no explicit flag distinguishing them. Columns were
audited one by one against a single test: **would this value actually exist
at the moment the application is being scored, before any decision has been
made or any payment has occurred?**

**Excluded as leakage:** `int_rate`, `grade`, `sub_grade`, and `installment`.
These are not raw applicant inputs — they are the **output of Lending Club's
own internal risk model**, assigned *after* the credit decision is made.
Including them would mean training our model to partially reproduce (and
depend on) a different, undisclosed risk model's decision, which:
- inflates offline metrics, since a downstream artifact of the outcome is
  being used to predict the outcome, and
- would be **impossible to reproduce in production** for a brand-new
  application, since the internal grade doesn't exist until after our model
  would need to run.

Other loan-performance-only fields (payment history, recoveries, last
payment amounts, hardship/settlement fields, and similar) were excluded on
the same "not known yet" basis.

**Kept despite looking suspicious at first glance:** the `sec_app_*` and
`*_joint` fields (co-applicant/joint-application details). These have
complex-sounding names but were verified empirically to carry real signal
**only** for `application_type = Joint App` rows and to be legitimately
available at application time (a joint applicant's information is submitted
with the application, not derived from loan performance) — so they were
retained rather than dropped by name-pattern alone.

`credit_history_length_months` (derived from `issue_d - earliest_cr_line`,
both known at application time) was engineered to replace the two raw date
columns directly, since raw calendar dates generalize poorly and can proxy
for "when this loan was issued" rather than a meaningful borrower
characteristic.

A counter-intuitive finding surfaced during this analysis:
`verification_status = Verified` loans defaulted at **23.8%**, versus
**14.8%** for `Not Verified` — the opposite of what "verification reduces
risk" would suggest. The likely explanation is that Lending Club selectively
verifies applications it already judges riskier, meaning this field is a
**proxy for pre-existing risk assessment**, not a risk-reducing intervention
in itself. It was kept as a feature (it's legitimately known at application
time) but interpreted carefully rather than taken at face value.

## Model Selection

Two model families were trained and tuned: Logistic Regression and Random
Forest, plus HistGradientBoosting during hyperparameter tuning.

| Model | Val ROC-AUC | Train/Val gap |
|---|---|---|
| Logistic Regression (baseline) | 0.7100 | 0.0085 |
| Random Forest (baseline) | — | 0.1319 (severe overfitting) |
| Logistic Regression (tuned) | 0.7100 | — |
| Random Forest (tuned) | 0.6918 | — |
| HistGradientBoosting (tuned) | 0.7092 | — |

**Final choice: Logistic Regression.** It matched or beat the more complex
alternatives on validation ROC-AUC while having a far smaller train/val gap
than Random Forest, meaning it generalizes more reliably rather than
memorizing the training set. It's also directly interpretable (coefficients
map to concrete risk factors, useful for both the review requirement in
section 2 and regulatory/explainability expectations in real lending), and
cheaper to serve and retrain. Given that even the more flexible
HistGradientBoosting model didn't meaningfully beat it, the added complexity
of a tree ensemble wasn't justified — the ~0.71 ROC-AUC ceiling appears to
be a property of this feature set rather than of model capacity.

(Hyperparameter search itself used a 20k-row subsample of the training data
for the search phase, given hardware constraints, with `lbfgs` as the solver
for Logistic Regression instead of `saga`.)

## Evaluation

Accuracy is not used as the primary metric — see [Q3](#required-documentation-questions-q1q8)
below. Instead, ROC-AUC, PR-AUC, precision, recall, F1, and the confusion
matrix at the operating threshold are all reported.

**Official final test set results** (calibrated Logistic Regression,
threshold = 0.10 — this is the configuration actually served by the API):

| Metric | Value |
|---|---|
| ROC-AUC | 0.7252 |
| PR-AUC | 0.3977 |
| Brier score | 0.1435 |
| Precision | 0.2354 |
| Recall | 0.9518 |
| F1 | 0.3774 |

Confusion matrix (15,001 test rows): TN=2679, FP=9311, FN=145, TP=2866.

ROC-AUC on the untouched test set (0.7252) is in line with — in fact
slightly better than — the validation ROC-AUC seen during tuning (0.7100
for the original uncalibrated model), confirming the model generalizes well
and that calibration did not degrade discriminative ability. Of the 3,011
loans that actually defaulted in the test set, only 145 were missed
(recall ≈ 95%) — the intended effect of a threshold derived from a cost
structure where missed defaults are far more expensive than false alarms.

For reference, the original (pre-calibration) tuned model at its original
threshold of 0.35 scored: ROC-AUC 0.7087, PR-AUC 0.3807, Precision 0.2575,
Recall 0.8831, F1 0.3988, with confusion matrix TN=4324, FP=7666, FN=352,
TP=2659. That configuration is **not** the one deployed — it's kept here
only to show the calibration + threshold change was evaluated against the
prior baseline, and both were checked on the same held-out test set.

## Probability Calibration

Initial check (reliability diagram + Brier score on the validation set)
showed the tuned model was **not** well-calibrated: predicted probabilities
were systematically too high across the board (e.g., ~50% predicted
corresponded to an actual default rate of ~23%), and the Brier score
(0.2153) was worse than a naive constant-prediction baseline (~0.16 for this
class balance). This is consistent with the effect of class rebalancing
during training, which improves ranking/discrimination but distorts the
probability scale.

**Fix:** the tuned Logistic Regression was refit inside
`CalibratedClassifierCV` (`method="sigmoid"`, `cv=5`), using **only the
training split** — validation and test data were never used to fit the
calibration, preserving their integrity for threshold selection and final
evaluation. This improved the Brier score to **0.1468**, and the reliability
diagram now closely tracks the ideal diagonal. See
[Q6](#required-documentation-questions-q1q8) for the full answer.

## Threshold Selection

With calibrated probabilities, the threshold was re-derived analytically
from the business cost structure (section 2 of the assignment): a missed
default (false negative) costs roughly the full outstanding principal, while
a false positive costs only the foregone profit margin and some goodwill —
"roughly an order of magnitude smaller." Treating this as a ~10:1 cost
ratio, the cost-minimizing threshold on calibrated probabilities is:

```
t* = C_FP / (C_FP + C_FN) ≈ 1 / (1 + 10) ≈ 0.09
```

The nearest evaluated point on a validation threshold scan, **t = 0.10**,
was adopted, then confirmed on the held-out test set: Precision = 0.2354,
Recall = 0.9518, F1 = 0.3774. This means the model catches about 95% of
actual defaults (missing only 145 of 3,011 defaulted loans in the test
set), at the cost of flagging a large number of applications that would not
have defaulted — a defensible trade given that missing a default is far
more expensive than an unnecessary manual review. See
[Q5](#required-documentation-questions-q1q8) for the full reasoning.

## Known Limitations

- **Performance ceiling:** ROC-AUC caps around 0.71 with this feature set;
  more complex models did not improve on it, suggesting the ceiling comes
  from the available application-time signal, not from under-fitting.
- **High false-positive volume at the chosen threshold:** by design (given
  the cost asymmetry), the 0.10 threshold flags a large share of applicants
  for review to avoid missing defaults; this needs to be weighed against
  the review team's actual capacity in a real deployment.
- **scikit-learn version drift:** trained artifacts were serialized with
  scikit-learn 1.9.0; environments running 1.8.0 will show
  `InconsistentVersionWarning` on load. `requirements.txt` pins
  `scikit-learn==1.9.0` to prevent this going forward, but this class of
  issue (silent behavior drift across library versions) is a real
  production risk if not enforced.
- **Unknown categories at inference:** `OneHotEncoder(handle_unknown="ignore")`
  handles categories unseen during training by encoding them as all-zero,
  which was observed to occur in practice (a value in the test set that
  train hadn't seen) and did not break inference — but it also means the
  model silently treats a genuinely novel category the same as "none of the
  known categories," with no explicit signal that this happened.
- **Sample size:** trained on a 100k-row stratified sample rather than the
  full ~890k-row dataset, for tractability within the project's time
  budget.

## Before Production

- Re-validate calibration and the threshold periodically against live
  outcome data — the cost ratio (10:1) used here was a reasonable reading
  of the stated business scenario, not a measured number, and should be
  replaced with the business's actual figures if available.
- Add monitoring for input distribution drift (e.g., applicant profiles
  changing over time in ways the training data doesn't reflect) and for the
  `OneHotEncoder` unknown-category rate specifically.
- Add authentication/rate-limiting to the API before exposing it beyond an
  internal network.
- Retrain on the full dataset (or a larger sample) if the 0.71 ROC-AUC
  ceiling needs to be pushed further and more compute/time is available.
- Set up a proper CI pipeline to enforce the scikit-learn version pin and
  run the test suite on every change (explicitly out of scope for this
  assignment, but a real gap before production).

## Required Documentation Questions (Q1–Q8)

**Q1. Which features did you exclude as leakage, and how did you decide?**
`int_rate`, `grade`, `sub_grade`, and `installment` were excluded, along
with other loan-performance-only fields (payment history, recoveries,
hardship/settlement data, etc.). The test applied to every column was
whether its value would exist at the moment the application is evaluated,
before any credit decision or repayment activity — not just whether the
column "sounds" like performance data. `int_rate`/`grade`/`sub_grade`/
`installment` failed this test specifically because they are Lending Club's
own internal risk-model output, assigned after the decision, not
applicant-provided information.

**Q2. Which features did you keep that a careless approach might have
excluded, and why did you keep them?**
`sec_app_*` and `*_joint` fields — their names suggest complexity or a
possible leakage risk, but they were empirically verified to carry
legitimate application-time signal only for joint applications, and were
retained rather than dropped by pattern-matching on the name alone.
`verification_status` was also kept despite its counter-intuitive
relationship with default risk (verified loans defaulted *more* often),
because it is genuinely known at application time — it was interpreted
carefully (as a proxy for Lending Club's own pre-screening) rather than
excluded for looking risk-reducing on its face.

**Q3. Why is accuracy insufficient as your primary metric for this problem,
specifically?**
The target is imbalanced (~80% fully paid / ~20% default). A model that
always predicts "fully paid" would score ~80% accuracy while catching zero
actual defaults — precisely the failure mode the business cares most about
avoiding, since missed defaults are the costly error. Accuracy treats both
error types as equally bad and rewards the majority-class shortcut, which
is the opposite of what this screening tool needs to optimize for.

**Q4. Which metric(s) did you prioritize, and how does that choice connect
to the business cost structure?**
Recall (and PR-AUC, given the class imbalance) were prioritized over
precision or accuracy, because the cost structure in section 2 makes false
negatives (missed defaults, costing ~full principal) roughly an order of
magnitude more expensive than false positives (an unnecessary flag for
review, costing only margin/goodwill). ROC-AUC and PR-AUC were used to
compare models' overall discriminative ability during selection; recall at
the operating threshold was used to check that the deployed decision point
actually reflects that cost asymmetry.

**Q5. What threshold did you choose, and what's the concrete reasoning
connecting it to the stated costs?**
Threshold = 0.10, applied to **calibrated** probabilities. Given
`C_FN ≈ 10 × C_FP` (from "roughly an order of magnitude" in the business
scenario), the cost-minimizing threshold is `t* = C_FP/(C_FP+C_FN) ≈ 0.09`;
0.10 is the nearest evaluated point on a validation threshold scan, confirmed
on the held-out test set with Recall = 0.9518 and Precision = 0.2354. This
threshold could only be derived this way *after* calibration — see Q6.

**Q6. Are your model's probabilities well-calibrated? How did you check,
and what did you conclude?**
Initially, no. A reliability diagram and Brier score on the validation set
showed the tuned model systematically overpredicted default probability
(e.g., predicting ~50% where the actual rate was ~23%), with a Brier score
(0.2153) worse than a naive baseline — likely caused by class rebalancing
during training, which helps ranking but distorts the probability scale.
This was fixed by refitting the model inside `CalibratedClassifierCV`
(sigmoid/Platt scaling, 5-fold CV) using only the training split, which
brought the Brier score down to 0.1468 and the reliability curve close to
the ideal diagonal. The calibrated model's probabilities are now reliable
enough to support a cost-based threshold derivation (Q5), which the
original probabilities were not.

**Q7. How did you validate that your evaluation numbers aren't inflated by
leakage or by improper preprocessing?**
Three checks: (1) the leakage audit in Q1/Q2 removed every column that
wouldn't exist at scoring time, and the resulting ROC-AUC (~0.71) is
moderate rather than suspiciously high — a near-perfect score would itself
have been a red flag for missed leakage; (2) all preprocessing (imputation,
encoding, scaling) was fit exclusively on the training split via a
pipeline, then only `.transform()`-applied to validation and test, so no
information from held-out data influenced preprocessing decisions; (3) the
calibration step was fit only on the training split (via cross-validation),
explicitly to avoid contaminating the validation set that had already been
used, and would still be needed, for threshold selection. The close
val/test agreement (ROC-AUC 0.7100 vs 0.7087) is consistent with a model
that isn't overfit to information it shouldn't have had.

**Q8. What is the single biggest weakness of your current solution, and
what would you do about it with two more weeks?**
The biggest weakness is the ~0.71 ROC-AUC ceiling — the model's ranking
ability, not just its probability calibration, is capped by the available
application-time feature set, and no amount of further tuning or
recalibration changes that. With two more weeks, the priority would be
feature engineering rather than more modeling: revisiting fields that were
conservatively dropped during the leakage audit to see if any have a
legitimately application-time-safe transformation (e.g., aggregating
`sec_app_*`/`*_joint` fields more richly, or deriving ratios/interactions
from the existing 89 features), and testing on the full ~890k-row dataset
instead of the 100k sample to see whether the ceiling is a sample-size
artifact or a genuine information limit.
