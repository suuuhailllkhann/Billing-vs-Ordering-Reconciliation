# Phase 2 ML methodology decision log

This document records the approved preparation contract for the synthetic
`prescription_renewal_within_window` problem. No model is trained in this phase.

## Prediction population and target

The source is the leakage-reviewed `refill_observations.csv`. Each row represents a
zero-refill prescription approaching expected supply exhaustion. The binary target is
`prescription_renewal_within_window`; its construction remains defined in the Phase 2A
data contract.

## Fixed chronological split

| Partition | Observation-date rule | Permitted use |
|---|---|---|
| Train | Before 2026-02-01 | Fit preprocessing; future time-aware CV and tuning |
| Validation | 2026-02-01 through 2026-03-31 | Future model/threshold selection |
| Test | 2026-04-01 or later | One final evaluation only |

Rows are sorted chronologically and are never randomly split. A chronological split
better approximates production, where a model learns from past observations and is
applied to later patients. It also prevents later outcome patterns from influencing
earlier training decisions. Future tuning must use time-aware cross-validation inside
Train only. Validation and Test must not participate in preprocessing fit, tuning, or
feature selection. Test must also remain untouched during model and threshold choice.

Repeated patients may appear across periods. This is intentional for the primary
production-like evaluation because a deployed pharmacy system will encounter both
returning and new patients. `patient_id` is excluded from modeling. Final testing will
report overall performance plus separate previously-seen-patient and unseen-patient
performance; those subgroup labels are evaluation metadata, not features.

## Modeling schema

The target is not a feature. These audit/identity fields are excluded:

- `observation_id`
- `observation_date`
- `expected_supply_end_date`
- `patient_id`
- `medication_id`
- `ndc`
- `prescription_id`
- `current_refills_remaining`

Identifiers are excluded because they are arbitrary labels that encourage memorization
and do not represent portable clinical or operational behavior. Dates used to define
the split and target horizon are excluded to avoid shortcut learning. The zero-refill
field defines eligibility and is constant in the modeling population.

All remaining legitimate numeric columns are initial candidate features. No feature
selection has been performed. The hidden synthetic `behavior_profile` is not present
in the observation dataset and must never be joined into modeling data.

## Insufficient refill-interval history

A missing `std_previous_refill_interval_days` means fewer than two prior intervals,
not corrupt data. Preparation therefore adds `refill_interval_std_available`:

- `1`: the raw standard deviation is available.
- `0`: insufficient history; the raw value was missing.

The missing numeric value is replaced by the median calculated from Train only. The
same frozen median is then applied to Validation and Test. This preserves the useful
missingness distinction while preventing later-period distributions from leaking into
training. Any future cross-validation must refit this median separately inside each
training fold.

## Planned model-specific preprocessing

- Logistic Regression: train-only median imputation, availability indicator, then a
  `StandardScaler` fitted on Train (or the training fold) only.
- Random Forest: train-only median imputation and indicator; no scaling.
- XGBoost: train-only missing handling and indicator; no scaling.
- LightGBM: train-only missing handling and indicator; no scaling.

The current dependency-light foundation implements the shared median-plus-indicator
transformation and records whether scaling is required. Actual library pipelines,
estimators, tuning, and scaler fitting are deferred until model implementation.

## Leakage controls

- Split before fitting any learned preprocessing value.
- Fit the median from Train only and freeze it for later periods.
- Exclude identifiers, eligibility fields, target-construction dates, future Rx data,
  and hidden synthetic profiles.
- Never randomly split.
- Reserve Test from model selection, tuning, feature selection, and threshold choice.
- Use time-aware CV within Train for future tuning.
- Fit every preprocessing step independently within each future CV training fold.

## Phase 2C controlled baseline comparison

Four complementary baselines were selected without tuning: logistic regression is an
interpretable linear reference; random forest represents bagged nonlinear trees;
XGBoost and LightGBM represent two widely used gradient-boosted tree implementations.
This comparison checks whether the synthetic behavioral signal is primarily linear or
benefits from nonlinear interactions. It does not select a final model.

All models fit Train only and are evaluated on Train and Validation. Test is not passed
to the comparison interface. Classification metrics use a fixed probability threshold
of 0.50. This is only a conventional baseline reference—not an operational threshold,
and no threshold analysis or optimization has occurred.

### Exact configurations

- Logistic Regression: Train-fitted median/indicator, `StandardScaler`, L2 via
  `l1_ratio=0.0`, `C=1.0`, `solver=lbfgs`, `max_iter=2000`, no class weight,
  `random_state=20260814`.
- Random Forest: 300 trees, otherwise default-like settings, no class weight,
  `random_state=20260814`, `n_jobs=1`.
- XGBoost: 300 estimators, learning rate 0.05, maximum depth 3, subsample 1.0,
  column sample 1.0, histogram tree method, log-loss evaluation, no class weighting,
  `random_state=20260814`, `n_jobs=1`.
- LightGBM: 300 estimators, learning rate 0.05, 31 leaves, unlimited maximum depth,
  no class weight, deterministic column-wise execution, `random_state=20260814`,
  `n_jobs=1`.

### Train metrics

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6925 | 0.7045 | 0.8299 | 0.7621 | 0.7447 | 0.7086 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 0.7730 | 0.7604 | 0.9017 | 0.8250 | 0.9085 | 0.8698 |
| LightGBM | 0.9936 | 0.9893 | 1.0000 | 0.9946 | 0.9999 | 0.9998 |

### Validation metrics

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6927 | 0.7057 | 0.8755 | 0.7815 | 0.7759 | 0.6897 |
| Random Forest | 0.6589 | 0.7165 | 0.7552 | 0.7354 | 0.7484 | 0.6629 |
| XGBoost | 0.6901 | 0.7276 | 0.8091 | 0.7662 | 0.7526 | 0.6810 |
| LightGBM | 0.6719 | 0.7309 | 0.7552 | 0.7429 | 0.7439 | 0.6555 |

Validation confusion matrices use `[[TN, FP], [FN, TP]]`:

- Logistic Regression: `[[55, 88], [30, 211]]`
- Random Forest: `[[71, 72], [59, 182]]`
- XGBoost: `[[70, 73], [46, 195]]`
- LightGBM: `[[76, 67], [59, 182]]`

Random Forest and LightGBM nearly memorize Train but lose substantial performance on
Validation, which is an obvious overfitting warning under these defaults. XGBoost has
a smaller but still material Train–Validation gap. Logistic Regression has nearly
identical Train and Validation accuracy and the strongest Validation PR-AUC/F1 in this
single untuned comparison, suggesting useful linear signal without obvious overfit.
These observations motivate later controlled work but do not establish a winner.

## Phase 2D Train-only hyperparameter tuning

Each model used `RandomizedSearchCV` with the fixed seed `20260814`, scoring
`average_precision`, and `TimeSeriesSplit(n_splits=5)` over Train only. Expanding
windows preserve temporal order and better represent repeated refitting on historical
data than shuffled CV. PR-AUC was optimized because renewal review is a positive-class
ranking problem and PR-AUC is more informative than accuracy when class prevalence or
operational review capacity changes.

The missing-history median/indicator transformer is inside every pipeline, so it is
refitted inside each CV training fold. Logistic scaling is also inside its pipeline.
Best estimators were refitted on full Train and evaluated on Validation at the unchanged
0.50 reference threshold. Test was not passed to tuning or evaluation.

### Trial counts and unchanged search spaces

- Logistic Regression, 10 trials: `C=[0.01,0.1,0.5,1,2,5,10]`, L1/L2. With
  scikit-learn 1.9, `l1_ratio=[1.0,0.0]` is the non-deprecated exact L1/L2 mapping;
  SAGA and `max_iter=5000` were fixed.
- Random Forest, 30 trials: `n_estimators=[200,400,600]`,
  `max_depth=[3,5,8,12,None]`, `min_samples_split=[2,5,10,20]`,
  `min_samples_leaf=[1,2,5,10]`, `max_features=[sqrt,log2,0.5,1.0]`.
- XGBoost, 40 trials: `n_estimators=[150,250,400,600]`,
  `learning_rate=[0.02,0.05,0.1]`, `max_depth=[2,3,4,5]`,
  `min_child_weight=[1,3,5,10]`, `subsample=[0.7,0.85,1.0]`,
  `colsample_bytree=[0.7,0.85,1.0]`, `reg_alpha=[0,0.01,0.1,0.5]`,
  `reg_lambda=[1,2,5,10]`.
- LightGBM, 40 trials: `n_estimators=[150,250,400,600]`,
  `learning_rate=[0.02,0.05,0.1]`, `num_leaves=[7,15,31,63]`,
  `max_depth=[3,5,8,-1]`, `min_child_samples=[10,20,40,80]`,
  `subsample=[0.7,0.85,1.0]`, `colsample_bytree=[0.7,0.85,1.0]`,
  `reg_alpha=[0,0.01,0.1,0.5]`, `reg_lambda=[0,1,5,10]`.

No search space was expanded.

### Best CV results and parameters

| Model | Best CV PR-AUC | Best parameters |
|---|---:|---|
| Logistic Regression | 0.7292 | `C=0.01`, L2 |
| Random Forest | 0.7502 | 200 trees, depth 12, split 20, leaf 2, sqrt features |
| XGBoost | 0.7404 | 400 estimators, LR 0.02, depth 4, child weight 1, subsample 1.0, columns 0.85, alpha 0.5, lambda 10 |
| LightGBM | 0.7398 | 250 estimators, LR 0.05, 7 leaves, depth 5, child samples 80, subsample 0.85, columns 0.85, alpha 0.5, lambda 1 |

### Tuned Train metrics

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6849 | 0.6891 | 0.8545 | 0.7629 | 0.7361 | 0.7000 |
| Random Forest | 0.8180 | 0.8010 | 0.9223 | 0.8574 | 0.9481 | 0.9201 |
| XGBoost | 0.7579 | 0.7467 | 0.8958 | 0.8145 | 0.8848 | 0.8427 |
| LightGBM | 0.7520 | 0.7492 | 0.8751 | 0.8073 | 0.8832 | 0.8370 |

### Tuned Validation metrics

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7005 | 0.7045 | 0.9004 | 0.7905 | 0.7692 | 0.6837 |
| Random Forest | 0.6797 | 0.7201 | 0.8008 | 0.7583 | 0.7699 | 0.6849 |
| XGBoost | 0.6979 | 0.7289 | 0.8257 | 0.7743 | 0.7730 | 0.7022 |
| LightGBM | 0.6719 | 0.7186 | 0.7842 | 0.7500 | 0.7515 | 0.6878 |

Validation confusion matrices (`[[TN, FP], [FN, TP]]`):

- Logistic Regression: `[[52,91],[24,217]]`
- Random Forest: `[[68,75],[48,193]]`
- XGBoost: `[[69,74],[42,199]]`
- LightGBM: `[[69,74],[52,189]]`

Regularization reduced the extreme Random Forest and LightGBM Train memorization, but
Random Forest retains the largest Train–Validation PR-AUC gap. XGBoost and LightGBM
also retain moderate gaps. Logistic Regression remains stable and high-recall, though
its tuned Validation PR-AUC is slightly below its baseline. Validation PR-AUC improved
for Random Forest, XGBoost, and LightGBM. These results remain comparison evidence only;
no winner, feature set, threshold, or final estimator has been selected.

## Feature-analysis stage: tuned Logistic Regression coefficients

The Phase 2D tuned Logistic Regression was refitted on Train only with the unchanged
configuration (`C=0.01`, L2/SAGA, `max_iter=5000`, seed `20260814`). The existing
fold-safe refill-interval median/availability transformation was fitted on Train, then
`StandardScaler` was fitted on Train before the classifier. Validation and Test were
not evaluated or used in this analysis.

Coefficients below are changes in model log-odds per one standard deviation increase
in a final model input, conditional on the other included inputs. They are sorted by
absolute magnitude.

| Feature | Coefficient | Absolute coefficient | Direction |
|---|---:|---:|---|
| `previous_on_time_fill_rate` | 0.49936279 | 0.49936279 | positive |
| `average_previous_timing_gap_days` | -0.13421066 | 0.13421066 | negative |
| `previous_fill_count` | 0.11387149 | 0.11387149 | positive |
| `previous_early_fill_rate` | -0.06458892 | 0.06458892 | negative |
| `latest_refill_timing_gap_days` | -0.05895071 | 0.05895071 | negative |
| `refill_interval_std_available` | -0.05106063 | 0.05106063 | negative |
| `medication_prior_fill_count` | 0.04926908 | 0.04926908 | positive |
| `average_previous_refill_interval_days` | -0.04845641 | 0.04845641 | negative |
| `medication_prior_average_days_supply` | 0.04333492 | 0.04333492 | positive |
| `current_refill_number` | -0.01875323 | 0.01875323 | negative |
| `current_quantity_billed` | 0.01754541 | 0.01754541 | positive |
| `medication_prior_average_quantity` | -0.01658121 | 0.01658121 | negative |
| `std_previous_refill_interval_days` | -0.01228752 | 0.01228752 | negative |
| `current_days_supply` | 0.00956247 | 0.00956247 | positive |
| `days_since_previous_fill` | -0.00800432 | 0.00800432 | negative |
| `prescription_age_days` | -0.00028218 | 0.00028218 | negative |

`previous_on_time_fill_rate` is the strongest positive coefficient by a wide margin;
`previous_fill_count` is the next strongest positive. The strongest negative
coefficients are `average_previous_timing_gap_days`, `previous_early_fill_rate`, and
`latest_refill_timing_gap_days`. `prescription_age_days`,
`days_since_previous_fill`, and `current_days_supply` are closest to zero.

The on-time and timing-gap variables point in the expected synthetic direction:
historically on-time behavior increases the fitted renewal score, while later average
and latest timing gaps decrease it. `std_previous_refill_interval_days` has a small
negative coefficient and its availability indicator has a larger negative coefficient;
these two terms jointly represent variability and whether enough history existed to
calculate it, so neither should be interpreted alone.

Coefficient magnitude is not a feature-selection rule. Correlated inputs can divide or
share a common effect, suppress one another, or change conditional signs. This warning
is especially relevant to the refill-count/history variables, the average/latest timing
variables and timing rates, the interval/day-supply variables, and the quantity/day-
supply summaries previously identified as highly correlated. No feature is marked for
removal from this analysis.

### Post-Test standardized odds-ratio interpretation

The locked Train-only Logistic Regression coefficients reproduced the earlier analysis
within `5e-8`. Because `StandardScaler` is in the pipeline, each odds ratio is
`exp(coefficient)` and represents the model's approximate multiplicative change in odds
for a one-standard-deviation increase in that feature, holding other included features
constant.

| Feature | Standardized coefficient | Odds ratio | Direction |
|---|---:|---:|---|
| `previous_on_time_fill_rate` | 0.49936279 | 1.64767103 | increases predicted renewal odds |
| `average_previous_timing_gap_days` | -0.13421066 | 0.87440584 | decreases predicted renewal odds |
| `previous_fill_count` | 0.11387149 | 1.12060810 | increases predicted renewal odds |
| `previous_early_fill_rate` | -0.06458892 | 0.93745276 | decreases predicted renewal odds |
| `latest_refill_timing_gap_days` | -0.05895071 | 0.94275324 | decreases predicted renewal odds |
| `refill_interval_std_available` | -0.05106063 | 0.95022106 | decreases predicted renewal odds |
| `medication_prior_fill_count` | 0.04926908 | 1.05050298 | increases predicted renewal odds |
| `average_previous_refill_interval_days` | -0.04845641 | 0.95269887 | decreases predicted renewal odds |
| `medication_prior_average_days_supply` | 0.04333492 | 1.04428758 | increases predicted renewal odds |
| `current_refill_number` | -0.01875323 | 0.98142152 | decreases predicted renewal odds |
| `current_quantity_billed` | 0.01754541 | 1.01770024 | increases predicted renewal odds |
| `medication_prior_average_quantity` | -0.01658121 | 0.98355550 | decreases predicted renewal odds |
| `std_previous_refill_interval_days` | -0.01228752 | 0.98778766 | decreases predicted renewal odds |
| `current_days_supply` | 0.00956247 | 1.00960834 | approximately neutral |
| `days_since_previous_fill` | -0.00800432 | 0.99202762 | approximately neutral |
| `prescription_age_days` | -0.00028218 | 0.99971786 | approximately neutral |

The on-time rate has the largest conditional association: one standard deviation higher
corresponds to 1.648 times the model-predicted renewal odds. One standard deviation
higher average timing gap corresponds to 0.874 times the odds, while one standard
deviation more previous fills corresponds to 1.121 times the odds. Early-fill rate,
latest timing gap, and availability of interval variability correspond to odds ratios
of 0.937, 0.943, and 0.950. The interval standard deviation (0.988) must be interpreted
together with its availability indicator because the indicator distinguishes observed
variability from Train-median imputation.

These odds ratios describe the fitted model, not causal effects. They are standardized-
feature odds ratios, and correlated predictors can share, suppress, or alter conditional
effects. Small coefficients do not justify feature removal. No p-values or significance
tests were calculated. The data are synthetic, so these relationships are not evidence
about real patients or pharmacies. This post-Test interpretation did not change the
locked model, features, preprocessing, hyperparameters, or threshold.

### Validation permutation importance

The locked full-feature Logistic Regression and preprocessing were fitted on Train only.
Permutation importance was calculated on the 384 Validation observations—not Test—using
scikit-learn `permutation_importance`, `average_precision` scoring, 30 repeats, seed
`20260814`, and one worker. Importance is the mean decrease in Validation PR-AUC after
permuting one final model input; standard deviation describes variation across repeats.

| Rank | Feature | Mean PR-AUC decrease | Repeat std |
|---:|---|---:|---:|
| 1 | `previous_on_time_fill_rate` | 0.09138744 | 0.02312792 |
| 2 | `previous_fill_count` | 0.02813946 | 0.00729682 |
| 3 | `average_previous_refill_interval_days` | 0.01413606 | 0.00540277 |
| 4 | `average_previous_timing_gap_days` | 0.01141233 | 0.01114648 |
| 5 | `latest_refill_timing_gap_days` | 0.00485516 | 0.00628469 |
| 6 | `medication_prior_fill_count` | 0.00303437 | 0.00252212 |
| 7 | `days_since_previous_fill` | 0.00191247 | 0.00080189 |
| 8 | `medication_prior_average_quantity` | 0.00164921 | 0.00108588 |
| 9 | `previous_early_fill_rate` | 0.00092024 | 0.00413596 |
| 10 | `std_previous_refill_interval_days` | 0.00030537 | 0.00155016 |
| 11 | `prescription_age_days` | 0.00008994 | 0.00011666 |
| 12 | `current_days_supply` | -0.00023618 | 0.00104775 |
| 13 | `medication_prior_average_days_supply` | -0.00037787 | 0.00252446 |
| 14 | `current_quantity_billed` | -0.00118901 | 0.00224362 |
| 15 | `current_refill_number` | -0.00364157 | 0.00205071 |
| 16 | `refill_interval_std_available` | -0.00477851 | 0.00406820 |

Permutation importance agrees with coefficients and odds ratios that on-time rate is
dominant, previous fill count is influential, and timing-gap history matters. Average
refill interval ranks third despite only the eighth-largest coefficient magnitude,
while early-fill rate and the availability indicator rank much lower than their
coefficient magnitudes. The interval standard deviation and availability indicator must
still be interpreted together. Features with near-zero coefficients mostly have small
permutation effects; `days_since_previous_fill` is a modest exception at rank seven.

The ranking also overlaps tree evidence: on-time rate led Random Forest and XGBoost;
latest timing gap was prominent in both; and refill-interval or medication-history
features received importance across the tree models. Exact ordering differs because
native tree importance, linear coefficients, and PR-AUC permutation degradation measure
different properties.

Correlated predictors can substitute for one another, causing permutation importance to
understate unique contribution when another feature retains similar information.
Negative or near-zero values can arise from sampling variation and correlation and do
not justify removal. The data are synthetic and these patterns are not real pharmacy
behavior. Test was not used, and this analysis did not reopen feature reduction or
produce any model, feature, hyperparameter, or threshold decision.

## Phase 2E tree-model feature importance

The tuned Random Forest, XGBoost, and LightGBM pipelines were refitted on Train only
with their unchanged Phase 2D hyperparameters, features, and Train-fitted missing-value
preprocessing. Validation and Test were not evaluated. Each table uses the estimator's
standard model-native `feature_importances_` attribute:

- Random Forest reports normalized mean decrease in impurity across its trees.
- XGBoost reports normalized average gain, its default sklearn-interface importance.
- LightGBM reports split counts, its default sklearn-interface importance.

Because these definitions and scales differ, ranks are useful for comparing ordering,
but raw importance values must not be compared across model families.

### Random Forest native importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `previous_on_time_fill_rate` | 0.19052349 |
| 2 | `latest_refill_timing_gap_days` | 0.09870261 |
| 3 | `average_previous_timing_gap_days` | 0.08802219 |
| 4 | `medication_prior_average_days_supply` | 0.08412047 |
| 5 | `medication_prior_average_quantity` | 0.08167697 |
| 6 | `medication_prior_fill_count` | 0.08022258 |
| 7 | `average_previous_refill_interval_days` | 0.06852035 |
| 8 | `std_previous_refill_interval_days` | 0.06735741 |
| 9 | `prescription_age_days` | 0.06704079 |
| 10 | `days_since_previous_fill` | 0.05666332 |
| 11 | `previous_fill_count` | 0.03375195 |
| 12 | `previous_early_fill_rate` | 0.02990880 |
| 13 | `current_quantity_billed` | 0.02767489 |
| 14 | `current_refill_number` | 0.01669998 |
| 15 | `current_days_supply` | 0.00622598 |
| 16 | `refill_interval_std_available` | 0.00288822 |

### XGBoost native importance

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `previous_on_time_fill_rate` | 0.34530938 |
| 2 | `latest_refill_timing_gap_days` | 0.10610010 |
| 3 | `days_since_previous_fill` | 0.05481017 |
| 4 | `average_previous_refill_interval_days` | 0.04884994 |
| 5 | `current_refill_number` | 0.04383821 |
| 6 | `medication_prior_average_quantity` | 0.04334843 |
| 7 | `current_quantity_billed` | 0.04151654 |
| 8 | `average_previous_timing_gap_days` | 0.04143182 |
| 9 | `std_previous_refill_interval_days` | 0.04078171 |
| 10 | `prescription_age_days` | 0.04077526 |
| 11 | `medication_prior_average_days_supply` | 0.03915129 |
| 12 | `previous_fill_count` | 0.03915083 |
| 13 | `medication_prior_fill_count` | 0.03740826 |
| 14 | `previous_early_fill_rate` | 0.03244859 |
| 15 | `current_days_supply` | 0.02339118 |
| 16 | `refill_interval_std_available` | 0.02168826 |

### LightGBM native importance

| Rank | Feature | Importance (split count) |
|---:|---|---:|
| 1 | `medication_prior_fill_count` | 214 |
| 2 | `average_previous_refill_interval_days` | 173 |
| 3 | `medication_prior_average_days_supply` | 169 |
| 4 | `std_previous_refill_interval_days` | 146 |
| 5 | `medication_prior_average_quantity` | 143 |
| 6 | `average_previous_timing_gap_days` | 135 |
| 7 | `previous_on_time_fill_rate` | 132 |
| 8 | `prescription_age_days` | 92 |
| 9 | `days_since_previous_fill` | 91 |
| 10 | `latest_refill_timing_gap_days` | 81 |
| 11 | `previous_fill_count` | 48 |
| 12 | `current_refill_number` | 35 |
| 13 | `previous_early_fill_rate` | 19 |
| 14 | `current_quantity_billed` | 9 |
| 15 | `current_days_supply` | 0 |
| 16 | `refill_interval_std_available` | 0 |

### Cross-model rank comparison

| Feature | Random Forest rank | XGBoost rank | LightGBM rank |
|---|---:|---:|---:|
| `previous_on_time_fill_rate` | 1 | 1 | 7 |
| `latest_refill_timing_gap_days` | 2 | 2 | 10 |
| `average_previous_timing_gap_days` | 3 | 8 | 6 |
| `medication_prior_average_days_supply` | 4 | 11 | 3 |
| `medication_prior_average_quantity` | 5 | 6 | 5 |
| `medication_prior_fill_count` | 6 | 13 | 1 |
| `average_previous_refill_interval_days` | 7 | 4 | 2 |
| `std_previous_refill_interval_days` | 8 | 9 | 4 |
| `prescription_age_days` | 9 | 10 | 8 |
| `days_since_previous_fill` | 10 | 3 | 9 |
| `previous_fill_count` | 11 | 12 | 11 |
| `previous_early_fill_rate` | 12 | 14 | 13 |
| `current_quantity_billed` | 13 | 7 | 14 |
| `current_refill_number` | 14 | 5 | 12 |
| `current_days_supply` | 15 | 15 | 15 |
| `refill_interval_std_available` | 16 | 16 | 16 |

Random Forest and XGBoost agree that `previous_on_time_fill_rate` and
`latest_refill_timing_gap_days` are their two leading features. This broadly supports
the Logistic Regression findings, where on-time rate was the strongest positive
coefficient and timing-gap measures were leading negative coefficients. LightGBM uses
the same signal but distributes more splits across medication-level and interval-
history variables. `previous_fill_count` and `previous_early_fill_rate` are below the
top tier in all three tree rankings even though Logistic Regression gives them the
second-largest positive and second-largest timing-rate coefficient magnitudes,
respectively. This difference can reflect nonlinear substitution and shared signal.

Within redundant Group 1, `average_previous_refill_interval_days` generally ranks
above `days_since_previous_fill`, while `current_days_supply` ranks 15th in every tree.
Within Group 2, both medication-level averages rank highly in Random Forest and
LightGBM, with XGBoost giving quantity the higher rank. Within Group 3, Random Forest
ranks latest and average timing gaps second and third; XGBoost emphasizes the latest
gap, while LightGBM emphasizes the average gap.

These model-native importances are evidence only. Impurity and split-based measures
can favor variables that offer more candidate split points, and correlated variables
can divide, substitute for, or mask one another's importance. A zero or low native
importance does not establish that a feature is useless. No feature was removed or
selected at this stage.

## Phase 2F reduced-feature experiment

An experimental 11-feature set was tested after full-feature tuning and interpretation.
The purpose was to determine whether a smaller representation could preserve useful
signal or reduce overfitting; it was not a permanent feature-removal decision. Evidence
from the combined correlation review, standardized Logistic Regression coefficients,
and tree-native importance rankings motivated temporarily excluding:

- `current_quantity_billed`
- `current_days_supply`
- `days_since_previous_fill`
- `previous_early_fill_rate`
- `prescription_age_days`

The retained experimental features were `previous_on_time_fill_rate`,
`latest_refill_timing_gap_days`, `average_previous_timing_gap_days`,
`std_previous_refill_interval_days`, `refill_interval_std_available`,
`previous_fill_count`, `average_previous_refill_interval_days`,
`current_refill_number`, `medication_prior_average_quantity`,
`medication_prior_average_days_supply`, and `medication_prior_fill_count`.

All four models were retuned because changing the feature space can change useful model
capacity and regularization. The experiment reused the unchanged seed, search spaces,
trial counts, fold-local preprocessing, `average_precision` scoring, and
`TimeSeriesSplit(n_splits=5)` on Train only. Best models were refitted on full Train and
evaluated on Validation at the unchanged 0.50 reference threshold. Test was not passed
to the experiment.

### Reduced-feature tuning results

| Model | Best CV PR-AUC | Best parameters |
|---|---:|---|
| Logistic Regression | 0.7350 | `C=0.01`, L2 |
| Random Forest | 0.7584 | 200 trees, depth 12, split 20, leaf 2, sqrt features |
| XGBoost | 0.7451 | 150 estimators, LR 0.10, depth 2, child weight 1, subsample 1.0, columns 0.85, alpha 0.1, lambda 1 |
| LightGBM | 0.7441 | 250 estimators, LR 0.05, 31 leaves, unlimited depth, child samples 80, subsample 0.85, columns 0.85, alpha 0.01, lambda 5 |

### Reduced-feature Train metrics

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6849 | 0.6897 | 0.8525 | 0.7625 | 0.7378 | 0.7013 |
| Random Forest | 0.8121 | 0.7962 | 0.9184 | 0.8530 | 0.9454 | 0.9166 |
| XGBoost | 0.7182 | 0.7225 | 0.8525 | 0.7821 | 0.8454 | 0.7961 |
| LightGBM | 0.8209 | 0.8175 | 0.8987 | 0.8562 | 0.9389 | 0.9100 |

### Reduced-feature Validation metrics

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7005 | 0.7045 | 0.9004 | 0.7905 | 0.7720 | 0.6863 |
| Random Forest | 0.6849 | 0.7239 | 0.8050 | 0.7623 | 0.7584 | 0.6865 |
| XGBoost | 0.7031 | 0.7361 | 0.8216 | 0.7765 | 0.7615 | 0.6934 |
| LightGBM | 0.6693 | 0.7298 | 0.7510 | 0.7403 | 0.7502 | 0.6805 |

Validation confusion matrices (`[[TN, FP], [FN, TP]]`):

- Logistic Regression: `[[52,91],[24,217]]`
- Random Forest: `[[69,74],[47,194]]`
- XGBoost: `[[72,71],[43,198]]`
- LightGBM: `[[76,67],[60,181]]`

### Full versus reduced Validation deltas

Each delta is reduced minus full; positive is an increase, not an automatic indication
that the reduced feature set is preferable.

| Model | Recall delta | Precision delta | F1 delta | PR-AUC delta | ROC-AUC delta |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0000 | 0.0000 | 0.0000 | +0.0028 | +0.0026 |
| Random Forest | +0.0042 | +0.0038 | +0.0040 | -0.0115 | +0.0016 |
| XGBoost | -0.0041 | +0.0072 | +0.0022 | -0.0115 | -0.0088 |
| LightGBM | -0.0332 | +0.0112 | -0.0097 | -0.0013 | -0.0073 |

### Train–Validation generalization gaps

Gaps are Train minus Validation. Smaller positive gaps suggest less apparent overfit,
while negative values mean Validation exceeded Train and are not evidence of leakage.

| Model | Full PR-AUC gap | Reduced PR-AUC gap | Full ROC-AUC gap | Reduced ROC-AUC gap |
|---|---:|---:|---:|---:|
| Logistic Regression | -0.0331 | -0.0342 | 0.0163 | 0.0151 |
| Random Forest | 0.1782 | 0.1869 | 0.2352 | 0.2301 |
| XGBoost | 0.1118 | 0.0839 | 0.1405 | 0.1026 |
| LightGBM | 0.1317 | 0.1888 | 0.1492 | 0.2296 |

The reduced Logistic Regression is essentially unchanged at threshold 0.50 and has
slightly higher ranking metrics. Reduced Random Forest has slightly better threshold
metrics but lower Validation PR-AUC and a slightly larger PR-AUC generalization gap.
Reduced XGBoost narrows both ranking-metric generalization gaps, while its Validation
PR-AUC and ROC-AUC are lower. Reduced LightGBM shows a substantially wider Train–
Validation gap and lower recall. These mixed results do not establish a winning feature
set; both the full and reduced specifications remain experimental pending manual review.

## Phase 2G full-feature Validation threshold analysis

Threshold analysis follows tuning because hyperparameters and feature space must be
fixed before examining how probability cutoffs change operational classifications.
The conventional 0.50 cutoff is a reference, not evidence that it matches pharmacy
follow-up capacity or the relative consequences of missed renewals and unnecessary
reviews.

The four tuned full 16-feature pipelines were refitted on Train only with unchanged
Phase 2D hyperparameters and leakage-safe preprocessing. Predicted probabilities were
generated for Validation only. Test and the reduced-feature models were not accessed.
Thresholds from 0.20 through 0.80 in increments of 0.05 were evaluated. Complete
reproducible tables are produced by `scripts/run_threshold_analysis.py`; a compact view
of three regions is shown here.

| Model | Threshold | Precision | Recall | F1 | Flagged | FP | FN | Missed per 100 actual | Unnecessary per 100 observations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.35 | 0.642 | 0.967 | 0.772 | 363 | 130 | 8 | 3.32 | 33.85 |
| Logistic Regression | **0.50 reference** | 0.705 | 0.900 | 0.791 | 308 | 91 | 24 | 9.96 | 23.70 |
| Logistic Regression | 0.65 | 0.745 | 0.606 | 0.668 | 196 | 50 | 95 | 39.42 | 13.02 |
| Random Forest | 0.35 | 0.675 | 0.913 | 0.776 | 326 | 106 | 21 | 8.71 | 27.60 |
| Random Forest | **0.50 reference** | 0.720 | 0.801 | 0.758 | 268 | 75 | 48 | 19.92 | 19.53 |
| Random Forest | 0.65 | 0.744 | 0.556 | 0.637 | 180 | 46 | 107 | 44.40 | 11.98 |
| XGBoost | 0.35 | 0.693 | 0.900 | 0.783 | 313 | 96 | 24 | 9.96 | 25.00 |
| XGBoost | **0.50 reference** | 0.729 | 0.826 | 0.774 | 273 | 74 | 42 | 17.43 | 19.27 |
| XGBoost | 0.65 | 0.748 | 0.639 | 0.689 | 206 | 52 | 87 | 36.10 | 13.54 |
| LightGBM | 0.35 | 0.696 | 0.884 | 0.779 | 306 | 93 | 28 | 11.62 | 24.22 |
| LightGBM | **0.50 reference** | 0.719 | 0.784 | 0.750 | 263 | 74 | 52 | 21.58 | 19.27 |
| LightGBM | 0.65 | 0.766 | 0.598 | 0.671 | 188 | 44 | 97 | 40.25 | 11.46 |

Threshold-independent Validation ranking metrics remain: Logistic Regression PR-AUC
0.7692 and ROC-AUC 0.6837; Random Forest 0.7699 and 0.6849; XGBoost 0.7730 and 0.7022;
LightGBM 0.7515 and 0.6878.

Operationally, a false negative represents an actual renewal that was not flagged;
`renewals_missed_per_100_actual` is therefore `FN / 241 * 100`. A false positive
represents an unnecessary pharmacy review; `unnecessary_followups_per_100_observations`
is `FP / 384 * 100`. No monetary cost is assigned.

The clearest recall declines occur above approximately 0.55 for Logistic Regression,
Random Forest, and LightGBM, and above approximately 0.70 for XGBoost. Precision tends
to improve as the review pool contracts, but improvements are not uniform; for example,
Logistic Regression gains precision around 0.50 and 0.75, XGBoost around 0.70, and the
tree models gain again near 0.80 while missing most renewals. These are descriptive
trade-offs only. No threshold or final model has been selected.

## Phase 2H Logistic Regression versus XGBoost stability

This pre-Test head-to-head comparison used the unchanged tuned full-feature candidates.
Five expanding `TimeSeriesSplit` folds were evaluated entirely inside Train, with
preprocessing refitted per fold. Validation was analyzed separately. Test remained
untouched.

### Train-only fold results at threshold 0.50

| Model | Fold | Train n | Val n | Positive rate | PR-AUC | ROC-AUC | Precision | Recall | F1 | Accuracy | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic | 1 | 289 | 285 | .5965 | .7530 | .6980 | .7365 | .6412 | .6855 | .6491 | 39 | 61 |
| Logistic | 2 | 574 | 285 | .5754 | .6495 | .6270 | .6266 | .8902 | .7355 | .6316 | 87 | 18 |
| Logistic | 3 | 859 | 285 | .5930 | .6837 | .6665 | .6881 | .8225 | .7493 | .6737 | 63 | 30 |
| Logistic | 4 | 1,144 | 285 | .6702 | .8375 | .7379 | .7544 | .9005 | .8210 | .7368 | 56 | 19 |
| Logistic | 5 | 1,429 | 285 | .5754 | .7222 | .7134 | .6712 | .8963 | .7676 | .6877 | 72 | 17 |
| XGBoost | 1 | 289 | 285 | .5965 | .7394 | .7057 | .7517 | .6588 | .7022 | .6667 | 37 | 58 |
| XGBoost | 2 | 574 | 285 | .5754 | .6614 | .5953 | .5895 | .8232 | .6870 | .5684 | 94 | 29 |
| XGBoost | 3 | 859 | 285 | .5930 | .7277 | .6974 | .6961 | .8402 | .7614 | .6877 | 62 | 27 |
| XGBoost | 4 | 1,144 | 285 | .6702 | .8572 | .7415 | .7619 | .8377 | .7980 | .7158 | 50 | 31 |
| XGBoost | 5 | 1,429 | 285 | .5754 | .7166 | .7034 | .6888 | .8232 | .7500 | .6842 | 61 | 29 |

### Stability summary

| Model | Metric | Mean | Std | Min | Max |
|---|---|---:|---:|---:|---:|
| Logistic | Recall | .8302 | .1104 | .6412 | .9005 |
| Logistic | F1 | .7518 | .0492 | .6855 | .8210 |
| Logistic | PR-AUC | .7292 | .0721 | .6495 | .8375 |
| Logistic | ROC-AUC | .6885 | .0430 | .6270 | .7379 |
| Logistic | Precision | .6954 | .0513 | .6266 | .7544 |
| Logistic | Accuracy | .6758 | .0404 | .6316 | .7368 |
| XGBoost | Recall | .7966 | .0774 | .6588 | .8402 |
| XGBoost | F1 | .7397 | .0452 | .6870 | .7980 |
| XGBoost | PR-AUC | .7404 | .0718 | .6614 | .8572 |
| XGBoost | ROC-AUC | .6887 | .0550 | .5953 | .7415 |
| XGBoost | Precision | .6976 | .0686 | .5895 | .7619 |
| XGBoost | Accuracy | .6646 | .0566 | .5684 | .7158 |

Logistic has higher mean recall and F1; XGBoost has lower recall and F1 variability.
XGBoost has slightly higher mean PR-AUC with essentially equal PR-AUC variability.
Logistic has lower ROC-AUC, precision, and accuracy variability. This is stability
evidence, not model selection.

### Validation operating points

| Model | Threshold | Precision | Recall | F1 | Accuracy | FP | FN | Flagged | Flag % | Missed/100 | Unnecessary/100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic | .35 | .6419 | .9668 | .7715 | .6406 | 130 | 8 | 363 | 94.5 | 3.32 | 33.85 |
| Logistic | .40 | .6552 | .9461 | .7742 | .6536 | 120 | 13 | 348 | 90.6 | 5.39 | 31.25 |
| Logistic | .45 | .6687 | .9129 | .7719 | .6615 | 109 | 21 | 329 | 85.7 | 8.71 | 28.39 |
| Logistic | .50 | .7045 | .9004 | .7905 | .7005 | 91 | 24 | 308 | 80.2 | 9.96 | 23.70 |
| Logistic | .55 | .7067 | .8299 | .7634 | .6771 | 83 | 41 | 283 | 73.7 | 17.01 | 21.61 |
| XGBoost | .35 | .6933 | .9004 | .7834 | .6875 | 96 | 24 | 313 | 81.5 | 9.96 | 25.00 |
| XGBoost | .40 | .6997 | .8797 | .7794 | .6875 | 91 | 29 | 303 | 78.9 | 12.03 | 23.70 |
| XGBoost | .45 | .7079 | .8548 | .7744 | .6875 | 85 | 35 | 291 | 75.8 | 14.52 | 22.14 |
| XGBoost | .50 | .7289 | .8257 | .7743 | .6979 | 74 | 42 | 273 | 71.1 | 17.43 | 19.27 |
| XGBoost | .55 | .7333 | .7759 | .7540 | .6823 | 68 | 54 | 255 | 66.4 | 22.41 | 17.71 |

### Approximately matched Validation recall

XGBoost points came only from the existing fixed grid; no interpolation or new search
occurred. Each cell lists recall / precision / F1 / FP / FN / flagged observations.

| Logistic threshold | Logistic | Closest XGBoost threshold | XGBoost |
|---:|---|---:|---|
| .35 | .9668 / .6419 / .7715 / 130 / 8 / 363 | .25 | .9544 / .6590 / .7797 / 119 / 11 / 349 |
| .40 | .9461 / .6552 / .7742 / 120 / 13 / 348 | .25 | .9544 / .6590 / .7797 / 119 / 11 / 349 |
| .45 | .9129 / .6687 / .7719 / 109 / 21 / 329 | .30 | .9212 / .6687 / .7749 / 110 / 19 / 332 |
| .50 | .9004 / .7045 / .7905 / 91 / 24 / 308 | .35 | .9004 / .6933 / .7834 / 96 / 24 / 313 |
| .55 | .8299 / .7067 / .7634 / 83 / 41 / 283 | .50 | .8257 / .7289 / .7743 / 74 / 42 / 273 |

At matched recall, XGBoost uses fewer reviews and false positives in the highest-recall
pair and near 0.83 recall. Results are nearly indistinguishable near 0.95 and 0.92
recall. At exactly 0.9004 recall, Logistic has higher precision and F1 with five fewer
reviews. Logistic's apparent 0.50 advantage therefore remains in that operating region,
but is not universal across thresholds.

Validation ranking metrics are Logistic PR-AUC 0.7692 and ROC-AUC 0.6837 versus
XGBoost 0.7730 and 0.7022. These measure ranking independently of the 0.50 cutoff.
XGBoost's added complexity provides a small PR-AUC advantage, a larger ROC-AUC
advantage, and workload benefits in some recall regions, but not a consistent practical
advantage everywhere. No final model or threshold has been selected; Test is untouched.

## Phase 2I pre-Test lock

Before first Test access, the final evaluation configuration was locked as the tuned
full 16-feature Logistic Regression (`C=0.01`, L2, SAGA, `max_iter=5000`) at threshold
0.50. Logistic Regression was chosen because it had higher mean time-fold recall and
F1, lower variability in several metrics, equal-recall Validation performance that
remained competitive with XGBoost, and materially lower complexity. Threshold 0.50 was
chosen because it preserved 0.9004 Validation recall with stronger precision and F1
than XGBoost at the exactly matched-recall point; this is a pre-Test operational choice,
not a claim of universal optimality.

The final fitting policy is **A: Train only**. This follows the already established
split contract: Train fits preprocessing/modeling, while Validation is reserved for
model and threshold selection and must not participate in preprocessing fit. Neither
this policy nor any locked choice may change in response to Test results.

### Final locked Test result

The locked Train-only pipeline was fitted once and evaluated on the previously untouched
Test partition at 0.50. Test contained 376 observations: 218 positive and 158 negative
(positive rate 0.5798). Accuracy was 0.6383, precision 0.6306, recall 0.9083, F1 0.7444,
PR-AUC 0.7352, and ROC-AUC 0.6710. The confusion matrix was `[[42,116],[20,198]]`.
It flagged 314 observations (83.51%), missed 9.17 renewals per 100 actual renewals, and
created 30.85 unnecessary follow-ups per 100 Test observations.

Using `patient_id` only as evaluation metadata, 245 seen-patient observations from 217
patients achieved precision 0.6364, recall 0.9433, F1 0.7600, PR-AUC 0.7572, and
ROC-AUC 0.6956 (`[[28,76],[8,133]]`). The 131 unseen-patient observations from 117
patients achieved precision 0.6190, recall 0.8442, F1 0.7143, PR-AUC 0.7012, and
ROC-AUC 0.6342 (`[[14,40],[12,65]]`). No subgroup-specific model was trained.

Relative to locked Validation, Test deltas were precision -0.0740, recall +0.0078,
F1 -0.0462, PR-AUC -0.0341, and ROC-AUC -0.0127. Recall remained consistent, while
lower precision and ranking metrics—especially for unseen patients—show meaningful
generalization limitations. The data are synthetic, the sample is modest, calibration
was not evaluated, and results do not establish real-world clinical performance.

**Test is now consumed.** It must not be reused for model selection, threshold tuning,
feature selection, or hyperparameter tuning. The result is reporting evidence only.

## Phase 2J post-Test error analysis

The locked model reproduced `TN=42`, `FP=116`, `FN=20`, and `TP=198`. Statistics below
use final model inputs after Train-fitted preprocessing and are mean / median / standard
deviation / 25th percentile / 75th percentile.

| Feature | FN (n=20) | TP (n=198) |
|---|---|---|
| On-time fill rate | .098 / .000 / .140 / .000 / .250 | .733 / .750 / .182 / .625 / .857 |
| Latest timing gap | 2.80 / 2.00 / 7.59 / -5.00 / 11.00 | .01 / .00 / 3.49 / -1.00 / 1.00 |
| Average timing gap | 2.20 / 2.50 / 5.68 / -3.25 / 6.00 | .56 / .27 / 1.80 / -.50 / 1.37 |
| Previous fill count | 3.05 / 3 / 1.23 / 2 / 4 | 6.04 / 5 / 2.84 / 4 / 8 |
| Refill interval std | 4.23 / 4.91 / 2.26 / 2.79 / 5.80 | 3.78 / 3.00 / 3.35 / 1.65 / 4.91 |
| Std available | .850 / 1 / .366 / 1 / 1 | .980 / 1 / .141 / 1 / 1 |
| Average refill interval | 62.20 / 61 / 24.24 / 37.80 / 86 | 41.52 / 31 / 20.97 / 29.59 / 58.95 |
| Medication prior fills | 1,144.85 / 1,130 / 85.55 / 1,098.75 / 1,178.50 | 1,197.85 / 1,170 / 116.80 / 1,119.25 / 1,222 |
| Prescription age | 207.20 / 181 / 96.14 / 160.75 / 247 | 161.03 / 148 / 95.52 / 98.50 / 177 |

| Feature | FP (n=116) | TN (n=42) |
|---|---|---|
| On-time fill rate | .624 / .615 / .206 / .500 / .750 | .157 / .200 / .139 / .000 / .250 |
| Latest timing gap | .47 / .00 / 4.61 / -1 / 1 | 7.45 / 8.50 / 5.97 / 2.50 / 12 |
| Average timing gap | .87 / .75 / 2.07 / -.78 / 2.21 | 5.69 / 6 / 3.10 / 3.10 / 8.09 |
| Previous fill count | 6.33 / 5 / 3.60 / 4 / 8 | 4.07 / 4 / 1.73 / 3 / 5 |
| Refill interval std | 5.01 / 3.51 / 4.92 / 2.50 / 5.94 | 6.21 / 5.68 / 4.21 / 4.66 / 7.23 |
| Std available | .948 / 1 / .222 / 1 / 1 | .952 / 1 / .216 / 1 / 1 |
| Average refill interval | 40.85 / 32.10 / 18.94 / 29.49 / 58.50 | 49.98 / 38.31 / 20.40 / 35.70 / 62.92 |
| Medication prior fills | 1,231.29 / 1,187.50 / 129.03 / 1,140.25 / 1,365.75 | 1,195.64 / 1,162 / 119.45 / 1,121.75 / 1,219 |
| Prescription age | 154.78 / 156.50 / 87.90 / 85.50 / 182.75 | 192.12 / 180 / 86.07 / 141.50 / 224 |

Seen patients produced `TP=133, FP=76, TN=28, FN=8`; error rate .3429, false-negative
rate .0567, and false-positive rate .7308. Unseen patients produced `TP=65, FP=40,
TN=14, FN=12`; error rate .3969, false-negative rate .1558, and false-positive rate
.7407. Patient identifiers were used only for this audit grouping.

Probability medians (10th/25th/75th/90th percentiles) were: TP .7086
(.5834/.6476/.7503/.7768), FN .4116 (.3561/.3856/.4358/.4620), FP .6623
(.5569/.6010/.7335/.7550), and TN .4250 (.3228/.3482/.4703/.4907). “Near threshold”
means within ±.05 of .50. Five FN (25.0%) and 11 FP (9.5%) were near threshold; the
remaining 15 FN and 105 FP were higher-confidence errors under this descriptive rule.

Missed renewals descriptively had much lower prior on-time rates, fewer prior fills,
longer average refill intervals, and older prescriptions than TP observations. False
positive follow-ups resembled TP observations more than TN observations on on-time
rate and timing gaps. These synthetic patterns imply no statistical significance or
model change and must not be presented as real pharmacy or patient behavior.

The Test set has already been consumed. Error-analysis findings are descriptive only
and were not used for model selection, feature selection, hyperparameter tuning, or
threshold adjustment.
