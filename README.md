# Pharmacy Billing vs. Ordering Reconciliation

This repository contains a local Python desktop application for validating and
reconciling medication billing and ordering data. Authoritative core logic lives
under `src/pharmacy_reconciliation` and remains independent of PySide6.

> Repository datasets and examples are synthetic and are not real pharmacy or patient records.

## Development setup

Python 3.11 through 3.13 is supported. From an isolated environment, install the
package and development dependencies with:

```text
python -m pip install -e ".[dev]"
```

Run the automated tests with:

```text
python -m pytest
```

Run the lightweight code-quality checks with:

```text
python -m ruff check .
python -m pyright
```

Runtime dependencies are intentionally limited to pandas, openpyxl, and PySide6.
openpyxl supports `.xlsx` input. PySide6 is retained for compatibility with the
existing desktop application; reconciliation and analytics modules do not import it.

## Raw export ingestion

The Phase 1B backend accepts `.csv` and `.xlsx` files through
`ingest_billing_file()` and `ingest_ordering_file()`:

```text
Raw pharmacy export
→ string-preserving file loading
→ header normalization for comparison
→ deterministic column mapping
→ canonical schema transformation
→ Phase 1A value normalization and validation
→ readiness decision
→ reconciliation
```

Other spreadsheet formats, including `.xls`, are rejected with a clear unsupported
file-type error. Excel sheet selection can be supplied by name or zero-based index.

Column mapping uses deterministic aliases and does not use machine learning or fuzzy matching.

Compatibility with any specific pharmacy software vendor has not been claimed or
verified unless an actual documented export format has been tested. The current
aliases are generic and tested only against fictional exports.

### Header normalization and mapping states

Header comparison applies Unicode compatibility normalization, trims whitespace,
lowercases text, converts runs of spaces/hyphens/punctuation to one underscore, and
removes surrounding underscores. Original header text is retained in audit metadata;
source values are not changed during header normalization.

Each source column receives one state:

- `EXACT`: normalized header equals a canonical field.
- `ALIAS`: normalized header equals an explicit registered alias.
- `MANUAL`: a human-confirmed override selected the canonical destination.
- `AMBIGUOUS`: multiple defensible destinations or source columns conflict.
- `UNMAPPED`: no supported mapping exists, or an automatic alias was superseded by a manual choice.

Unknown columns such as store, comment, price, or batch fields remain `UNMAPPED` and
do not by themselves block ingestion. Their original and normalized names remain in
the mapping result for a future UI.

The generic billing aliases cover transaction/claim identifiers, patient identifiers
and names, DOB, prescription/Rx numbers, NDC, medication descriptions, fill/service
dates, insurance/plan names, BIN, dispensed/billed quantity, days supply, and refills
remaining. Ordering aliases cover order/PO identifiers, NDC, medication descriptions,
order dates, and ordered/purchase quantities. Vague names such as `id`, `date`,
`number`, `amount`, and `value` are intentionally not registered.

### Conflicts, manual mapping, and readiness

The mapper reports duplicate normalized headers, one source matching several fields,
and several sources targeting one canonical field. It never selects among conflicting
columns automatically. A future UI can submit original source headers as overrides:

```python
result = ingest_billing_file(
    "synthetic_export.csv",
    manual_mapping={
        "Disp Qty": "quantity_billed",
        "Rx Date": "billing_date",
    },
)
```

Manual destinations must be valid canonical fields, source headers must exist and be
unique, and two manual sources cannot claim the same destination. A manual selection
takes precedence over competing automatic aliases, which are retained as unmapped
audit entries.

Missing required mappings or unresolved conflicts prevent canonical transformation.
Missing optional fields do not. Once structurally mapped, the canonical frame is
passed to the existing Phase 1A validator; mapping code does not duplicate value
validation.

An `IngestionResult` contains the input/file type, row count, original columns,
per-column mapping records, conflicts, missing required fields, normalization notes,
canonical data when available, full validation results, valid/invalid counts, warning
counts, and `ready_for_reconciliation`. Readiness is true only when required mappings
exist, conflicts are resolved, and validation has no errors. Ambiguous numeric dates
such as `07/08/2026` are errors; an unambiguous form such as `2026-07-08` is required.
Warnings such as possible duplicates remain visible but do not alone block readiness;
records are never silently discarded.

## Canonical data contract

CSV loaders read all source fields as strings first so identifiers such as NDC are
not converted to numbers. Validation then produces the canonical types below.
Real pharmacy exports are not assumed to use these names; source-specific column
mapping remains a future ingestion responsibility.

### Billing

| Field | Kind | Required | Meaning |
|---|---|---:|---|
| `billing_id` | Identifier | Yes | Source billing transaction |
| `patient_id` | Identifier | Yes | Internal patient reference |
| `patient_name` | Identity/display | No | Display-only; not an ML feature by default |
| `date_of_birth` | Date/identity | No | Display-only; not an ML feature by default |
| `prescription_id` | Identifier | Yes | Prescription/fill reference |
| `ndc` | String identifier | Yes | Trimmed but otherwise unchanged |
| `drug_name` | Categorical string | Yes | Exact-match medication name |
| `billing_date` | Date | Yes | Reconciliation-period date |
| `insurance_name` | Categorical string | No | Billing insurer |
| `bin_number` | Categorical string | No | Billing BIN |
| `quantity_billed` | Numeric | Yes | Non-negative billed quantity |
| `days_supply` | Integer numeric | No | Non-negative supply duration |
| `refills_remaining` | Integer numeric | No | Non-negative refill count |

### Ordering

| Field | Kind | Required | Meaning |
|---|---|---:|---|
| `order_id` | Identifier | Yes | Source order transaction |
| `ndc` | String identifier | Yes | Trimmed but otherwise unchanged |
| `drug_name` | Categorical string | Yes | Exact-match medication name |
| `ordered_date` | Date | Yes | Reconciliation-period date |
| `quantity_ordered` | Numeric | Yes | Non-negative ordered quantity |

## Reconciliation rules

The matching key is the exact normalized pair `ndc + drug_name`. No fuzzy matching
or universal NDC padding is performed. Billing and ordering records are filtered to
the same **inclusive** start and end dates before both are aggregated at medication
level. An outer merge retains medications present on only one side.

For each medication:

```text
net_difference = total_ordered - total_billed
```

- Zero is `MATCHED`.
- A positive difference is `EXTRA` and becomes `extra_quantity`.
- A negative difference is `SHORT`; its absolute value becomes `short_quantity`.
- `short_quantity` and `extra_quantity` cannot both be positive.

Validation errors stop reconciliation. Exact duplicate candidates are reported as
warnings and retained because the system cannot determine whether repeated records
are accidental or legitimate transactions.

## Billing analytics

Insurance analytics aggregate billed quantities separately by medication, insurer,
and BIN, with unique patient counts. They never attach ordered quantities. Missing
insurance or BIN values are grouped as `UNKNOWN`, ensuring the insurance breakdown
continues to sum to medication-level billed totals.

Patient analytics provide both medication-level unique-patient/billed-quantity
summaries and row-level fill detail. Row-level detail retains identity/display,
prescription, insurer, date, quantity, days-supply, and refill fields for future
authorized internal workflows.

## Synthetic development data

Phase 2 modeling concluded with a pre-Test lock on the tuned full-feature Logistic
Regression (`C=0.01`, L2/SAGA) at threshold 0.50 and a Train-only final fitting policy.
The single final Test evaluation is documented in `docs/ml_methodology.md`. Test is now
consumed and must not be reused for model, threshold, feature, or hyperparameter choices.

### Local MLflow tracking

Install the ML dependencies, record the established history, and launch the local UI:

```powershell
python -m pip install -e ".[ml]"
python scripts/track_mlflow_history.py
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000` after starting the UI. Run metadata is stored in the
Git-ignored local `mlflow.db`; model artifacts are stored under the ignored
`mlflow_artifacts/` directory. The legacy ignored `mlruns/` directory is not migrated or
modified. Running tracking does not rerun tuning or change locked decisions.

`data/synthetic` contains a small, manually calculable fictional dataset. It covers
matched, short, extra, billing-only, order-only, multi-insurer, multi-patient,
out-of-period, whitespace-normalization, duplicate, invalid-quantity, malformed-date,
negative-value, and missing-identifier cases.

## Privacy and repository policy

Never commit real patient names, dates of birth, prescription records, billing
records, pharmacy exports, credentials, PHI, PII, generated reports, or confidential
company information. Patient identity fields are included solely for authorized
internal display/review needs and are not ML features by default.

The prior exploratory notebook with outputs of unconfirmed provenance was removed
from the Phase 1 milestone. Only reviewed fictional fixtures under `data/synthetic`
belong in source control.

## Desktop application

`app.py` now resolves the root `main_window.py` module, and the missing icon reference
was removed from the PyInstaller specification. The desktop UI now calls a Qt-free
`ReconciliationController`, which orchestrates the authoritative ingestion,
reconciliation, insurance, and patient modules under `src/pharmacy_reconciliation`.
The obsolete pre-Phase-1 processing module and generated bytecode were removed.

The desktop workflow supports billing and ordering CSV/XLSX selection, concise file
readiness summaries, manual resolution of ambiguous/unmapped columns, inclusive date
selection, medication-level inventory results, medication filtering, insurance/BIN
breakdowns, unique-patient summaries, patient billing detail, and inventory PDF
export. Empty exports and periods with no activity are handled without exposing Python
tracebacks. The UI does not implement or duplicate mapping, validation, reconciliation,
or analytics rules.

## Synthetic longitudinal research data

Phase 2A adds a fixed-seed generator for fictional patients, medications,
prescriptions, and repeated fills over an 18-month study window. It also produces a
point-in-time table for zero-refill patients 10 days before expected supply exhaustion.
The target captures receipt of a new Rx number for the same patient and exact NDC
after observation and through 7 days after expected supply end.

The builder retains only facts known by the outreach date, censors rows without a
complete target window, and preserves the observation date for a future chronological
split. Identifiers and the zero-refill eligibility field remain audit-only rather than
model features. No model is trained or evaluated.
Full assumptions, target semantics, feature definitions, and leakage exclusions are
documented in `docs/phase2a_longitudinal_data.md`. Regenerate the reviewed artifacts
with:

```text
python scripts/generate_longitudinal_data.py
```

## ML preparation methodology

Phase 2 preparation uses a fixed chronological split: Train before 2026-02-01,
Validation from 2026-02-01 through 2026-03-31, and Test from 2026-04-01 onward. It
never randomly splits observations. Identifiers, dates, NDC/Rx fields, and the
zero-refill eligibility field are excluded from modeling.

Insufficient refill-interval history receives an explicit availability indicator and
a median learned from Train only; Validation and Test never influence preprocessing
fit. Repeated patients may occur across periods for production-like evaluation, while
`patient_id` remains excluded. Final testing will later report overall, previously-seen
patient, and unseen-patient performance. Full rationale and leakage controls are in
`docs/ml_methodology.md`.

Phase 2C now includes an untuned Train/Validation comparison of logistic regression,
random forest, XGBoost, and LightGBM using the fixed 0.50 reference threshold. Test
remains untouched, and no final model, tuned configuration, or operational threshold
has been selected. Exact configurations and metrics are recorded in the methodology.

Phase 2D adds deterministic Train-only tuning with five expanding time-series folds
and PR-AUC scoring. Tuned estimators are compared on Validation at the unchanged 0.50
reference threshold; Test remains untouched. No final model or threshold is selected.
