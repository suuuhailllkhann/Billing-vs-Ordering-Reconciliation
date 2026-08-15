"""Basic integrity summaries for generated longitudinal research data."""

from typing import Any, cast

import pandas as pd

from pharmacy_reconciliation.research.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SupplyEndObservationResult,
)
from pharmacy_reconciliation.synthetic.longitudinal import SyntheticLongitudinalDataset


def summarize_longitudinal_dataset(
    dataset: SyntheticLongitudinalDataset,
    observations: pd.DataFrame,
    observation_result: SupplyEndObservationResult | None = None,
) -> dict[str, Any]:
    fills_per_patient = dataset.fills.groupby("patient_id").size()
    prescriptions = dataset.prescriptions.set_index("prescription_id")
    fills_with_rx = dataset.fills.join(
        prescriptions[["prescription_date", "refills_authorized"]],
        on="prescription_id",
        validate="many_to_one",
    )
    invalid_chronology = int(
        (fills_with_rx["fill_date"] < fills_with_rx["prescription_date"]).sum()
    )
    invalid_refill_accounting = int((
        fills_with_rx["refills_remaining"]
        != fills_with_rx["refills_authorized"] - fills_with_rx["refill_number"]
    ).sum())
    chronological_fills = dataset.fills.sort_values(
        ["patient_id", "medication_id", "fill_date", "fill_id"], kind="mergesort"
    ).copy()
    groups = chronological_fills.groupby(["patient_id", "medication_id"], sort=False)
    chronological_fills["previous_fill_date"] = groups["fill_date"].shift(1)
    chronological_fills["previous_days_supply"] = groups["days_supply"].shift(1)
    chronological_fills["interval_days"] = (
        chronological_fills["fill_date"] - chronological_fills["previous_fill_date"]
    ).dt.days
    chronological_fills["timing_gap_days"] = (
        chronological_fills["interval_days"] - chronological_fills["previous_days_supply"]
    )
    gaps = chronological_fills["timing_gap_days"].dropna()
    timing_counts = {
        "early": int((gaps < -1).sum()),
        "approximately_on_time": int(gaps.between(-1, 1, inclusive="both").sum()),
        "late": int((gaps > 1).sum()),
    }
    prescriptions_per_history = dataset.prescriptions.groupby(
        ["patient_id", "medication_id"]
    ).size()
    positives = int(observations[TARGET_COLUMN].sum())
    observation_count = len(observations)
    summary = {
        "seed": dataset.config.seed,
        "study_start_date": dataset.config.start_date.isoformat(),
        "study_end_date": dataset.config.end_date.isoformat(),
        "patient_count": len(dataset.patients),
        "patient_profile_counts": {
            str(profile): int(count)
            for profile, count in dataset.patients["behavior_profile"]
            .value_counts()
            .sort_index()
            .items()
        },
        "medication_count": len(dataset.medications),
        "prescription_count": len(dataset.prescriptions),
        "fill_count": len(dataset.fills),
        "prediction_observation_count": observation_count,
        "positive_target_count": positives,
        "negative_target_count": observation_count - positives,
        "target_rate": positives / observation_count if observation_count else 0.0,
        "target_name": TARGET_COLUMN,
        "observation_date_min": observations["observation_date"].min().date().isoformat(),
        "observation_date_max": observations["observation_date"].max().date().isoformat(),
        "fill_date_min": dataset.fills["fill_date"].min().date().isoformat(),
        "fill_date_max": dataset.fills["fill_date"].max().date().isoformat(),
        "fills_per_patient": {
            "minimum": int(fills_per_patient.min()),
            "median": float(fills_per_patient.median()),
            "mean": float(fills_per_patient.mean()),
            "maximum": int(fills_per_patient.max()),
        },
        "days_supply_counts": {
            str(int(cast(Any, value))): int(count)
            for value, count in dataset.fills["days_supply"].value_counts().sort_index().items()
        },
        "observed_refill_timing_counts": timing_counts,
        "patient_medication_history_count": int(groups.ngroups),
        "single_fill_history_count": int((groups.size() == 1).sum()),
        "renewed_patient_medication_history_count": int((prescriptions_per_history > 1).sum()),
        "additional_renewal_prescription_count": int(
            (prescriptions_per_history - 1).clip(lower=0).sum()
        ),
        "missing_feature_counts": {
            column: int(observations[column].isna().sum()) for column in FEATURE_COLUMNS
        },
        "duplicate_patient_ids": int(dataset.patients["patient_id"].duplicated().sum()),
        "duplicate_medication_ids": int(dataset.medications["medication_id"].duplicated().sum()),
        "duplicate_prescription_ids": int(dataset.prescriptions["prescription_id"].duplicated().sum()),
        "duplicate_fill_ids": int(dataset.fills["fill_id"].duplicated().sum()),
        "invalid_fill_before_prescription_count": invalid_chronology,
        "invalid_refill_accounting_count": invalid_refill_accounting,
    }
    if observation_result is not None:
        summary.update({
            "candidate_fill_count": observation_result.candidate_fill_count,
            "censored_observation_count": observation_result.censored_count,
            "lead_time_ineligible_count": observation_result.lead_time_ineligible_count,
            "zero_refill_candidate_count": observation_result.zero_refill_candidate_count,
            "renewal_already_received_count": (
                observation_result.renewal_already_received_count
            ),
            "outreach_lead_days": observation_result.config.outreach_lead_days,
            "target_window_after_days": observation_result.config.window_after_days,
        })
    return summary
