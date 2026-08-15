"""Leakage-aware point-in-time features for synthetic refill research."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import fmean, pstdev
from typing import Any, cast

import pandas as pd

IDENTIFIER_COLUMNS = (
    "observation_id", "observation_date", "expected_supply_end_date",
    "patient_id", "medication_id", "ndc", "prescription_id",
)
AUDIT_COLUMNS = ("current_refills_remaining",)
FEATURE_COLUMNS = (
    "current_quantity_billed", "current_days_supply", "current_refill_number",
    "previous_fill_count", "days_since_previous_fill",
    "average_previous_refill_interval_days", "std_previous_refill_interval_days",
    "latest_refill_timing_gap_days", "average_previous_timing_gap_days",
    "previous_early_fill_rate", "previous_on_time_fill_rate",
    "prescription_age_days", "medication_prior_fill_count",
    "medication_prior_average_days_supply", "medication_prior_average_quantity",
)
TARGET_COLUMN = "prescription_renewal_within_window"
LEGACY_TARGET_COLUMN = "refill_within_14_days"


@dataclass(frozen=True)
class SupplyEndTargetConfig:
    """Fixed business timing for supply-exhaustion observations."""

    outreach_lead_days: int = 10
    window_before_days: int = 0
    window_after_days: int = 7

    def __post_init__(self) -> None:
        if min(self.outreach_lead_days, self.window_before_days, self.window_after_days) < 0:
            raise ValueError("Supply-end timing values cannot be negative.")


@dataclass(frozen=True)
class SupplyEndObservationResult:
    """Primary observations plus transparent eligibility accounting."""

    observations: pd.DataFrame
    candidate_fill_count: int
    zero_refill_candidate_count: int
    censored_count: int
    lead_time_ineligible_count: int
    renewal_already_received_count: int
    config: SupplyEndTargetConfig


def _timestamp(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _prepared_inputs(
    fills: pd.DataFrame, prescriptions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fill_data = fills.copy()
    fill_data["fill_date"] = pd.to_datetime(fill_data["fill_date"]).dt.normalize()
    prescription_data = prescriptions.copy()
    prescription_data["prescription_date"] = pd.to_datetime(
        prescription_data["prescription_date"]
    ).dt.normalize()
    if "ndc" not in prescription_data:
        prescription_ndcs = fill_data.groupby("prescription_id")["ndc"].first()
        prescription_data["ndc"] = prescription_data["prescription_id"].map(
            prescription_ndcs
        )
    ordered = fill_data.sort_values(
        ["patient_id", "medication_id", "fill_date", "fill_id"], kind="mergesort"
    ).reset_index(drop=True)
    ordered["next_fill_date"] = ordered.groupby(
        ["patient_id", "medication_id"], sort=False
    )["fill_date"].shift(-1)
    return ordered, prescription_data


def build_supply_end_observation_result(
    fills: pd.DataFrame,
    prescriptions: pd.DataFrame,
    study_end_date: str | date | datetime | pd.Timestamp,
    config: SupplyEndTargetConfig | None = None,
) -> SupplyEndObservationResult:
    """Build zero-refill observations for exact-NDC prescription renewal."""
    target_config = config or SupplyEndTargetConfig()
    study_end = _timestamp(study_end_date)
    ordered, prescription_data = _prepared_inputs(fills, prescriptions)
    prescription_dates = prescription_data.set_index("prescription_id")["prescription_date"]
    ordered["expected_supply_end_date"] = ordered["fill_date"] + pd.to_timedelta(
        ordered["days_supply"], unit="D"
    )
    ordered["observation_date"] = ordered["expected_supply_end_date"] - timedelta(
        days=target_config.outreach_lead_days
    )
    ordered["target_window_end"] = ordered["expected_supply_end_date"] + timedelta(
        days=target_config.window_after_days
    )

    zero_refill = ordered["refills_remaining"] == 0
    lead_time_ineligible = ordered["observation_date"] < ordered["fill_date"]
    censored = ordered["target_window_end"] > study_end
    renewal_already_received = pd.Series(False, index=ordered.index)
    qualifying_renewal_date: dict[int, pd.Timestamp | None] = {}
    for fill in ordered.itertuples(index=True):
        fill = cast(Any, fill)
        current_rx_date = pd.Timestamp(prescription_dates.loc[fill.prescription_id])
        new_rx = prescription_data.loc[
            (prescription_data["patient_id"] == fill.patient_id)
            & (prescription_data["ndc"].astype(str) == str(fill.ndc))
            & (prescription_data["prescription_id"] != fill.prescription_id)
            & (prescription_data["prescription_date"] > current_rx_date)
        ].sort_values(["prescription_date", "prescription_id"], kind="mergesort")
        already = new_rx.loc[
            new_rx["prescription_date"] <= pd.Timestamp(fill.observation_date)
        ]
        renewal_already_received.loc[fill.Index] = not already.empty
        future = new_rx.loc[
            new_rx["prescription_date"] > pd.Timestamp(fill.observation_date)
        ]
        qualifying_renewal_date[fill.Index] = (
            pd.Timestamp(future.iloc[0]["prescription_date"]) if not future.empty else None
        )
    eligible = (
        zero_refill & ~lead_time_ineligible & ~censored & ~renewal_already_received
    )

    medication_history: dict[str, list[tuple[pd.Timestamp, str, int, int]]] = {}
    for fill in ordered.itertuples(index=False):
        fill = cast(Any, fill)
        medication_history.setdefault(fill.medication_id, []).append((
            pd.Timestamp(fill.fill_date), str(fill.fill_id),
            int(fill.days_supply), int(fill.quantity_billed),
        ))

    rows: list[dict[str, object]] = []
    for (_, _), group in ordered.groupby(["patient_id", "medication_id"], sort=False):
        prior_dates: list[pd.Timestamp] = []
        prior_days_supply: list[int] = []
        intervals: list[int] = []
        timing_gaps: list[int] = []
        for fill in group.itertuples(index=True):
            fill = cast(Any, fill)
            fill_date = pd.Timestamp(fill.fill_date)
            observation_date = pd.Timestamp(fill.observation_date)
            days_since_previous = None
            latest_timing_gap = None
            if prior_dates:
                days_since_previous = (fill_date - prior_dates[-1]).days
                intervals.append(days_since_previous)
                latest_timing_gap = days_since_previous - prior_days_supply[-1]
                timing_gaps.append(latest_timing_gap)

            if bool(eligible.loc[fill.Index]):
                strictly_prior_medication = [
                    item for item in medication_history[fill.medication_id]
                    if item[0] < observation_date and item[1] != fill.fill_id
                ]
                med_days_supply = [item[2] for item in strictly_prior_medication]
                med_quantities = [item[3] for item in strictly_prior_medication]
                prescription_date = pd.Timestamp(prescription_dates.loc[fill.prescription_id])
                renewal_date = qualifying_renewal_date[fill.Index]
                target = int(
                    renewal_date is not None
                    and renewal_date <= pd.Timestamp(fill.target_window_end)
                )
                rows.append({
                    "observation_id": fill.fill_id,
                    "observation_date": observation_date,
                    "expected_supply_end_date": pd.Timestamp(fill.expected_supply_end_date),
                    "patient_id": fill.patient_id,
                    "medication_id": fill.medication_id,
                    "ndc": fill.ndc,
                    "prescription_id": fill.prescription_id,
                    "current_refills_remaining": int(fill.refills_remaining),
                    "current_quantity_billed": int(fill.quantity_billed),
                    "current_days_supply": int(fill.days_supply),
                    "current_refill_number": int(fill.refill_number),
                    "previous_fill_count": len(prior_dates),
                    "days_since_previous_fill": days_since_previous,
                    "average_previous_refill_interval_days": fmean(intervals) if intervals else None,
                    "std_previous_refill_interval_days": pstdev(intervals) if len(intervals) >= 2 else None,
                    "latest_refill_timing_gap_days": latest_timing_gap,
                    "average_previous_timing_gap_days": fmean(timing_gaps) if timing_gaps else None,
                    "previous_early_fill_rate": (
                        sum(gap < -1 for gap in timing_gaps) / len(timing_gaps)
                        if timing_gaps else None
                    ),
                    "previous_on_time_fill_rate": (
                        sum(abs(gap) <= 1 for gap in timing_gaps) / len(timing_gaps)
                        if timing_gaps else None
                    ),
                    "prescription_age_days": (observation_date - prescription_date).days,
                    "medication_prior_fill_count": len(strictly_prior_medication),
                    "medication_prior_average_days_supply": fmean(med_days_supply) if med_days_supply else None,
                    "medication_prior_average_quantity": fmean(med_quantities) if med_quantities else None,
                    TARGET_COLUMN: target,
                })

            prior_dates.append(fill_date)
            prior_days_supply.append(int(fill.days_supply))

    columns = [*IDENTIFIER_COLUMNS, *AUDIT_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN]
    observations = pd.DataFrame(rows, columns=columns).sort_values(
        ["observation_date", "observation_id"], kind="mergesort"
    ).reset_index(drop=True)
    return SupplyEndObservationResult(
        observations=observations,
        candidate_fill_count=len(ordered),
        zero_refill_candidate_count=int(zero_refill.sum()),
        censored_count=int((zero_refill & censored).sum()),
        lead_time_ineligible_count=int(
            (zero_refill & ~censored & lead_time_ineligible).sum()
        ),
        renewal_already_received_count=int(
            (
                zero_refill
                & ~censored
                & ~lead_time_ineligible
                & renewal_already_received
            ).sum()
        ),
        config=target_config,
    )


def build_supply_end_observations(
    fills: pd.DataFrame,
    prescriptions: pd.DataFrame,
    study_end_date: str | date | datetime | pd.Timestamp,
    config: SupplyEndTargetConfig | None = None,
) -> pd.DataFrame:
    """Return the primary supply-end observation table."""
    return build_supply_end_observation_result(
        fills, prescriptions, study_end_date, config
    ).observations


def build_fill_date_refill_observations(
    fills: pd.DataFrame,
    prescriptions: pd.DataFrame,
    study_end_date: str | date | datetime | pd.Timestamp,
) -> pd.DataFrame:
    """Build the superseded Phase 2A target for explicit comparison only."""
    study_end = _timestamp(study_end_date)
    ordered, _ = _prepared_inputs(fills, prescriptions)
    rows: list[dict[str, object]] = []
    for fill in ordered.itertuples(index=False):
        fill = cast(Any, fill)
        observation_date = pd.Timestamp(fill.fill_date)
        if observation_date > study_end - timedelta(days=14):
            continue
        days_to_next = (
            (pd.Timestamp(fill.next_fill_date) - observation_date).days
            if pd.notna(fill.next_fill_date) else None
        )
        rows.append({
            "observation_id": fill.fill_id,
            "observation_date": observation_date,
            LEGACY_TARGET_COLUMN: int(days_to_next is not None and 1 <= days_to_next <= 14),
        })
    return pd.DataFrame(
        rows, columns=["observation_id", "observation_date", LEGACY_TARGET_COLUMN]
    ).sort_values(["observation_date", "observation_id"], kind="mergesort").reset_index(drop=True)


# The compatibility name resolves to the new authoritative primary target.
build_refill_observations = build_supply_end_observations
