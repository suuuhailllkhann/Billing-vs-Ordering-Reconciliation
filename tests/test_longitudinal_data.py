import random
from dataclasses import replace
from datetime import timedelta

import pandas as pd

from pharmacy_reconciliation.research.features import (
    AUDIT_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_supply_end_observation_result,
    build_supply_end_observations,
)
from pharmacy_reconciliation.research.integrity import summarize_longitudinal_dataset
from pharmacy_reconciliation.synthetic.longitudinal import (
    LongitudinalConfig,
    _profile_assignments,
    _renewal_probability,
    _timing_interval,
    generate_longitudinal_dataset,
)


def _small_dataset():
    return generate_longitudinal_dataset(
        LongitudinalConfig(patient_count=20, medication_count=5)
    )


def _renewal_case(
    renewal_date: str | None = "2026-01-31",
    renewal_ndc: str = "90000000999",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prescriptions = [{
        "prescription_id": "SYN-RX-A",
        "patient_id": "SYN-PAT-A",
        "medication_id": "SYN-MED-A",
        "ndc": "90000000999",
        "prescription_date": "2026-01-01",
        "quantity_prescribed": 30,
        "days_supply": 30,
        "refills_authorized": 0,
    }]
    if renewal_date is not None:
        prescriptions.append({
            "prescription_id": "SYN-RX-B",
            "patient_id": "SYN-PAT-A",
            "medication_id": "SYN-MED-A",
            "ndc": renewal_ndc,
            "prescription_date": renewal_date,
            "quantity_prescribed": 30,
            "days_supply": 30,
            "refills_authorized": 2,
        })
    fills = pd.DataFrame([{
        "fill_id": "SYN-FILL-1",
        "patient_id": "SYN-PAT-A",
        "prescription_id": "SYN-RX-A",
        "medication_id": "SYN-MED-A",
        "ndc": "90000000999",
        "drug_name": "Synthetic History Tablets",
        "fill_date": "2026-01-01",
        "quantity_billed": 30,
        "days_supply": 30,
        "refill_number": 0,
        "refills_remaining": 0,
    }])
    return fills, pd.DataFrame(prescriptions)


def test_generation_is_reproducible_for_same_seed():
    config = LongitudinalConfig(patient_count=20, medication_count=5, seed=1234)
    first = generate_longitudinal_dataset(config)
    second = generate_longitudinal_dataset(config)
    for first_frame, second_frame in zip(
        (first.patients, first.medications, first.prescriptions, first.fills),
        (second.patients, second.medications, second.prescriptions, second.fills),
    ):
        pd.testing.assert_frame_equal(first_frame, second_frame)
    changed = generate_longitudinal_dataset(replace(config, seed=1235))
    assert not first.fills.equals(changed.fills)


def test_hidden_profile_assignment_is_seeded_and_40_40_20():
    first = _profile_assignments(100, random.Random(1234))
    second = _profile_assignments(100, random.Random(1234))
    assert first == second
    assert first.count("Consistent") == 40
    assert first.count("Average") == 40
    assert first.count("Inconsistent") == 20


def test_profile_changes_refill_timing_distribution():
    counts = {}
    for profile in ("Consistent", "Average", "Inconsistent"):
        rng = random.Random(4321)
        gaps = [_timing_interval(rng, 30, profile) - 30 for _ in range(4000)]
        counts[profile] = {
            "on_time": sum(abs(gap) <= 1 for gap in gaps),
            "late": sum(gap > 1 for gap in gaps),
        }
    assert counts["Consistent"]["on_time"] > counts["Average"]["on_time"]
    assert counts["Average"]["on_time"] > counts["Inconsistent"]["on_time"]
    assert counts["Inconsistent"]["late"] > counts["Average"]["late"]
    assert counts["Average"]["late"] > counts["Consistent"]["late"]


def test_profile_and_behavioral_history_adjust_renewal_probability():
    assert _renewal_probability("Consistent", [], 2) == 0.75
    assert _renewal_probability("Average", [], 2) == 0.60
    assert _renewal_probability("Inconsistent", [], 2) == 0.40
    assert round(_renewal_probability("Consistent", [0, 1, 0], 6), 2) == 0.95
    assert round(_renewal_probability("Average", [-2, -2, 0], 3), 2) == 0.57
    assert round(_renewal_probability("Inconsistent", [4, 3], 2), 2) == 0.25


def test_ids_relationships_chronology_and_refill_accounting_are_valid():
    dataset = _small_dataset()
    assert dataset.patients["patient_id"].is_unique
    assert dataset.medications["medication_id"].is_unique
    assert dataset.prescriptions["prescription_id"].is_unique
    assert dataset.fills["fill_id"].is_unique
    joined = dataset.fills.merge(
        dataset.prescriptions[
            ["prescription_id", "prescription_date", "refills_authorized"]
        ],
        on="prescription_id",
        validate="many_to_one",
    )
    assert (joined["fill_date"] >= joined["prescription_date"]).all()
    assert (
        joined["refills_remaining"]
        == joined["refills_authorized"] - joined["refill_number"]
    ).all()
    assert dataset.prescriptions["ndc"].map(type).eq(str).all()


def test_only_zero_refill_fills_are_eligible():
    fills, prescriptions = _renewal_case()
    nonzero = fills.iloc[0].copy()
    nonzero["fill_id"] = "SYN-FILL-2"
    nonzero["patient_id"] = "SYN-PAT-B"
    nonzero["refills_remaining"] = 1
    fills = pd.concat([fills, nonzero.to_frame().T], ignore_index=True)
    result = build_supply_end_observation_result(fills, prescriptions, "2026-03-01")
    assert result.zero_refill_candidate_count == 1
    assert result.observations["current_refills_remaining"].eq(0).all()


def test_prediction_date_is_ten_days_before_expected_supply_end():
    fills, prescriptions = _renewal_case()
    row = build_supply_end_observations(fills, prescriptions, "2026-03-01").iloc[0]
    assert row["expected_supply_end_date"] == pd.Timestamp("2026-01-31")
    assert row["observation_date"] == pd.Timestamp("2026-01-21")


def test_new_rx_at_plus_seven_boundary_is_positive():
    fills, prescriptions = _renewal_case("2026-02-07")
    row = build_supply_end_observations(fills, prescriptions, "2026-03-01").iloc[0]
    assert row[TARGET_COLUMN] == 1


def test_new_rx_after_plus_seven_boundary_is_negative():
    fills, prescriptions = _renewal_case("2026-02-08")
    row = build_supply_end_observations(fills, prescriptions, "2026-03-01").iloc[0]
    assert row[TARGET_COLUMN] == 0


def test_same_rx_refill_is_not_a_new_prescription():
    fills, prescriptions = _renewal_case(None)
    later = fills.iloc[0].copy()
    later["fill_id"] = "SYN-FILL-2"
    later["fill_date"] = "2026-01-31"
    later["refill_number"] = 1
    fills = pd.concat([fills, later.to_frame().T], ignore_index=True)
    row = build_supply_end_observations(fills, prescriptions, "2026-03-01").iloc[0]
    assert row[TARGET_COLUMN] == 0


def test_different_ndc_is_not_an_exact_product_renewal():
    fills, prescriptions = _renewal_case(renewal_ndc="90000000888")
    row = build_supply_end_observations(fills, prescriptions, "2026-03-01").iloc[0]
    assert row[TARGET_COLUMN] == 0


def test_renewal_on_or_before_prediction_date_suppresses_observation():
    for renewal_date in ("2026-01-20", "2026-01-21"):
        fills, prescriptions = _renewal_case(renewal_date)
        result = build_supply_end_observation_result(
            fills, prescriptions, "2026-03-01"
        )
        assert result.renewal_already_received_count == 1
        assert result.observations.empty


def test_incomplete_follow_up_window_is_censored_not_negative():
    fills, prescriptions = _renewal_case(None)
    result = build_supply_end_observation_result(fills, prescriptions, "2026-02-06")
    assert result.censored_count == 1
    assert result.observations.empty


def test_audit_identifiers_and_eligibility_field_are_not_model_features():
    fills, prescriptions = _renewal_case()
    observations = build_supply_end_observations(fills, prescriptions, "2026-03-01")
    assert "ndc" in observations
    assert observations["ndc"].map(type).eq(str).all()
    assert "current_refills_remaining" in AUDIT_COLUMNS
    assert "current_refills_remaining" not in FEATURE_COLUMNS
    assert {"observation_id", "patient_id", "prescription_id", "ndc"}.isdisjoint(
        FEATURE_COLUMNS
    )
    assert "behavior_profile" not in observations
    assert "behavior_profile" not in FEATURE_COLUMNS


def test_no_future_or_target_construction_fields_are_exported():
    fills, prescriptions = _renewal_case()
    observations = build_supply_end_observations(fills, prescriptions, "2026-03-01")
    forbidden = {
        "new_prescription_id", "renewal_date", "next_fill_date",
        "target_window_end", "patient_name", "date_of_birth",
    }
    assert forbidden.isdisjoint(observations.columns)


def test_generated_positive_targets_are_exact_new_rx_renewals_in_window():
    dataset = generate_longitudinal_dataset()
    result = build_supply_end_observation_result(
        dataset.fills, dataset.prescriptions, dataset.config.end_date
    )
    positives = result.observations.loc[
        result.observations[TARGET_COLUMN] == 1
    ]
    assert not positives.empty
    for row in positives.itertuples(index=False):
        matches = dataset.prescriptions.loc[
            (dataset.prescriptions["patient_id"] == row.patient_id)
            & (dataset.prescriptions["ndc"] == row.ndc)
            & (dataset.prescriptions["prescription_id"] != row.prescription_id)
            & (dataset.prescriptions["prescription_date"] > row.observation_date)
            & (
                dataset.prescriptions["prescription_date"]
                <= row.expected_supply_end_date + timedelta(days=7)
            )
        ]
        assert not matches.empty
    summary = summarize_longitudinal_dataset(
        dataset, result.observations, result
    )
    assert summary["invalid_fill_before_prescription_count"] == 0
    assert summary["invalid_refill_accounting_count"] == 0
