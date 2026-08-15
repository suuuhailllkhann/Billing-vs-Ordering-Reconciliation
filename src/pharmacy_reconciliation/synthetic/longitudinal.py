"""Auditable generator for fictional longitudinal pharmacy activity."""

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal, Sequence, cast

import pandas as pd


@dataclass(frozen=True)
class LongitudinalConfig:
    seed: int = 20260814
    patient_count: int = 1500
    medication_count: int = 12
    start_date: date = date(2025, 1, 1)
    end_date: date = date(2026, 6, 30)
    max_fills_per_patient_medication: int = 14


@dataclass(frozen=True)
class SyntheticLongitudinalDataset:
    patients: pd.DataFrame
    medications: pd.DataFrame
    prescriptions: pd.DataFrame
    fills: pd.DataFrame
    config: LongitudinalConfig


_FIRST_NAMES = (
    "Avery", "Blair", "Casey", "Devon", "Emery", "Finley", "Gray", "Harper",
    "Indigo", "Jordan", "Kai", "Lane", "Morgan", "Noel", "Oakley", "Parker",
    "Quinn", "Reese", "Sage", "Taylor",
)
_LAST_NAMES = (
    "Example", "Fiction", "Imaginary", "Sample", "Synthetic", "Testcase",
)
_DRUG_NAMES = (
    "Aureline Tablets", "Borealis Capsules", "Citrine Solution", "Dovetail Tablets",
    "Ember Suspension", "Fable Capsules", "Glimmer Tablets", "Harbor Solution",
    "Juniper Tablets", "Kestrel Capsules", "Lantern Suspension", "Meadow Tablets",
)
_INSURANCE_OPTIONS = (
    ("Northstar Synthetic Plan", "001234"),
    ("Blue Orchard Fictional", "005678"),
    ("Cedar Example Benefit", "009876"),
)
_DAYS_SUPPLY_OPTIONS = (7, 14, 30, 60, 90)
_DAYS_SUPPLY_WEIGHTS = (0.08, 0.14, 0.51, 0.17, 0.10)
PatientProfile = Literal["Consistent", "Average", "Inconsistent"]
_TIMING_WEIGHTS: dict[PatientProfile, tuple[float, float, float]] = {
    "Consistent": (0.15, 0.75, 0.10),
    "Average": (0.20, 0.55, 0.25),
    "Inconsistent": (0.15, 0.30, 0.55),
}
_BASE_RENEWAL_PROBABILITY: dict[PatientProfile, float] = {
    "Consistent": 0.75,
    "Average": 0.60,
    "Inconsistent": 0.40,
}


def _timing_interval(
    rng: random.Random, days_supply: int, profile: PatientProfile
) -> int:
    behavior = rng.choices(
        ("early", "on_time", "late"),
        weights=_TIMING_WEIGHTS[profile],
        k=1,
    )[0]
    if behavior == "early":
        offset = -rng.randint(2, min(7, max(2, days_supply - 1)))
    elif behavior == "on_time":
        offset = rng.randint(-1, 1)
    else:
        offset = rng.randint(2, 16)
    return max(1, days_supply + offset)


def _profile_assignments(count: int, rng: random.Random) -> list[PatientProfile]:
    """Return a seeded shuffle with exact 40/40/20 proportions when divisible."""
    consistent_count = round(count * 0.40)
    average_count = round(count * 0.40)
    profiles: list[PatientProfile] = []
    for _ in range(consistent_count):
        profiles.append("Consistent")
    for _ in range(average_count):
        profiles.append("Average")
    for _ in range(count - consistent_count - average_count):
        profiles.append("Inconsistent")
    rng.shuffle(profiles)
    return profiles


def _renewal_probability(
    profile: PatientProfile,
    timing_gaps: Sequence[int],
    previous_fill_count: int,
) -> float:
    """Calculate renewal probability from history known at observation time."""
    probability = _BASE_RENEWAL_PROBABILITY[profile]
    if timing_gaps:
        on_time_rate = sum(abs(gap) <= 1 for gap in timing_gaps) / len(timing_gaps)
        if on_time_rate >= 0.70:
            probability += 0.10
        elif on_time_rate < 0.40:
            probability -= 0.10

        latest_gap = timing_gaps[-1]
        if abs(latest_gap) <= 1:
            probability += 0.05
        elif latest_gap > 1:
            probability -= 0.05

    if previous_fill_count >= 6:
        probability += 0.05
    elif previous_fill_count >= 3:
        probability += 0.02
    return min(1.0, max(0.0, probability))


def _make_patients(config: LongitudinalConfig, rng: random.Random) -> pd.DataFrame:
    rows = []
    profiles = _profile_assignments(config.patient_count, rng)
    for index, profile in enumerate(profiles, start=1):
        dob = date(1940, 1, 1) + timedelta(days=rng.randint(0, 60 * 365))
        first = _FIRST_NAMES[(index - 1) % len(_FIRST_NAMES)]
        last = _LAST_NAMES[((index - 1) // len(_FIRST_NAMES)) % len(_LAST_NAMES)]
        rows.append({
            "patient_id": f"SYN-PAT-{index:04d}",
            "patient_name": f"{first} {last} {index:04d}",
            "date_of_birth": dob,
            "behavior_profile": profile,
        })
    return pd.DataFrame(rows)


def _make_medications(config: LongitudinalConfig) -> pd.DataFrame:
    if config.medication_count > len(_DRUG_NAMES):
        raise ValueError(f"medication_count cannot exceed {len(_DRUG_NAMES)}.")
    return pd.DataFrame([
        {
            "medication_id": f"SYN-MED-{index:03d}",
            "ndc": f"9{index:010d}",
            "drug_name": _DRUG_NAMES[index - 1],
        }
        for index in range(1, config.medication_count + 1)
    ])


def generate_longitudinal_dataset(
    config: LongitudinalConfig | None = None,
) -> SyntheticLongitudinalDataset:
    """Generate fictional patients, prescriptions, and repeated medication fills."""
    config = config or LongitudinalConfig()
    if config.start_date >= config.end_date:
        raise ValueError("start_date must be before end_date.")
    rng = random.Random(config.seed)
    patients = _make_patients(config, rng)
    medications = _make_medications(config)
    prescription_rows: list[dict[str, object]] = []
    fill_rows: list[dict[str, object]] = []
    prescription_counter = 0
    fill_counter = 0

    study_days = (config.end_date - config.start_date).days
    for patient in patients.itertuples(index=False):
        patient_profile = cast(PatientProfile, patient.behavior_profile)
        primary_insurance = rng.choice(_INSURANCE_OPTIONS)
        medication_total = rng.choices((1, 2, 3, 4), weights=(0.28, 0.39, 0.24, 0.09), k=1)[0]
        medication_indices = rng.sample(range(len(medications)), medication_total)
        for medication_index in medication_indices:
            medication = medications.iloc[medication_index]
            next_fill_date = config.start_date + timedelta(days=rng.randint(0, min(study_days, 365)))
            episode_days_supply = rng.choices(
                _DAYS_SUPPLY_OPTIONS,
                weights=_DAYS_SUPPLY_WEIGHTS,
                k=1,
            )[0]
            daily_units = rng.choice((1, 1, 1, 2))
            total_episode_fills = 0
            episode_active = True
            prior_fill_date: date | None = None
            prior_fill_days_supply: int | None = None
            timing_gaps: list[int] = []

            while (
                episode_active
                and next_fill_date <= config.end_date
                and total_episode_fills < config.max_fills_per_patient_medication
            ):
                prescription_counter += 1
                prescription_id = f"SYN-RX-{prescription_counter:06d}"
                refills_authorized = rng.randint(1, 5)
                prescription_date = max(
                    config.start_date,
                    next_fill_date - timedelta(days=rng.randint(0, 7)),
                )
                prescribed_quantity = daily_units * episode_days_supply
                prescription_rows.append({
                    "prescription_id": prescription_id,
                    "patient_id": patient.patient_id,
                    "medication_id": medication["medication_id"],
                    "ndc": medication["ndc"],
                    "prescription_date": prescription_date,
                    "quantity_prescribed": prescribed_quantity,
                    "days_supply": episode_days_supply,
                    "refills_authorized": refills_authorized,
                })

                exhausted = True
                for refill_number in range(refills_authorized + 1):
                    if (
                        next_fill_date > config.end_date
                        or total_episode_fills >= config.max_fills_per_patient_medication
                    ):
                        episode_active = False
                        exhausted = False
                        break
                    fill_counter += 1
                    refills_remaining = refills_authorized - refill_number
                    quantity_variation = rng.choices((-1, 0, 1), weights=(0.08, 0.84, 0.08), k=1)[0]
                    quantity_billed = max(1, prescribed_quantity + quantity_variation * daily_units)
                    insurance = (
                        rng.choice(_INSURANCE_OPTIONS)
                        if rng.random() < 0.10
                        else primary_insurance
                    )
                    fill_rows.append({
                        "fill_id": f"SYN-FILL-{fill_counter:07d}",
                        "billing_id": f"SYN-BILL-{fill_counter:07d}",
                        "fill_date": next_fill_date,
                        "patient_id": patient.patient_id,
                        "prescription_id": prescription_id,
                        "medication_id": medication["medication_id"],
                        "ndc": medication["ndc"],
                        "drug_name": medication["drug_name"],
                        "quantity_billed": quantity_billed,
                        "days_supply": episode_days_supply,
                        "refill_number": refill_number,
                        "refills_remaining": refills_remaining,
                        "insurance_name": insurance[0],
                        "bin_number": insurance[1],
                    })
                    total_episode_fills += 1
                    if prior_fill_date is not None and prior_fill_days_supply is not None:
                        timing_gaps.append(
                            (next_fill_date - prior_fill_date).days
                            - prior_fill_days_supply
                        )
                    prior_fill_date = next_fill_date
                    prior_fill_days_supply = episode_days_supply

                    if refill_number < refills_authorized:
                        if rng.random() < 0.13:
                            episode_active = False
                            exhausted = False
                            break
                        next_fill_date += timedelta(
                            days=_timing_interval(
                                rng, episode_days_supply, patient_profile
                            )
                        )

                if not episode_active:
                    break
                if exhausted:
                    renewal_probability = _renewal_probability(
                        patient_profile,
                        timing_gaps,
                        total_episode_fills - 1,
                    )
                    if rng.random() >= renewal_probability:
                        break
                    next_fill_date += timedelta(
                        days=_timing_interval(
                            rng, episode_days_supply, patient_profile
                        )
                    )
                    if rng.random() < 0.18:
                        episode_days_supply = rng.choices(
                            _DAYS_SUPPLY_OPTIONS,
                            weights=_DAYS_SUPPLY_WEIGHTS,
                            k=1,
                        )[0]

    prescriptions = pd.DataFrame(prescription_rows).sort_values(
        ["prescription_date", "prescription_id"], kind="mergesort"
    ).reset_index(drop=True)
    fills = pd.DataFrame(fill_rows).sort_values(
        ["fill_date", "fill_id"], kind="mergesort"
    ).reset_index(drop=True)
    for frame, columns in (
        (patients, ("date_of_birth",)),
        (prescriptions, ("prescription_date",)),
        (fills, ("fill_date",)),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column])
    return SyntheticLongitudinalDataset(patients, medications, prescriptions, fills, config)
