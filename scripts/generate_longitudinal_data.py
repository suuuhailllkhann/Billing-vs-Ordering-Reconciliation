"""Generate the reviewed Phase 2A synthetic longitudinal CSV artifacts."""

import json
from pathlib import Path

from pharmacy_reconciliation.research.features import build_supply_end_observation_result
from pharmacy_reconciliation.research.integrity import summarize_longitudinal_dataset
from pharmacy_reconciliation.synthetic.longitudinal import (
    LongitudinalConfig,
    generate_longitudinal_dataset,
)

ROOT = Path(__file__).parents[1]
OUTPUT_DIR = ROOT / "data" / "synthetic" / "longitudinal"


def main() -> None:
    config = LongitudinalConfig()
    dataset = generate_longitudinal_dataset(config)
    observation_result = build_supply_end_observation_result(
        dataset.fills,
        dataset.prescriptions,
        config.end_date,
    )
    observations = observation_result.observations
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_formats = {
        "patients.csv": (dataset.patients, ["date_of_birth"]),
        "medications.csv": (dataset.medications, []),
        "prescriptions.csv": (dataset.prescriptions, ["prescription_date"]),
        "fills.csv": (dataset.fills, ["fill_date"]),
        "refill_observations.csv": (
            observations,
            ["observation_date", "expected_supply_end_date"],
        ),
    }
    for filename, (frame, date_columns) in date_formats.items():
        output = frame.copy()
        for column in date_columns:
            output[column] = output[column].dt.strftime("%Y-%m-%d")
        output.to_csv(OUTPUT_DIR / filename, index=False, lineterminator="\n")
    summary = summarize_longitudinal_dataset(dataset, observations, observation_result)
    (OUTPUT_DIR / "integrity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
