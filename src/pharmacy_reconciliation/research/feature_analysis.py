"""Train-only coefficient analysis for the tuned logistic regression."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pharmacy_reconciliation.research.baselines import BASELINE_RANDOM_SEED
from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN
from pharmacy_reconciliation.research.preparation import (
    MODEL_FEATURE_COLUMNS,
    chronological_split,
)
from pharmacy_reconciliation.research.tuning import FoldLocalRefillPreprocessor

TUNED_LOGISTIC_C = 0.01


def tuned_logistic_pipeline(
    seed: int = BASELINE_RANDOM_SEED,
) -> Pipeline:
    """Recreate the selected Phase 2D logistic configuration exactly."""
    return Pipeline([
        ("preprocessor", FoldLocalRefillPreprocessor()),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=TUNED_LOGISTIC_C,
            solver="saga",
            max_iter=5000,
            random_state=seed,
            class_weight=None,
            l1_ratio=0.0,
        )),
    ])


def fit_tuned_logistic_coefficients(observations: pd.DataFrame) -> pd.DataFrame:
    """Fit on the fixed Train period and return all standardized coefficients."""
    train = chronological_split(observations).train
    model = tuned_logistic_pipeline()
    model.fit(train.loc[:, FEATURE_COLUMNS], train[TARGET_COLUMN].astype("int8"))

    classifier = model.named_steps["classifier"]
    coefficients = classifier.coef_[0]
    if len(coefficients) != len(MODEL_FEATURE_COLUMNS):
        raise RuntimeError("Fitted coefficient count does not match the model feature contract.")

    result = pd.DataFrame({
        "feature": MODEL_FEATURE_COLUMNS,
        "coefficient": coefficients,
    })
    result["absolute_coefficient"] = result["coefficient"].abs()
    result["direction"] = result["coefficient"].map(
        lambda value: "positive" if value >= 0 else "negative"
    )
    return result.sort_values(
        "absolute_coefficient", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
