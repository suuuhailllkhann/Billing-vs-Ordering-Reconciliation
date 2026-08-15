"""Train-only native feature importance for the tuned tree models."""

from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from pharmacy_reconciliation.research.baselines import BASELINE_RANDOM_SEED
from pharmacy_reconciliation.research.features import FEATURE_COLUMNS, TARGET_COLUMN
from pharmacy_reconciliation.research.preparation import (
    MODEL_FEATURE_COLUMNS,
    chronological_split,
)
from pharmacy_reconciliation.research.tuning import FoldLocalRefillPreprocessor

TREE_MODEL_NAMES = ("Random Forest", "XGBoost", "LightGBM")


def tuned_tree_pipelines(
    seed: int = BASELINE_RANDOM_SEED,
) -> dict[str, Pipeline]:
    """Recreate the three selected Phase 2D tree configurations exactly."""
    classifiers: dict[str, Any] = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
            class_weight=None,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.02,
            max_depth=4,
            min_child_weight=1,
            subsample=1.0,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=10,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=7,
            max_depth=5,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.5,
            reg_lambda=1,
            random_state=seed,
            n_jobs=1,
            class_weight=None,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        ),
    }
    return {
        name: Pipeline([
            ("preprocessor", FoldLocalRefillPreprocessor()),
            ("classifier", classifier),
        ])
        for name, classifier in classifiers.items()
    }


def fit_tuned_tree_importances(
    observations: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Fit each tuned tree on the fixed Train period and rank native importances."""
    train = chronological_split(observations).train
    features = train.loc[:, FEATURE_COLUMNS]
    target = train[TARGET_COLUMN].astype("int8")
    results: dict[str, pd.DataFrame] = {}
    for name, model in tuned_tree_pipelines().items():
        model.fit(features, target)
        classifier = model.named_steps["classifier"]
        importances = classifier.feature_importances_
        if len(importances) != len(MODEL_FEATURE_COLUMNS):
            raise RuntimeError(
                f"{name} importance count does not match the model feature contract."
            )
        result = pd.DataFrame({
            "feature": MODEL_FEATURE_COLUMNS,
            "importance": importances,
        }).sort_values("importance", ascending=False, kind="mergesort")
        result = result.reset_index(drop=True)
        result["rank"] = result.index + 1
        results[name] = result.loc[:, ["feature", "importance", "rank"]]
    return results


def combined_tree_importance_ranks(
    importances: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Align the three native-importance ranks by feature."""
    combined = pd.DataFrame({"feature": MODEL_FEATURE_COLUMNS})
    for name in TREE_MODEL_NAMES:
        ranks = importances[name].loc[:, ["feature", "rank"]].rename(
            columns={"rank": f"{name} rank"}
        )
        combined = combined.merge(ranks, on="feature", validate="one_to_one")
    return combined.sort_values(
        [f"{name} rank" for name in TREE_MODEL_NAMES], kind="mergesort"
    ).reset_index(drop=True)
