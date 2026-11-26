from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


CATEGORICAL_COLUMNS = [
    "Account_Balance",
    "Payment_Status_of_Previous_Credit",
    "Purpose",
    "Value_Savings_Stocks",
    "Length_of_current_employment",
    "Sex_Marital_Status",
    "Guarantors",
    "Most_valuable_available_asset",
    "Concurrent_Credits",
    "Type_of_apartment",
    "Occupation",
    "Telephone",
    "Foreign_Worker",
]


def make_preprocessor(features):
    if isinstance(features, pd.DataFrame):
        categorical = [column for column in CATEGORICAL_COLUMNS if column in features.columns]
        numeric = [column for column in features.columns if column not in categorical]
    else:
        categorical_indices = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
        categorical = [index for index in categorical_indices if index < features.shape[1]]
        numeric = [index for index in range(features.shape[1]) if index not in categorical]

    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    numeric_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical),
            ("numeric", numeric_pipeline, numeric),
        ]
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    train_features = data.train_features
    test_features = data.test_features
    train_target = np.asarray(data.train_target).ravel()
    test_target = np.asarray(data.test_target).ravel()
    seed = int(data.random_seed)

    estimators = [
        (
            "random_forest",
            make_pipeline(
                make_preprocessor(train_features),
                RandomForestClassifier(
                    n_estimators=400,
                    max_features="sqrt",
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ),
        (
            "extra_trees",
            make_pipeline(
                make_preprocessor(train_features),
                ExtraTreesClassifier(
                    n_estimators=400,
                    max_features="sqrt",
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ),
        (
            "gradient_boosting",
            make_pipeline(
                make_preprocessor(train_features),
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.04,
                    max_depth=2,
                    min_samples_leaf=5,
                    subsample=0.85,
                    random_state=seed,
                ),
            ),
        ),
        (
            "logistic_regression",
            make_pipeline(
                make_preprocessor(train_features),
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ),
    ]

    model = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[2, 2, 2, 1],
        n_jobs=-1,
    )
    model.fit(train_features, train_target)

    classes = list(model.classes_)
    positive_index = classes.index(1) if 1 in classes else len(classes) - 1
    probabilities = model.predict_proba(test_features)[:, positive_index]
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name="Soft Voting: Random Forest, Extra Trees, Gradient Boosting and Logistic Regression",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(test_target, probabilities)),
        accuracy=float(accuracy_score(test_target, predictions)),
        precision=float(precision_score(test_target, predictions, zero_division=0)),
        recall=float(recall_score(test_target, predictions, zero_division=0)),
        f1=float(f1_score(test_target, predictions, zero_division=0)),
    )
