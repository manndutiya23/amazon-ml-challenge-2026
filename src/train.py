import pandas as pd
import numpy as np

from pathlib import Path
from xgboost import XGBClassifier

from sklearn.metrics import precision_score, recall_score


FEATURE_COLUMNS = [
    "name_missing",
    "address_missing",
    "name_exact",
    "name_ratio",
    "name_token_set",
    "name_token_sort",
    "name_jaccard",
    "name_char_jaccard",
    "name_length_difference",
    "name_length_ratio",
    "address_exact",
    "address_ratio",
    "address_token_set",
    "address_token_sort",
    "address_jaccard",
    "address_char_jaccard",
    "address_length_difference",
    "address_length_ratio",
    "digit_overlap",
    "country_same",
    "both_exact",
]


def train_model(X_train, y_train):
    """
    Train the first baseline pairwise matching model.
    """

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def predict_probabilities(model, X):
    return model.predict_proba(X)[:, 1]


if __name__ == "__main__":
    print("Model training module loaded successfully.")
    print("Waiting for candidate-pair training data.")