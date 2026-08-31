"""
==============================================================================
 CTR Prediction — pandas + scikit-learn version (NO PySpark)
 Same cleaning/feature logic as ctr_pipeline.py, for a direct timing comparison.
==============================================================================

Run this locally (needs: pandas, numpy, scikit-learn — no Spark, no Docker):

    pip install pandas numpy scikit-learn
    python ctr_pipeline_sklearn.py --input Criteo_1M_with_nans.csv --output ./results_sklearn

Expect this to be FASTER on a small sample (no distributed-computing overhead
to set up) but to hit a wall as data size grows past what fits in one
machine's RAM — that contrast is the actual point of the comparison for your
report's "Technical Execution" section.
==============================================================================
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INT_COLS = [f"intCol_{i}" for i in range(13)]
CAT_COLS = [f"catCol_{i}" for i in range(26)]
TARGET = "target"
SKEWED_COLS = ["intCol_1", "intCol_2", "intCol_4", "intCol_5", "intCol_6",
               "intCol_7", "intCol_8", "intCol_10", "intCol_11", "intCol_12"]
N_HASH_FEATURES = 2 ** 12


def load_and_clean(path):
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype(int)
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("missing").astype(str)
    for c in SKEWED_COLS:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))
    return df


class HasherWrapper:
    """Wraps sklearn's FeatureHasher for a DataFrame of categorical columns —
    mirrors PySpark's FeatureHasher so the two pipelines use equivalent logic."""
    def __init__(self, cols, n_features):
        self.cols = cols
        self.hasher = FeatureHasher(n_features=n_features, input_type="string")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = X[self.cols].astype(str).apply(
            lambda r: [f"{c}={v}" for c, v in zip(self.cols, r)], axis=1
        )
        return self.hasher.transform(rows)


def build_pipeline(model_type):
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])

    if model_type == "lr":
        clf = LogisticRegression(max_iter=1000)
        grid = {"clf__C": [1.0, 10.0], "clf__penalty": ["l2"]}
    elif model_type == "rf":
        clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid = {"clf__max_depth": [5, 10], "clf__n_estimators": [50]}
    elif model_type == "gbt":
        clf = GradientBoostingClassifier(random_state=42)
        grid = {"clf__max_depth": [5], "clf__n_estimators": [20, 50]}
    else:
        raise ValueError(model_type)

    return numeric_pipe, clf, grid


def run_model(model_type, X_train, X_test, y_train, y_test, out_dir):
    numeric_pipe, clf, grid = build_pipeline(model_type)

    num_transformed_train = numeric_pipe.fit_transform(X_train[INT_COLS])
    num_transformed_test = numeric_pipe.transform(X_test[INT_COLS])

    hasher = HasherWrapper(CAT_COLS, N_HASH_FEATURES)
    cat_transformed_train = hasher.transform(X_train)
    cat_transformed_test = hasher.transform(X_test)

    from scipy.sparse import hstack, csr_matrix
    Xtr = hstack([csr_matrix(num_transformed_train), cat_transformed_train]).tocsr()
    Xte = hstack([csr_matrix(num_transformed_test), cat_transformed_test]).tocsr()

    t0 = time.time()
    search = GridSearchCV(
        Pipeline([("clf", clf)]), grid, scoring="roc_auc", cv=2, n_jobs=-1
    )
    search.fit(Xtr, y_train)
    fit_time = time.time() - t0

    best = search.best_estimator_
    y_pred = best.predict(Xte)
    y_prob = best.predict_proba(Xte)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_points = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]

    metrics = {
        "AUC_ROC": roc_auc_score(y_test, y_prob),
        "AUC_PR": average_precision_score(y_test, y_prob),
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "Weighted_Precision": precision_score(y_test, y_pred, average="weighted"),
        "Weighted_Recall": recall_score(y_test, y_pred, average="weighted"),
        "fit_time_sec": round(fit_time, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_points": roc_points,
        "best_params": search.best_params_,
    }

    with open(os.path.join(out_dir, f"metrics_{model_type}.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"[{model_type.upper()}] " + json.dumps(
        {k: v for k, v in metrics.items()
         if not isinstance(v, (list, dict))}, indent=2
    ))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="./results_sklearn")
    parser.add_argument("--models", default="lr,rf,gbt")
    args = parser.parse_args()
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    os.makedirs(args.output, exist_ok=True)

    t0 = time.time()
    print("[1/3] Loading + cleaning data...")
    df = load_and_clean(args.input)
    load_time = time.time() - t0
    print(f"    rows={len(df)}, load+clean time={load_time:.2f}s")

    print("[2/3] Splitting train/test (80/20)...")
    X = df[INT_COLS + CAT_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[3/3] Training models: {model_list}")
    for model_type in model_list:
        print(f"  Training {model_type.upper()}...")
        run_model(model_type, X_train, X_test, y_train, y_test, args.output)

    total_time = time.time() - t0
    print(f"\nAll done in {total_time:.2f}s total. Results in {args.output}/")


if __name__ == "__main__":
    main()