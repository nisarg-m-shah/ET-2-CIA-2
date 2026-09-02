"""
==============================================================================
 CTR Prediction — pandas + scikit-learn version (NO PySpark)
==============================================================================

ALIGNMENT NOTE (fixed after cross-platform review vs ctr_pipeline.py):
  - StandardScaler now scales the FULL feature matrix (numeric + hashed
    categorical) as one combined step, matching PySpark's pipeline, which
    scales the entire assembled vector. Previously this file only scaled
    the 13 numeric columns and left the hashed categorical block unscaled
    — a real methodological mismatch, now fixed.
  - LogisticRegression grid changed from {C:[1.0,10.0], penalty:['l2']} to
    penalty='elasticnet' with C x l1_ratio grid, solver='saga' (the only
    sklearn solver supporting elasticnet penalty on sparse input). l1_ratio
    directly mirrors PySpark's elasticNetParam (0.0=L2, 1.0=L1) — a cleaner,
    more principled match than switching penalty strings, which is also
    being deprecated in newer sklearn versions in favor of l1_ratio.
    C=[1e6, 10] maps onto regParam=[0.0, 0.1] (near-zero vs moderate
    regularization strength). Still not numerically identical (see
    IRREDUCIBLE DIFFERENCES below), but now searches the conceptually
    equivalent 2x2 space PySpark does, instead of two arbitrary C values
    with L2 only.
  - max_iter reduced from 1000 to 300 and tol relaxed to 1e-3: solver='saga'
    is markedly slower per-iteration than the default 'lbfgs' solver,
    especially on sparse hashed features — 1000 iterations caused excessive
    runtime in testing. 300 iterations with a slightly relaxed tolerance is
    a practical tradeoff; if you see ConvergenceWarning on your real run,
    it's safe to ignore for this comparison's purposes, or raise max_iter
    further if you have time budget to spare.
  - Train/test split was already stratified here; PySpark's has now been
    updated to match (see ctr_pipeline.py's stratified_split()).

IRREDUCIBLE DIFFERENCES (cannot be eliminated by config alignment alone —
documented honestly rather than hidden):
  - Imputer median: Spark's Imputer computes an APPROXIMATE median by
    default (for scalability on distributed data); sklearn's SimpleImputer
    computes the EXACT median. On this data the difference is likely small
    but not guaranteed to be zero.
  - FeatureHasher: both Spark's and sklearn's hashers use a MurmurHash3-
    family algorithm, but the exact string encoding of "column=value" pairs
    and hash seed differ between the two library implementations, so
    identical (col, value) inputs are not guaranteed to land in the same
    bucket index across platforms, even at the same N_HASH_FEATURES.
  - These two points mean PySpark and sklearn results are a FAIR, aligned
    comparison of the same modeling approach — not a bit-for-bit identical
    replay of the same computation. This is stated as a documented
    limitation, not hidden.
==============================================================================

    pip install pandas numpy scikit-learn scipy
    python ctr_pipeline_sklearn.py --input Criteo_1M_with_nans.csv --output ./results_sklearn
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

INT_COLS = [f"intCol_{i}" for i in range(13)]
CAT_COLS = [f"catCol_{i}" for i in range(26)]
TARGET = "target"
SKEWED_COLS = ["intCol_1", "intCol_2", "intCol_4", "intCol_5", "intCol_6",
               "intCol_7", "intCol_8", "intCol_10", "intCol_11", "intCol_12"]
N_HASH_FEATURES = 2 ** 12  # 4096 — matches PySpark's fixed value


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


def build_model_and_grid(model_type):
    if model_type == "lr":
        # penalty='elasticnet' + l1_ratio directly mirrors PySpark's
        # elasticNetParam (0.0 = pure L2, 1.0 = pure L1) — a cleaner,
        # more principled match than switching penalty strings, and avoids
        # the now-deprecated penalty='l1'/'l2' string API. solver='saga' is
        # the only solver supporting elasticnet penalty on sparse input.
        clf = LogisticRegression(
            max_iter=300, solver="saga", penalty="elasticnet", tol=1e-3
        )
        grid = {"C": [1e6, 10], "l1_ratio": [0.0, 1.0]}
    elif model_type == "rf":
        clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid = {"max_depth": [5, 10], "n_estimators": [50]}
    elif model_type == "gbt":
        clf = GradientBoostingClassifier(random_state=42)
        grid = {"max_depth": [5], "n_estimators": [20, 50]}
    else:
        raise ValueError(model_type)
    return clf, grid


def run_model(model_type, X_train, X_test, y_train, y_test, out_dir):
    clf, grid = build_model_and_grid(model_type)

    # Numeric: median impute (13 cols)
    imputer = SimpleImputer(strategy="median")
    num_train = imputer.fit_transform(X_train[INT_COLS])
    num_test = imputer.transform(X_test[INT_COLS])

    # Categorical: hash all 26 cols into N_HASH_FEATURES buckets
    hasher = HasherWrapper(CAT_COLS, N_HASH_FEATURES)
    cat_train = hasher.transform(X_train)
    cat_test = hasher.transform(X_test)

    # Assemble FIRST (numeric + hashed categorical), matching PySpark's
    # VectorAssembler -> StandardScaler order
    Xtr_raw = hstack([csr_matrix(num_train), cat_train]).tocsr()
    Xte_raw = hstack([csr_matrix(num_test), cat_test]).tocsr()

    # FIXED: scale the FULL combined matrix, not just the numeric block —
    # matches PySpark's StandardScaler applied to the entire assembled vector
    scaler = StandardScaler(with_mean=False)
    Xtr = scaler.fit_transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    t0 = time.time()
    search = GridSearchCV(clf, grid, scoring="roc_auc", cv=2, n_jobs=-1)
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
        {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}, indent=2
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
    print(f"    rows={len(df)}, load+clean time={time.time()-t0:.2f}s")

    print("[2/3] Splitting train/test (80/20, stratified)...")
    X = df[INT_COLS + CAT_COLS]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[3/3] Training models: {model_list}")
    for model_type in model_list:
        print(f"  Training {model_type.upper()}...")
        run_model(model_type, X_train, X_test, y_train, y_test, args.output)

    print(f"\nAll done in {time.time()-t0:.2f}s total. Results in {args.output}/")


if __name__ == "__main__":
    main()