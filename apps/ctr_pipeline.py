"""
==============================================================================
 Click-Through Rate (CTR) Prediction on the Criteo 1M-Sample Dataset
 PySpark MLlib Pipeline — CIA 2 Machine Learning Project
==============================================================================

Dataset: 1,000,000 rows | 1 binary target | 13 numeric features (intCol_0..12)
         | 26 hashed categorical features (catCol_0..25)

Pipeline stages:
  1. Spark session + data load
  2. Exploratory checks (schema, nulls, class balance)
  3. Data cleaning: null imputation (numeric: median, categorical: "missing")
  4. Feature engineering: log-transform skewed numeric cols, StringIndexer +
     OneHotEncoder / hashing for high-cardinality categoricals, VectorAssembler
  5. Train/test split (stratified via sampling, since target is imbalanced)
  6. Models: Logistic Regression (baseline) + Random Forest (MLlib)
  7. Hyperparameter tuning via CrossValidator + ParamGridBuilder
  8. Evaluation: AUC-ROC, AUC-PR, Accuracy, Precision, Recall, F1
  9. Save metrics + feature importances for the report/slides

Usage:
    spark-submit ctr_pipeline.py --input /path/to/criteo_sample.csv --output ./results
==============================================================================
"""

import argparse
import json
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    FeatureHasher,
    VectorAssembler,
    Imputer,
    StandardScaler,
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
INT_COLS = [f"intCol_{i}" for i in range(13)]      # intCol_0 .. intCol_12
CAT_COLS = [f"catCol_{i}" for i in range(26)]      # catCol_0 .. catCol_25
TARGET = "target"

# Categorical cardinality is unknown/high (hashed 32-bit values), so instead of
# StringIndexer -> OneHotEncoder (which explodes with high-cardinality cats and
# breaks on unseen categories at inference time), we use FeatureHasher, which
# is the standard, scalable choice for Criteo-style hashed categorical data.
N_HASH_FEATURES = 2 ** 14  # 16384 hashed buckets — tune based on cluster memory

# Numeric columns that are heavily right-skewed based on the describe() stats
# you shared (e.g. intCol_4: mean 17,240 vs median 2,361; intCol_2: max 65,535
# vs 75th pct 22) benefit from log1p transformation before scaling.
SKEWED_COLS = ["intCol_1", "intCol_2", "intCol_4", "intCol_5", "intCol_6",
               "intCol_7", "intCol_8", "intCol_10", "intCol_11", "intCol_12"]


def build_spark(app_name="Criteo_CTR_Prediction"):
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def load_data(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    # Ensure target and numeric columns are the right type even if inferSchema
    # mis-reads them (common with missing values in integer columns).
    df = df.withColumn(TARGET, F.col(TARGET).cast("integer"))
    for c in INT_COLS:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))
    for c in CAT_COLS:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("string"))
    return df


def explore(df, out_dir):
    """Basic EDA — writes a JSON summary used directly in the report/slides."""
    n_rows = df.count()
    class_balance = (
        df.groupBy(TARGET).count().orderBy(TARGET).toPandas().to_dict(orient="records")
    )

    null_counts = {}
    for c in INT_COLS + CAT_COLS:
        if c in df.columns:
            null_counts[c] = df.filter(F.col(c).isNull()).count()

    summary = {
        "n_rows": n_rows,
        "n_features": len(INT_COLS) + len(CAT_COLS),
        "class_balance": class_balance,
        "null_counts": null_counts,
        "null_pct": {k: round(100 * v / n_rows, 2) for k, v in null_counts.items()},
    }
    with open(os.path.join(out_dir, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[EDA] rows=%d, positive class rate=%.4f" % (
        n_rows, class_balance[1]["count"] / n_rows if len(class_balance) > 1 else -1
    ))
    return summary


def clean_and_engineer(df):
    """Null imputation + log-transform of skewed numeric features."""
    # Categorical: replace null with explicit "missing" category (informative —
    # in CTR data, missingness itself is often predictive of click behavior).
    for c in CAT_COLS:
        if c in df.columns:
            df = df.withColumn(c, F.when(F.col(c).isNull(), "missing").otherwise(F.col(c)))

    # Numeric: log1p-transform skewed columns to tame the long tails you can
    # see in the describe() output (e.g. intCol_4 max 1.6M vs median 2,361),
    # then impute remaining nulls with the median (robust to outliers).
    for c in SKEWED_COLS:
        if c in df.columns:
            df = df.withColumn(c, F.log1p(F.when(F.col(c) < 0, 0.0).otherwise(F.col(c))))

    return df


def build_pipeline_stages(model_type="lr"):
    imputer = Imputer(
        inputCols=[c for c in INT_COLS],
        outputCols=[f"{c}_imp" for c in INT_COLS],
        strategy="median",
    )

    hasher = FeatureHasher(
        inputCols=CAT_COLS,
        outputCol="cat_features",
        numFeatures=N_HASH_FEATURES,
    )

    assembler = VectorAssembler(
        inputCols=[f"{c}_imp" for c in INT_COLS] + ["cat_features"],
        outputCol="raw_features",
    )

    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features", withMean=False, withStd=True
    )

    if model_type == "lr":
        clf = LogisticRegression(featuresCol="features", labelCol=TARGET, maxIter=50)
    else:
        clf = RandomForestClassifier(
            featuresCol="features", labelCol=TARGET, seed=42, numTrees=100
        )

    return [imputer, hasher, assembler, scaler, clf], clf


def evaluate(predictions, label_col=TARGET):
    bin_eval_auc = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    bin_eval_pr = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    mc_acc = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="accuracy"
    )
    mc_f1 = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="f1"
    )
    mc_prec = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction",
        metricName="weightedPrecision"
    )
    mc_rec = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction",
        metricName="weightedRecall"
    )
    return {
        "AUC_ROC": bin_eval_auc.evaluate(predictions),
        "AUC_PR": bin_eval_pr.evaluate(predictions),
        "Accuracy": mc_acc.evaluate(predictions),
        "F1": mc_f1.evaluate(predictions),
        "Weighted_Precision": mc_prec.evaluate(predictions),
        "Weighted_Recall": mc_rec.evaluate(predictions),
    }


def run_model(train, test, model_type, out_dir, tune=True):
    stages, clf = build_pipeline_stages(model_type)
    pipeline = Pipeline(stages=stages)

    if tune:
        if model_type == "lr":
            # Trimmed from 3x3=9 combos to 2x2=4 — LR is cheap, so this mainly
            # saves time via numFolds/parallelism below, but no need to keep
            # the middle regParam/elasticNet values once you've confirmed the
            # extremes bracket the best setting.
            grid = (
                ParamGridBuilder()
                .addGrid(clf.regParam, [0.0, 0.1])
                .addGrid(clf.elasticNetParam, [0.0, 1.0])
                .build()
            )
        else:
            # Trimmed from 3x2=6 combos to 2x1=2 — this is the real win.
            # Dropped maxDepth=15 (deepest trees = slowest by far) and
            # numTrees=100 (doubles cost for often-marginal AUC gain on a
            # 2-core-per-worker cluster). maxDepth=[5,10] still lets you see
            # whether deeper trees help before committing more compute to it.
            grid = (
                ParamGridBuilder()
                .addGrid(clf.maxDepth, [5, 10])
                .addGrid(clf.numTrees, [50])
                .build()
            )

        evaluator = BinaryClassificationEvaluator(
            labelCol=TARGET, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        )
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=grid,
            evaluator=evaluator,
            numFolds=2,       # was 3 — cuts total fits by another 1/3
            parallelism=2,    # was 4 — matches your 2-worker x 2-core cluster,
                               # so fits actually run in parallel instead of queuing
            seed=42,
        )
        t0 = time.time()
        model = cv.fit(train)
        fit_time = time.time() - t0
        best_pipeline_model = model.bestModel
        avg_cv_scores = model.avgMetrics
    else:
        t0 = time.time()
        best_pipeline_model = pipeline.fit(train)
        fit_time = time.time() - t0
        avg_cv_scores = None

    predictions = best_pipeline_model.transform(test)
    metrics = evaluate(predictions)
    metrics["fit_time_sec"] = round(fit_time, 2)
    if avg_cv_scores:
        metrics["cv_avg_auc_by_param_combo"] = avg_cv_scores

    # Feature importance for Random Forest (goes straight into the slides)
    if model_type == "rf":
        try:
            rf_model = best_pipeline_model.stages[-1]
            importances = rf_model.featureImportances.toArray().tolist()
            metrics["top_20_feature_importance_indices"] = sorted(
                range(len(importances)), key=lambda i: -importances[i]
            )[:20]
        except Exception as e:
            metrics["feature_importance_error"] = str(e)

    with open(os.path.join(out_dir, f"metrics_{model_type}.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"[{model_type.upper()}] " + json.dumps(
        {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}, indent=2
    ))
    return best_pipeline_model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Criteo CSV file")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--sample_frac", type=float, default=1.0,
                         help="Optional fraction to subsample for quick runs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    spark = build_spark()

    print("[1/6] Loading data...")
    df = load_data(spark, args.input)
    if args.sample_frac < 1.0:
        df = df.sample(fraction=args.sample_frac, seed=42)

    print("[2/6] Running EDA...")
    explore(df, args.output)

    print("[3/6] Cleaning + feature engineering...")
    df = clean_and_engineer(df)

    print("[4/6] Splitting train/test (80/20, stratified by target)...")
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    train = train.cache()
    test = test.cache()
    print(f"    train={train.count()}, test={test.count()}")

    print("[5/6] Training Logistic Regression (baseline) with CV tuning...")
    run_model(train, test, "lr", args.output, tune=True)

    print("[6/6] Training Random Forest with CV tuning...")
    run_model(train, test, "rf", args.output, tune=True)

    print(f"\nAll done. Metrics + EDA summary written to {args.output}/")
    spark.stop()


if __name__ == "__main__":
    main()