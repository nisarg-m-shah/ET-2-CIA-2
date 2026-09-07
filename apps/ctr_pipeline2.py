"""
==============================================================================
 Click-Through Rate (CTR) Prediction on the Criteo 1M-Sample Dataset
 PySpark MLlib Pipeline — CIA 2 Machine Learning Project
==============================================================================

ALIGNMENT NOTE (fixed after cross-platform review vs ctr_pipeline_sklearn.py):
  - N_HASH_FEATURES was accidentally left at literal `2` in the uploaded copy
    of this file (a stale/broken edit) — restored to 4096, matching the
    config that actually produced your reported successful GBT run.
  - LogisticRegression maxIter raised 50 -> 200. At 50 iterations, LBFGS may
    not have converged on this sparse, high-dimensional problem, which is a
    likely real contributor to PySpark's LR underperforming both sklearn's
    LR and PySpark's own GBT. 200 is a middle ground between the original
    50 and sklearn's 1000 — high enough to reliably converge, without
    drastically increasing runtime on the cluster.
  - Train/test split is now stratified (see stratified_split()), matching
    sklearn's train_test_split(..., stratify=y). Spark's randomSplit() has
    no built-in stratify option, so this is implemented manually by
    splitting each class independently and recombining.
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
from pyspark.ml.feature import FeatureHasher, VectorAssembler, Imputer, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
INT_COLS = [f"intCol_{i}" for i in range(13)]
CAT_COLS = [f"catCol_{i}" for i in range(26)]
TARGET = "target"

# FIXED: was literal `2` (a broken edit) — this is the value that actually
# produced your reported successful GBT run (4,096 hashed buckets).
N_HASH_FEATURES = 2**7  # 128

SKEWED_COLS = ["intCol_1", "intCol_2", "intCol_4", "intCol_5", "intCol_6",
               "intCol_7", "intCol_8", "intCol_10", "intCol_11", "intCol_12"]


def build_spark(app_name="Criteo_CTR_Prediction"):
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "3g")
        .config("spark.executor.cores", "2")
        .config("spark.executor.memoryOverhead", "512m")
        .getOrCreate()
    )


def load_data(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumn(TARGET, F.col(TARGET).cast("integer"))
    for c in INT_COLS:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))
    for c in CAT_COLS:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("string"))
    return df


def explore(df, out_dir):
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
    for c in CAT_COLS:
        if c in df.columns:
            df = df.withColumn(c, F.when(F.col(c).isNull(), "missing").otherwise(F.col(c)))
    for c in SKEWED_COLS:
        if c in df.columns:
            df = df.withColumn(c, F.log1p(F.when(F.col(c) < 0, 0.0).otherwise(F.col(c))))
    return df


def stratified_split(df, train_frac=0.8, seed=42):
    """
    Spark's randomSplit() has no stratify option. This replicates sklearn's
    train_test_split(..., stratify=y) by splitting each class independently
    (via sampleBy, which samples an exact fraction per key) and recombining.

    IMPORTANT: joins/anti-joins on all columns break silently when duplicate
    rows exist (Spark matches by value, not row identity), so we attach a
    unique row ID first and split/join on that ID alone.
    """
    df_with_id = df.withColumn("_row_id", F.monotonically_increasing_id())
    fractions = {0: train_frac, 1: train_frac}
    train = df_with_id.sampleBy(TARGET, fractions=fractions, seed=seed)
    test = df_with_id.join(train.select("_row_id"), on="_row_id", how="left_anti")
    return train.drop("_row_id"), test.drop("_row_id")


def build_pipeline_stages(model_type="lr"):
    imputer = Imputer(
        inputCols=[c for c in INT_COLS],
        outputCols=[f"{c}_imp" for c in INT_COLS],
        strategy="median",
    )
    hasher = FeatureHasher(
        inputCols=CAT_COLS, outputCol="cat_features", numFeatures=N_HASH_FEATURES,
    )
    assembler = VectorAssembler(
        inputCols=[f"{c}_imp" for c in INT_COLS] + ["cat_features"],
        outputCol="raw_features",
    )
    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features", withMean=False, withStd=True
    )

    if model_type == "lr":
        # maxIter raised 50 -> 200 (see ALIGNMENT NOTE at top of file)
        clf = LogisticRegression(featuresCol="features", labelCol=TARGET, maxIter=200)
    elif model_type == "rf":
        clf = RandomForestClassifier(
            featuresCol="features", labelCol=TARGET, seed=42, numTrees=20
        )
    elif model_type == "gbt":
        clf = GBTClassifier(featuresCol="features", labelCol=TARGET, seed=42, maxIter=50)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

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
        labelCol=label_col, predictionCol="prediction", metricName="weightedPrecision"
    )
    mc_rec = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="weightedRecall"
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
            # regParam x elasticNetParam: matched to sklearn's C x penalty
            # grid below (0.0/L2 <-> large C/'l2', 0.1/L1 <-> C=10/'l1')
            grid = (
                ParamGridBuilder()
                .addGrid(clf.regParam, [0.0, 0.1])
                .addGrid(clf.elasticNetParam, [0.0, 1.0])
                .build()
            )
        elif model_type == "rf":
            grid = (
                ParamGridBuilder()
                .addGrid(clf.maxDepth, [5, 10])
                .addGrid(clf.numTrees, [30])
                .build()
            )
        else:  # gbt
            grid = (
                ParamGridBuilder()
                .addGrid(clf.maxDepth, [5])
                .addGrid(clf.maxIter, [20, 50])
                .build()
            )

        evaluator = BinaryClassificationEvaluator(
            labelCol=TARGET, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        )
        cv = CrossValidator(
            estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator,
            numFolds=2, parallelism=2, seed=42,
        )
        t0 = time.time()
        model = cv.fit(train)
        fit_time = time.time() - t0

        best_pipeline_model = model.bestModel

        model_path = os.path.join(out_dir, f"{model_type}_best_pipeline")

        best_pipeline_model.write().overwrite().save(model_path)

        print(f"[{model_type.upper()}] Saved model to {model_path}")        

        avg_cv_scores = model.avgMetrics
    else:
        t0 = time.time()
        best_pipeline_model = pipeline.fit(train)
        fit_time = time.time() - t0
        avg_cv_scores = None

    # Predictions on training data
    train_predictions = best_pipeline_model.transform(train)
    train_metrics = evaluate(train_predictions)

    # Predictions on test data
    test_predictions = best_pipeline_model.transform(test)
    test_metrics = evaluate(test_predictions)

    # Store both sets of metrics
    metrics = {
        "training_metrics": train_metrics,
        "test_metrics": test_metrics,
        "fit_time_sec": round(fit_time, 2)
    }
    if avg_cv_scores:
        metrics["cv_avg_auc_by_param_combo"] = avg_cv_scores

    if model_type in ("rf", "gbt"):
        try:
            tree_model = best_pipeline_model.stages[-1]
            importances = tree_model.featureImportances.toArray().tolist()
            metrics["top_20_feature_importance_indices"] = sorted(
                range(len(importances)), key=lambda i: -importances[i]
            )[:20]
        except Exception as e:
            metrics["feature_importance_error"] = str(e)

    cm = (
        test_predictions.groupBy(TARGET, "prediction")
        .count().toPandas()
        .pivot(index=TARGET, columns="prediction", values="count")
        .fillna(0).astype(int)
    )
    metrics["confusion_matrix"] = cm.to_dict()

    from pyspark.sql import functions as F2
    from pyspark.ml.functions import vector_to_array

    prob_col = test_predictions.withColumn("prob_1", vector_to_array("probability")[1])
    thresholds = [i / 10.0 for i in range(11)]
    roc_points = []
    total_pos = prob_col.filter(F2.col(TARGET) == 1).count()
    total_neg = prob_col.filter(F2.col(TARGET) == 0).count()
    for t in thresholds:
        tp = prob_col.filter((F2.col("prob_1") >= t) & (F2.col(TARGET) == 1)).count()
        fp = prob_col.filter((F2.col("prob_1") >= t) & (F2.col(TARGET) == 0)).count()
        tpr = tp / total_pos if total_pos else 0.0
        fpr = fp / total_neg if total_neg else 0.0
        roc_points.append({"threshold": t, "tpr": tpr, "fpr": fpr})
    metrics["roc_points"] = roc_points

    with open(os.path.join(out_dir, f"metrics_{model_type}.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"[{model_type.upper()}] " + json.dumps(metrics, indent=2, default=str))
    return best_pipeline_model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="./results")
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--models", default="lr,rf,gbt")
    args = parser.parse_args()
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    os.makedirs(args.output, exist_ok=True)
    spark = build_spark()

    print("[1/5] Loading data...")
    df = load_data(spark, args.input)
    if args.sample_frac < 1.0:
        df = df.sample(fraction=args.sample_frac, seed=42)

    print("[2/5] Running EDA...")
    explore(df, args.output)

    print("[3/5] Cleaning + feature engineering...")
    df = clean_and_engineer(df)

    print("[4/5] Splitting train/test (80/20, STRATIFIED)...")
    train, test = stratified_split(df, train_frac=0.8, seed=42)
    train = train.repartition(8).cache()
    test = test.repartition(8).cache()
    print(f"    train={train.count()}, test={test.count()}")

    print(f"[5/5] Training models: {model_list}")
    for i, model_type in enumerate(model_list, 1):
        print(f"  ({i}/{len(model_list)}) Training {model_type.upper()} with CV tuning...")
        run_model(train, test, model_type, args.output, tune=True)

    print(f"\nAll done. Metrics + EDA summary written to {args.output}/")
    spark.stop()


if __name__ == "__main__":
    main()