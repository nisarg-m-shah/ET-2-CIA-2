"""
Visualize CTR model results — NO PySpark needed here at all.

Run this on your laptop (not inside the Spark container) after copying the
results/*.json files out of ./apps/results/. Only needs: pandas, matplotlib.

    pip install pandas matplotlib
    python plot_results.py --results ./apps/results --models lr rf gbt
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(results_dir, model_type):
    path = os.path.join(results_dir, f"metrics_{model_type}.json")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def plot_roc_curves(all_metrics, out_dir):
    plt.figure(figsize=(6, 6))
    for model_type, m in all_metrics.items():
        if not m or "roc_points" not in m:
            continue
        pts = sorted(m["roc_points"], key=lambda p: p["fpr"])
        fpr = [p["fpr"] for p in pts]
        tpr = [p["tpr"] for p in pts]
        plt.plot(fpr, tpr, marker="o", label=f"{model_type.upper()} (AUC={m['AUC_ROC']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — CTR Prediction")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "roc_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_confusion_matrices(all_metrics, out_dir):
    n = sum(1 for m in all_metrics.values() if m and "confusion_matrix" in m)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    i = 0
    for model_type, m in all_metrics.items():
        if not m or "confusion_matrix" not in m:
            continue
        cm = pd.DataFrame(m["confusion_matrix"]).fillna(0)
        ax = axes[i]
        im = ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks(range(len(cm.columns)))
        ax.set_xticklabels([f"pred={c}" for c in cm.columns])
        ax.set_yticks(range(len(cm.index)))
        ax.set_yticklabels([f"true={r}" for r in cm.index])
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, int(cm.values[r, c]), ha="center", va="center")
        ax.set_title(model_type.upper())
        i += 1
    plt.tight_layout()
    out_path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_metric_comparison(all_metrics, out_dir):
    rows = []
    for model_type, m in all_metrics.items():
        if not m:
            continue
        rows.append({
            "model": model_type.upper(),
            "AUC_ROC": m.get("AUC_ROC"),
            "AUC_PR": m.get("AUC_PR"),
            "Accuracy": m.get("Accuracy"),
            "F1": m.get("F1"),
        })
    if not rows:
        return
    df = pd.DataFrame(rows).set_index("model")
    ax = df.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print("\n" + df.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="./results", help="Path to results directory")
    parser.add_argument("--models", nargs="+", default=["lr", "rf", "gbt"])
    parser.add_argument("--out", default=None, help="Where to save plots (default: same as --results)")
    args = parser.parse_args()
    out_dir = args.out or args.results
    os.makedirs(out_dir, exist_ok=True)

    all_metrics = {m: load_metrics(args.results, m) for m in args.models}
    plot_roc_curves(all_metrics, out_dir)
    plot_confusion_matrices(all_metrics, out_dir)
    plot_metric_comparison(all_metrics, out_dir)


if __name__ == "__main__":
    main()