"""
compare_algorithms.py
─────────────────────
Trains and compares multiple ML algorithms for digital ad classification.
Saves the best model and prints a full comparison report.

Usage:
    python compare_algorithms.py

Why each algorithm:
    - LinearSVC      : Fast, high-dimensional text baseline (current model)
    - Logistic Reg.  : Probabilistic, interpretable, great for TF-IDF
    - Naive Bayes    : Classic NLP algorithm, very fast, strong on sparse data
    - Random Forest  : Ensemble method, handles non-linearity well
    - Gradient Boost : XGBoost-style boosting, powerful but slower on text
    - KNN            : Distance-based, included as a simple baseline
"""

import os
import re
import pickle
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DATA_REAL      = "data/ConcatenatedDigitalAdData.xlsx"
DATA_SYNTHETIC = "data/synthetic_data.csv"
MODEL_SAVE     = "notebook/model/adv_model.sav"
MODEL_BACKUP   = "notebook/model/adv_model_backup.sav"
REPORT_DIR     = "data/comparison_report"
os.makedirs(REPORT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─────────────────────────────────────────────
# TFIDF SHARED CONFIG
# ─────────────────────────────────────────────
def make_tfidf(max_features=20000, ngram=(1, 2)):
    return TfidfVectorizer(
        sublinear_tf=True, min_df=2, max_df=0.90,
        norm="l2", encoding="latin-1",
        ngram_range=ngram, stop_words="english",
        max_features=max_features,
    )

# ─────────────────────────────────────────────
# ALGORITHM DEFINITIONS
# Each entry: (name, reason, pipeline)
# ─────────────────────────────────────────────
ALGORITHMS = [
    (
        "LinearSVC (Current)",
        "High-dimensional linear classifier. Excellent for TF-IDF sparse vectors. "
        "Fast training and inference. Current production model.",
        Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(class_weight="balanced", C=1.0, max_iter=10000), cv=5
            )),
        ]),
    ),
    (
        "Logistic Regression",
        "Probabilistic linear model. Naturally outputs calibrated probabilities. "
        "Highly interpretable and performs extremely well on text classification. "
        "Strong alternative to SVM when confidence scores matter.",
        Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", LogisticRegression(
                class_weight="balanced", C=5.0,
                max_iter=1000, solver="lbfgs", multi_class="multinomial"
            )),
        ]),
    ),
    (
        "Multinomial Naive Bayes",
        "Classic Bayesian text classifier. Assumes feature independence (naive). "
        "Extremely fast and surprisingly competitive for bag-of-words/TF-IDF features. "
        "Best for quick baselines and when training data is limited.",
        Pipeline([
            ("tfidf", TfidfVectorizer(
                sublinear_tf=False, min_df=2, max_df=0.90,
                norm=None, encoding="latin-1",          # MNB needs non-negative values
                ngram_range=(1, 2), stop_words="english",
                max_features=20000,
            )),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
    ),
    (
        "Random Forest",
        "Ensemble of decision trees using bagging. Handles non-linear patterns and "
        "feature interactions. Less optimal for high-dimensional sparse text but "
        "included to demonstrate ensemble methods on TF-IDF features.",
        Pipeline([
            ("tfidf", make_tfidf(max_features=5000)),   # Reduced for memory
            ("clf", RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                random_state=42, n_jobs=-1
            )),
        ]),
    ),
    (
        "Gradient Boosting",
        "Sequential ensemble that corrects prior errors. Powerful on structured data. "
        "Included to show trade-off: strong accuracy potential but much slower to train "
        "on high-dimensional text compared to linear models.",
        Pipeline([
            ("tfidf", make_tfidf(max_features=3000, ngram=(1, 1))),  # Keep small for speed
            ("clf", GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1,
                max_depth=5, random_state=42
            )),
        ]),
    ),
    (
        "K-Nearest Neighbors",
        "Instance-based learner — classifies by the k closest training examples. "
        "Included as a simple non-parametric baseline. Generally weaker on "
        "high-dimensional TF-IDF vectors due to the curse of dimensionality.",
        Pipeline([
            ("tfidf", make_tfidf(max_features=3000)),
            ("clf", KNeighborsClassifier(n_neighbors=7, metric="cosine")),
        ]),
    ),
]

# ─────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────
def load_data():
    print("Loading data...")
    real_df = pd.read_excel(DATA_REAL)[["JobType", "Job_Description"]].dropna()
    real_df["JobType"] = real_df["JobType"].str.strip()

    if os.path.exists(DATA_SYNTHETIC):
        syn_df = pd.read_csv(DATA_SYNTHETIC)
        syn_df["JobType"] = syn_df["JobType"].str.strip()
        combined = pd.concat(
            [real_df[["Job_Description", "JobType"]], syn_df[["Job_Description", "JobType"]]],
            ignore_index=True,
        )
    else:
        print("  WARNING: Synthetic data not found — using real data only.")
        combined = real_df[["Job_Description", "JobType"]]

    combined["Cleaned_Text"] = combined["Job_Description"].apply(clean_text)
    combined = combined[combined["Cleaned_Text"].str.len() > 20].reset_index(drop=True)
    print(f"  Total samples: {len(combined)}")
    print(combined["JobType"].value_counts().to_string())
    return combined

# ─────────────────────────────────────────────
# TRAIN & EVALUATE ALL ALGORITHMS
# ─────────────────────────────────────────────
def evaluate_all(X_train, X_test, y_train, y_test, X_all, y_all):
    results = []
    trained_pipelines = {}

    for name, reason, pipeline in ALGORITHMS:
        print(f"\n{'='*60}")
        print(f"  Algorithm : {name}")
        print(f"  Why used  : {reason[:80]}...")
        print(f"{'='*60}")

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = pipeline.predict(X_test)
        acc    = accuracy_score(y_test, y_pred) * 100
        f1_mac = f1_score(y_test, y_pred, average="macro") * 100
        f1_wt  = f1_score(y_test, y_pred, average="weighted") * 100

        # 5-fold CV
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_sc  = cross_val_score(pipeline, X_all, y_all, cv=cv, scoring="accuracy")
        cv_mean = cv_sc.mean() * 100
        cv_std  = cv_sc.std() * 100

        print(f"  Test Accuracy   : {acc:.2f}%")
        print(f"  Macro F1        : {f1_mac:.2f}%")
        print(f"  Weighted F1     : {f1_wt:.2f}%")
        print(f"  5-Fold CV       : {cv_mean:.2f}% (+/- {cv_std:.2f}%)")
        print(f"  Train Time      : {train_time:.1f}s")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred))

        results.append({
            "Algorithm"      : name,
            "Test Accuracy"  : acc,
            "Macro F1"       : f1_mac,
            "Weighted F1"    : f1_wt,
            "CV Mean"        : cv_mean,
            "CV Std"         : cv_std,
            "Train Time (s)" : round(train_time, 2),
            "y_pred"         : y_pred,
            "Reason"         : reason,
        })
        trained_pipelines[name] = pipeline

    return results, trained_pipelines

# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────
COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4"]

def plot_accuracy_comparison(results_df):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    names   = results_df["Algorithm"].tolist()
    accs    = results_df["Test Accuracy"].tolist()
    cv_means= results_df["CV Mean"].tolist()
    cv_stds = results_df["CV Std"].tolist()

    x   = np.arange(len(names))
    w   = 0.35

    bars1 = ax.bar(x - w/2, accs,     w, label="Test Accuracy", color=COLORS[0], alpha=0.9)
    bars2 = ax.bar(x + w/2, cv_means, w, label="CV Mean",       color=COLORS[1], alpha=0.9,
                   yerr=cv_stds, capsize=4, error_kw={"ecolor": "#ffffff", "alpha": 0.6})

    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", color="#f8fafc", fontsize=9)
    ax.set_ylabel("Accuracy (%)", color="#f8fafc")
    ax.set_title("Algorithm Accuracy Comparison", color="#f8fafc", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                color="#f8fafc", fontsize=8)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"\nSaved: {path}")


def plot_f1_comparison(results_df):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    names  = results_df["Algorithm"].tolist()
    f1_mac = results_df["Macro F1"].tolist()
    f1_wt  = results_df["Weighted F1"].tolist()
    x      = np.arange(len(names))
    w      = 0.35

    ax.bar(x - w/2, f1_mac, w, label="Macro F1",    color=COLORS[2], alpha=0.9)
    ax.bar(x + w/2, f1_wt,  w, label="Weighted F1", color=COLORS[4], alpha=0.9)

    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", color="#f8fafc", fontsize=9)
    ax.set_ylabel("F1 Score (%)", color="#f8fafc")
    ax.set_title("F1 Score Comparison (Macro vs Weighted)", color="#f8fafc", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "f1_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"Saved: {path}")


def plot_training_time(results_df):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    names  = results_df["Algorithm"].tolist()
    times  = results_df["Train Time (s)"].tolist()
    colors = [COLORS[i % len(COLORS)] for i in range(len(names))]

    bars = ax.barh(names, times, color=colors, alpha=0.9)
    ax.set_xlabel("Training Time (seconds)", color="#f8fafc")
    ax.set_title("Training Time per Algorithm", color="#f8fafc", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{t:.1f}s", va="center", color="#f8fafc", fontsize=9)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "training_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(name, y_test, y_pred, classes):
    cm     = confusion_matrix(y_test, y_pred, labels=classes)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                cbar_kws={"shrink": 0.8})

    ax.set_xlabel("Predicted", color="#f8fafc")
    ax.set_ylabel("Actual",    color="#f8fafc")
    safe = name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
    ax.set_title(f"Confusion Matrix — {name}", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#94a3b8", labelsize=8)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, f"cm_{safe}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"Saved: {path}")


def plot_radar(results_df):
    """Radar chart comparing all algorithms across 3 metrics."""
    cats   = ["Test Accuracy", "Macro F1", "Weighted F1", "CV Mean"]
    N      = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for i, row in results_df.iterrows():
        values = [row[c] for c in cats]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                label=row["Algorithm"], color=COLORS[i % len(COLORS)])
        ax.fill(angles, values, alpha=0.07, color=COLORS[i % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color="#f8fafc", size=10)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#94a3b8", size=8)
    ax.tick_params(colors="#334155")
    ax.grid(color="#334155", linestyle="--", alpha=0.5)
    ax.spines["polar"].set_color("#334155")

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=9)
    ax.set_title("Multi-Metric Radar Chart", color="#f8fafc",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "radar_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"Saved: {path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load data
    df = load_data()
    X  = df["Cleaned_Text"]
    y  = df["JobType"]
    classes = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # 2. Train & evaluate all
    results_raw, trained = evaluate_all(X_train, X_test, y_train, y_test, X, y)

    # 3. Build summary table
    summary_cols = ["Algorithm", "Test Accuracy", "Macro F1", "Weighted F1",
                    "CV Mean", "CV Std", "Train Time (s)"]
    results_df = pd.DataFrame(results_raw)[summary_cols].sort_values(
        "Test Accuracy", ascending=False
    ).reset_index(drop=True)

    print("\n" + "="*70)
    print("  FINAL COMPARISON SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)

    # Identify best
    best_name = results_df.iloc[0]["Algorithm"]
    best_acc  = results_df.iloc[0]["Test Accuracy"]
    print(f"\n[BEST] Algorithm  : {best_name}  ({best_acc:.2f}% accuracy)")

    # 4. Save comparison table
    csv_path = os.path.join(REPORT_DIR, "algorithm_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved table      : {csv_path}")

    # Reasons report
    reasons_path = os.path.join(REPORT_DIR, "algorithm_reasons.txt")
    with open(reasons_path, "w") as f:
        f.write("WHY EACH ALGORITHM WAS USED\n")
        f.write("="*60 + "\n\n")
        for r in results_raw:
            f.write(f"Algorithm: {r['Algorithm']}\n")
            f.write(f"Reason   : {r['Reason']}\n\n")
    print(f"Saved reasons: {reasons_path}")

    # 5. Generate all plots
    print("\nGenerating charts...")
    plot_accuracy_comparison(results_df)
    plot_f1_comparison(results_df)
    plot_training_time(results_df)
    plot_radar(results_df)

    # Confusion matrices for each algorithm
    for r in results_raw:
        plot_confusion_matrix(r["Algorithm"], y_test, r["y_pred"], classes)

    # 6. Save best model
    best_pipeline = trained[best_name]
    if os.path.exists(MODEL_SAVE):
        shutil.copy2(MODEL_SAVE, MODEL_BACKUP)
        print(f"\nBacked up old model  -> {MODEL_BACKUP}")
    with open(MODEL_SAVE, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"Saved best model     -> {MODEL_SAVE}  ({best_name})")

    # 7. Print algorithm explanations
    print("\n" + "="*70)
    print("  ALGORITHM EXPLANATIONS")
    print("="*70)
    for r in results_raw:
        rank = results_df[results_df["Algorithm"] == r["Algorithm"]].index[0] + 1
        print(f"\n#{rank} {r['Algorithm']}")
        print(f"   {r['Reason']}")

    print("\nDone! Check data/comparison_report/ for all charts and CSVs.")
    print("Restart the Streamlit app to load the best model.")
