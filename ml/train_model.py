
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


# -----------------------------------------
# 1. Project paths
# -----------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "cleaned_job_postings.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "data"

MODEL_FILE = MODEL_DIR / "jobshield_model.joblib"
METRICS_FILE = REPORT_DIR / "model_evaluation.json"
MATRIX_FILE = REPORT_DIR / "confusion_matrix.png"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# 2. Configuration
# -----------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

MAX_FEATURES = 50000


# -----------------------------------------
# 3. Load dataset
# -----------------------------------------

def main():

    print("\nJOBSHIELD AI - MODEL TRAINING")
    print("=" * 50)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")
    print("Total records:", len(df))

    required_columns = ["combined_text", "fraudulent"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    # Keep only valid labels and non-empty text
    df["fraudulent"] = pd.to_numeric(
        df["fraudulent"], errors="coerce"
    )

    df = df.dropna(
        subset=["combined_text", "fraudulent"]
    ).copy()

    df = df[df["fraudulent"].isin([0, 1])].copy()

    df["fraudulent"] = df["fraudulent"].astype(int)

    df["combined_text"] = (
        df["combined_text"].fillna("").astype(str)
    )

    df = df[
        df["combined_text"].str.strip().ne("")
    ].copy()

    X = df["combined_text"]
    y = df["fraudulent"]

    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    if y.nunique() != 2:
        raise ValueError(
            "Training requires both target classes (0 and 1)."
        )

    # -----------------------------------------
    # 4. Train-test split
    # -----------------------------------------

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # -----------------------------------------
    # 5. Build ML pipeline
    # -----------------------------------------

    print("\nBuilding TF-IDF + Logistic Regression model...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=MAX_FEATURES,
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                dtype="float32",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
                solver="liblinear",
            ),
        ),
    ])

    # -----------------------------------------
    # 6. Train model
    # -----------------------------------------

    print("\nTraining started...")
    print("This may take a few minutes.")

    model.fit(X_train, y_train)

    print("Training completed!")

    # -----------------------------------------
    # 7. Predictions
    # -----------------------------------------

    print("\nEvaluating model...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    pr_auc = average_precision_score(
        y_test, y_prob
    )

    roc_auc = roc_auc_score(
        y_test, y_prob
    )

    # -----------------------------------------
    # 8. Print evaluation report
    # -----------------------------------------

    print("\nCLASSIFICATION REPORT")
    print("-" * 50)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Legitimate",
                "Potentially Fraudulent",
            ],
            zero_division=0,
        )
    )

    print("\nKEY METRICS")
    print("-" * 50)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # -----------------------------------------
    # 9. Confusion matrix
    # -----------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Legitimate",
            "Potentially Fraudulent",
        ],
        yticklabels=[
            "Legitimate",
            "Potentially Fraudulent",
        ],
    )

    plt.title("JobShield AI - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    plt.savefig(
        MATRIX_FILE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -----------------------------------------
    # 10. Save model
    # -----------------------------------------

    joblib.dump(model, MODEL_FILE)

    # -----------------------------------------
    # 11. Save metrics
    # -----------------------------------------

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "dataset_records": int(len(df)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=[
                "Legitimate",
                "Potentially Fraudulent",
            ],
            output_dict=True,
            zero_division=0,
        ),
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    # -----------------------------------------
    # 12. Final results
    # -----------------------------------------

    print("\nSAVED FILES")
    print("-" * 50)

    print("Model:", MODEL_FILE)
    print("Metrics:", METRICS_FILE)
    print("Confusion matrix:", MATRIX_FILE)

    print("\nJOBSHIELD AI MODEL TRAINING COMPLETED!")


if __name__ == "__main__":
    main()
