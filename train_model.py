"""
Train the NLP model for the Extreme vs Moderate Opinion Identifier.

Recommended for final project:
1. Run this file in Google Colab or locally.
2. It downloads the large Yelp Review Full dataset from Hugging Face.
3. It trains a 5-level opinion intensity model.
4. It saves model artifacts in /models for the Streamlit app.

Run:
    python train_model.py --sample-size 150000 --test-size 30000

For stronger final results on Colab:
    python train_model.py --sample-size 300000 --test-size 50000
"""

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from nlp_utils import clean_text, STAR_LABELS, star_to_bucket


def dataset_to_frame(split, sample_size: int, seed: int) -> pd.DataFrame:
    ds = load_dataset("Yelp/yelp_review_full", split=split)
    sample_size = min(sample_size, len(ds))
    ds = ds.shuffle(seed=seed).select(range(sample_size))
    df = ds.to_pandas()
    df = df.rename(columns={"label": "star_label"})
    df["star_level"] = df["star_label"] + 1
    df["broad_category"] = df["star_label"].apply(star_to_bucket)
    df["detail_label"] = df["star_label"].map(STAR_LABELS)
    return df[["text", "star_label", "star_level", "broad_category", "detail_label"]]


def to_binary_extreme(y):
    y = np.array(y)
    return np.where((y == 0) | (y == 4), "Extreme", "Moderate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=150000, help="Training rows to sample from Yelp train split.")
    parser.add_argument("--test-size", type=int, default=30000, help="Testing rows to sample from Yelp test split.")
    parser.add_argument("--max-features", type=int, default=75000, help="TF-IDF max features.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading large Yelp Review Full dataset...")
    train_df = dataset_to_frame("train", args.sample_size, args.seed)
    test_df = dataset_to_frame("test", args.test_size, args.seed)

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print("\nTraining distribution:")
    print(train_df["detail_label"].value_counts().sort_index())

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            lowercase=False,
            ngram_range=(1, 2),
            max_features=args.max_features,
            min_df=3,
            max_df=0.90,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            max_iter=20,
            tol=1e-3,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )),
    ])

    start = time.time()
    pipeline.fit(train_df["text"], train_df["star_label"])
    train_seconds = round(time.time() - start, 2)

    y_pred = pipeline.predict(test_df["text"])
    y_true = test_df["star_label"].values

    star_accuracy = accuracy_score(y_true, y_pred)
    star_f1_macro = f1_score(y_true, y_pred, average="macro")

    y_true_binary = to_binary_extreme(y_true)
    y_pred_binary = to_binary_extreme(y_pred)
    binary_accuracy = accuracy_score(y_true_binary, y_pred_binary)
    binary_f1_macro = f1_score(y_true_binary, y_pred_binary, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, labels=["Extreme", "Moderate"], zero_division=0
    )

    metrics = {
        "project_name": "Extreme vs Moderate Opinion Identifier",
        "dataset": "Yelp/yelp_review_full",
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "model": "TF-IDF n-grams + SGD Logistic Regression",
        "training_seconds": train_seconds,
        "star_level_accuracy": round(float(star_accuracy), 4),
        "star_level_macro_f1": round(float(star_f1_macro), 4),
        "binary_extreme_moderate_accuracy": round(float(binary_accuracy), 4),
        "binary_extreme_moderate_macro_f1": round(float(binary_f1_macro), 4),
        "binary_class_metrics": {
            "Extreme": {
                "precision": round(float(precision[0]), 4),
                "recall": round(float(recall[0]), 4),
                "f1": round(float(f1[0]), 4),
                "support": int(support[0]),
            },
            "Moderate": {
                "precision": round(float(precision[1]), 4),
                "recall": round(float(recall[1]), 4),
                "f1": round(float(f1[1]), 4),
                "support": int(support[1]),
            },
        },
        "five_level_classification_report": classification_report(
            y_true,
            y_pred,
            target_names=[STAR_LABELS[i] for i in range(5)],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_labels": [STAR_LABELS[i] for i in range(5)],
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "label_mapping": {
            "1 star": "Extreme Negative",
            "2 stars": "Moderate Negative",
            "3 stars": "Neutral / Mixed",
            "4 stars": "Moderate Positive",
            "5 stars": "Extreme Positive",
            "Project category": "Extreme = 1 or 5 stars; Moderate = 2, 3, or 4 stars",
        },
    }

    model_path = output_dir / "opinion_intensity_pipeline.joblib"
    metrics_path = output_dir / "metrics.json"
    profile_path = output_dir / "dataset_profile.json"

    joblib.dump(pipeline, model_path, compress=3)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    profile = {
        "train_distribution": train_df["detail_label"].value_counts().to_dict(),
        "test_distribution": test_df["detail_label"].value_counts().to_dict(),
        "sample_training_examples": train_df.sample(min(10, len(train_df)), random_state=args.seed).to_dict(orient="records"),
    }
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print("\nModel saved:", model_path)
    print("Metrics saved:", metrics_path)
    print("\nImportant results:")
    print(json.dumps({
        "star_level_accuracy": metrics["star_level_accuracy"],
        "star_level_macro_f1": metrics["star_level_macro_f1"],
        "binary_extreme_moderate_accuracy": metrics["binary_extreme_moderate_accuracy"],
        "binary_extreme_moderate_macro_f1": metrics["binary_extreme_moderate_macro_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
