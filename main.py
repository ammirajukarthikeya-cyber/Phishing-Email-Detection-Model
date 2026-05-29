"""
Phishing Email Detection - Main Training & Evaluation Script

This script:
1. Generates (or loads) the email dataset
2. Extracts features using the FeatureExtractor pipeline
3. Trains an ensemble classifier
4. Evaluates performance with accuracy, confusion matrix, and feature importance
5. Saves the trained model and feature extractor for later prediction
"""

import os
import sys
import time
import joblib
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.feature_extraction import FeatureExtractor
from src.model import PhishingDetector
from src.utils import print_banner, print_section, print_metric, display_results, COLORS


def main():
    """Main training and evaluation pipeline."""
    print_banner()
    start_time = time.time()

    # -- Step 1: Load or Generate Dataset --
    print_section("Step 1: Loading Dataset", "[DATA]")

    data_path = os.path.join("data", "emails.csv")

    if not os.path.exists(data_path):
        print("  Dataset not found. Generating synthetic dataset...")
        from data.generate_dataset import generate_dataset
        df = generate_dataset(n_samples=2000, output_path=data_path)
    else:
        df = pd.read_csv(data_path)
        print(f"  Loaded dataset: {len(df)} samples")

    # Show dataset stats
    n_phishing = (df["label"] == 1).sum()
    n_safe = (df["label"] == 0).sum()
    print_metric("Total Samples", len(df))
    print_metric("Phishing Emails", f"{n_phishing} ({n_phishing / len(df) * 100:.1f}%)")
    print_metric("Safe Emails", f"{n_safe} ({n_safe / len(df) * 100:.1f}%)")

    # -- Step 2: Feature Extraction --
    print_section("Step 2: Extracting Features", "[FEATURES]")

    # Combine subject and body for feature extraction
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

    extractor = FeatureExtractor(max_tfidf_features=100)
    X = extractor.fit_transform(df["text"])
    y = df["label"].values

    print_metric("Feature Matrix Shape", f"{X.shape[0]} samples x {X.shape[1]} features")
    print_metric("Handcrafted Features", 11)
    print_metric("TF-IDF Features", X.shape[1] - 11)

    # -- Step 3: Train Model --
    print_section("Step 3: Training Model", "[TRAIN]")

    detector = PhishingDetector(random_state=42)
    detector.split_data(X, y, test_size=0.2)
    detector.train()

    # -- Step 4: Evaluate Model --
    print_section("Step 4: Evaluating Model", "[EVAL]")

    y_true, y_pred, accuracy = detector.evaluate()

    # Get feature importances
    importances = detector.get_feature_importances()
    feature_names = extractor.feature_names_

    # Display all results
    display_results(y_true, y_pred, importances, feature_names, output_dir="outputs")

    # -- Step 5: Cross-Validation --
    print_section("Step 5: Cross-Validation", "[CV]")
    detector.cross_validate(X, y, cv=5)

    # -- Step 6: Save Model & Extractor --
    print_section("Step 6: Saving Model", "[SAVE]")

    detector.save_model("outputs/phishing_model.pkl")
    joblib.dump(extractor, "outputs/feature_extractor.pkl")
    print(f"  [SAVED] Feature extractor saved to: outputs/feature_extractor.pkl")

    # -- Summary --
    elapsed = time.time() - start_time
    print(f"\n{COLORS['cyan']}{COLORS['bold']}{'=' * 62}")
    print(f"  DONE! Pipeline complete in {elapsed:.2f}s")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Model saved to: outputs/phishing_model.pkl")
    print(f"  Plots saved to: outputs/")
    print(f"{'=' * 62}{COLORS['end']}\n")


if __name__ == "__main__":
    main()
