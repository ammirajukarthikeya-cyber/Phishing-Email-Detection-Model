"""
Phishing Detection Model
Trains and evaluates a Random Forest classifier for email classification.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


class PhishingDetector:
    """
    Phishing email detection model using an ensemble of classifiers.

    Uses a VotingClassifier combining:
    - Random Forest (primary)
    - Gradient Boosting
    - Logistic Regression
    """

    def __init__(self, random_state=42):
        self.random_state = random_state

        # Individual classifiers
        self._rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        self._gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
        )
        self._lr = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )

        # Ensemble
        self.model = VotingClassifier(
            estimators=[
                ("rf", self._rf),
                ("gb", self._gb),
                ("lr", self._lr),
            ],
            voting="soft",
        )

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, X, y, test_size=0.2):
        """
        Split features and labels into train/test sets.

        Args:
            X: Feature matrix.
            y: Labels.
            test_size: Fraction of data for testing.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        print(f"  Training samples: {self.X_train.shape[0]}")
        print(f"  Testing samples:  {self.X_test.shape[0]}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, X_train=None, y_train=None):
        """
        Train the ensemble model.

        Args:
            X_train: Training features (uses stored if None).
            y_train: Training labels (uses stored if None).
        """
        X = X_train if X_train is not None else self.X_train
        y = y_train if y_train is not None else self.y_train

        if X is None or y is None:
            raise ValueError("No training data. Call split_data() first or provide X_train, y_train.")

        print("  Training ensemble model (Random Forest + Gradient Boosting + Logistic Regression)...")
        self.model.fit(X, y)
        print("  [OK] Model training complete!")

    def predict(self, X):
        """Predict labels for given features."""
        return self.model.predict(X)

    def evaluate(self, X_test=None, y_test=None):
        """
        Evaluate model on test data.

        Returns:
            Tuple of (y_true, y_pred, accuracy).
        """
        X = X_test if X_test is not None else self.X_test
        y = y_test if y_test is not None else self.y_test

        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        return y, y_pred, acc

    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation and return scores.

        Args:
            X: Feature matrix.
            y: Labels.
            cv: Number of folds.

        Returns:
            Array of cross-validation scores.
        """
        print(f"  Running {cv}-fold cross-validation...")
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        print(f"  CV Scores: {scores}")
        print(f"  Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return scores

    def get_feature_importances(self):
        """
        Get feature importances from the Random Forest component.

        Returns:
            numpy array of feature importances.
        """
        # Access the fitted RF estimator from the VotingClassifier
        rf_model = self.model.named_estimators_["rf"]
        return rf_model.feature_importances_

    def save_model(self, filepath="outputs/phishing_model.pkl"):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"  [SAVED] Model saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath="outputs/phishing_model.pkl"):
        """Load a trained model from disk."""
        detector = cls()
        detector.model = joblib.load(filepath)
        print(f"  [LOADED] Model loaded from: {filepath}")
        return detector
