from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class TextClassifier:
    """Simple text classifier wrapper using a Vectorizer and LogisticRegression."""

    def __init__(self, vectorizer: Any) -> None:
        self.vectorizer = vectorizer
        self._model: LogisticRegression | None = None

    def fit(self, texts: List[str], labels: List[int]) -> None:
        X = self.vectorizer.fit_transform(texts)
        # Ensure numpy array
        X = np.asarray(X)
        model = LogisticRegression(solver="liblinear")
        model.fit(X, labels)
        self._model = model

    def predict(self, texts: List[str]) -> List[int]:
        if self._model is None:
            raise ValueError("Model is not trained. Call fit() first.")
        X = self.vectorizer.transform(texts)
        X = np.asarray(X)
        preds = self._model.predict(X)
        return [int(p) for p in preds]

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }


__all__ = ["TextClassifier"]
