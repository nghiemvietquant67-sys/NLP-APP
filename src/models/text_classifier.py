"""A minimal TextClassifier example for Lab 5 and Lab 6 demo.

This module provides a simple sk-learn pipeline for classifying text using TF-IDF and LogisticRegression.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class TextClassifier:
    def __init__(self, max_features=5000):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features)),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
