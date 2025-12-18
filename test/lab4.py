"""Improvement experiment: test Naive Bayes + simple preprocessing."""

from sklearn.model_selection import train_test_split

from src.lab_4 import RegexTokenizer, TfidfVectorizer, TextClassifier


def simple_preprocess(text: str) -> str:
    # lowercase and remove simple punctuation
    import re

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def test_improvement_nb_preprocessing():
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad.",
    ]
    labels = [1, 0, 1, 0, 1, 0]

    texts = [simple_preprocess(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.33, random_state=42
    )

    tokenizer = RegexTokenizer()
    vectorizer = TfidfVectorizer(tokenizer)
    clf = TextClassifier(vectorizer)

    # Train Multinomial Naive Bayes
    clf.fit(X_train, y_train, model="nb")
    preds = clf.predict(X_test)
    metrics = clf.evaluate(y_test, preds)

    print("Improved model metrics (NB + preprocessing):", metrics)

    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


if __name__ == "__main__":
    test_improvement_nb_preprocessing()
