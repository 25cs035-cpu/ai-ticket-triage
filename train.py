import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import load_and_prepare

# model name -> column in the CSV that holds the answer we want to predict
TARGETS = {
    "category": "type",
    "department": "queue",
    "priority": "priority",
}


def train_one_model(df, model_name, label_column):
    print("=" * 60)
    print(f"Training {model_name} model (predicting '{label_column}')")

    X = df["text"]          # features: the ticket text
    y = df[label_column]    # labels: the correct answers

    # 80% of tickets to learn from, 20% kept hidden to test the model fairly.
    # stratify=y keeps the label mix the same in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # A pipeline runs the steps in order. Because TF-IDF is inside it,
    # it learns its vocabulary from the TRAINING tickets only (no data leakage).
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)),
               ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Accuracy you get by always guessing the most common label (for comparison)
    majority_baseline = y_test.value_counts(normalize=True).max()

    print(f"Test accuracy:      {accuracy:.3f}")
    print(f"Always-guess-most-common baseline: {majority_baseline:.3f}")
    print()
    print(classification_report(y_test, predictions, zero_division=0))

    # Save the trained pipeline so the app can load it later
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, f"models/{model_name}_model.pkl")
    print(f"Saved: models/{model_name}_model.pkl")


if __name__ == "__main__":
    data = load_and_prepare()
    print("Tickets used:", len(data))

    for name, column in TARGETS.items():
        train_one_model(data, name, column)