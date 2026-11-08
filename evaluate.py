import joblib
from sklearn.model_selection import train_test_split

from preprocessing import load_and_prepare
from train import TARGETS   # same model names and label columns as training

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

if __name__ == "__main__":
    data = load_and_prepare()

    for name, column in TARGETS.items():
        model = joblib.load(f"models/{name}_model.pkl")

        X = data["text"]
        y = data[column]

        # Same split settings as train.py, so these are tickets the model never trained on
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        probabilities = model.predict_proba(X_test)
        confidence = probabilities.max(axis=1)                       # how sure for each ticket
        predicted = model.classes_[probabilities.argmax(axis=1)]     # predicted label
        correct = predicted == y_test.to_numpy()                     # True where prediction is right

        print("=" * 62)
        print(f"{name} model (predicting '{column}')")
        print(f"Overall accuracy on test tickets: {correct.mean():.3f}")
        print()
        print("Threshold | Auto-handled | Accuracy on auto-handled | Sent to review")

        for t in THRESHOLDS:
            keep = confidence >= t          # tickets the model is confident enough about
            share = keep.mean()
            accuracy = correct[keep].mean() if keep.any() else float("nan")
            print(f"{t:>9.2f} | {share:>12.0%} | {accuracy:>24.3f} | {1 - share:>14.0%}")
