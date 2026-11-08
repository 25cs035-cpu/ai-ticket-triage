import joblib

from preprocessing import clean_text

# Where each trained model was saved by train.py
MODEL_FILES = {
    "category": "models/category_model.pkl",
    "department": "models/department_model.pkl",
    "priority": "models/priority_model.pkl",
}

# If ANY model is less sure than this, the ticket is flagged for a human.
# 0.50 is just a starting value; we will tune it later with real data.
DEFAULT_THRESHOLD = 0.50


def load_models():
    """Load the three saved models from disk."""
    return {name: joblib.load(path) for name, path in MODEL_FILES.items()}


def predict_ticket(text, models, threshold=DEFAULT_THRESHOLD):
    """Predict category, department and priority for one ticket message."""
    cleaned = clean_text(text)   # same cleaning that was used for training

    result = {}
    needs_review = False

    for name, model in models.items():
        # predict_proba gives one probability per possible label
        probabilities = model.predict_proba([cleaned])[0]
        best_index = probabilities.argmax()          # position of the biggest probability

        result[name] = str(model.classes_[best_index])           # the predicted label
        result[name + "_confidence"] = float(probabilities[best_index])  # how sure the model is

        if probabilities[best_index] < threshold:
            needs_review = True

    result["needs_review"] = needs_review
    return result


# Quick test: runs only when you run this file directly
if __name__ == "__main__":
    models = load_models()

    samples = [
        "I was charged twice for my invoice this month and I need a refund.",
        "Our production server is down and customers cannot log in. Please fix this urgently!",
        "Ich kann mich seit heute Morgen nicht mehr in mein Konto einloggen.",
        "Hello, could you tell me more about your pricing plans?",
    ]

    for text in samples:
        result = predict_ticket(text, models)
        print("-" * 60)
        print("Ticket:", text)
        print(f"Category:   {result['category']} ({result['category_confidence']:.0%})")
        print(f"Department: {result['department']} ({result['department_confidence']:.0%})")
        print(f"Priority:   {result['priority']} ({result['priority_confidence']:.0%})")
        print("Needs human review:", result["needs_review"])
