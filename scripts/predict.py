import os
import joblib
from scripts.data_processing import preprocess_text

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fake_news_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}\nPlease ensure it is trained and saved.")


def predict(title, text):
    if not isinstance(title, str) or not isinstance(text, str):
        return {
            "label": "Invalid",
            "confidence": "0%",
            "error": "Both title and text must be strings."
        }

    combined = f"{title.strip()} {text.strip()}".strip()

    if not combined:
        return {
            "label": "Invalid",
            "confidence": "0%",
            "error": "Combined input is empty after stripping."
        }

    processed = preprocess_text(combined)
    if not processed:
        return {
            "label": "Invalid",
            "confidence": "0%",
            "error": "Text could not be processed meaningfully."
        }

    pred = model.predict([processed])[0]
    proba = model.predict_proba([processed])[0]

    return {
        "label": "Fake" if pred == 1 else "Real",
        "confidence": f"{max(proba) * 100:.2f}%",
        "probabilities": {
            "Real": round(proba[0], 4),
            "Fake": round(proba[1], 4)
        }
    }

# An optional test run
if __name__ == "__main__":
    sample_title = "NASA confirms moon base construction"
    sample_text = "NASA has announced plans to build a moon base starting 2026..."
    result = predict(sample_title, sample_text)
    print(result)
