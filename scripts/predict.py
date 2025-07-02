import joblib
from data_processing import preprocess_text

model = joblib.load("models/fake_news_model.pkl")

def predict(title, text):
    full_text = f"{title} {text}"
    processed = preprocess_text(full_text)
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
    sample_title = "NASA confirms"
    sample_text = "moon base construction starts in 2026."
    result = predict(sample_title, sample_text)
    print(result)