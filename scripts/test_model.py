from time import time
from data_processing import preprocess_text
from pathlib import Path
import joblib

# Define paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fake_news_model.pkl"

# Function to load the model
def load_model(path=MODEL_PATH):
    try:
        model = joblib.load(path)
        print(f"✅ Model loaded successfully from {path}")
        return model
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {path}")
        return None
model = load_model()
if model is None:
    raise RuntimeError("Model could not be loaded. Please check the path and try again.")

# Test with sample predictions
def test_samples(model):
    samples = [
        ("Scientists confirm climate change is accelerating at unprecedented rates. New data shows ice caps melting 40% faster than previous estimates.", 0),
        ("BREAKING: Celebrities injecting themselves with alien DNA to gain immortality, secret documents reveal!", 1),
        ("New study finds that regular exercise can reduce the risk of heart disease by up to 30%", 0),
        ("Government secretly adding mind-control chemicals to drinking water supplies nationwide", 1)
    ]

    print("\n🧪 Sample Predictions:")
    for text, true_label in samples:
        processed = preprocess_text(text)
        prediction = model.predict([processed])[0]
        proba = model.predict_proba([processed])[0]

        print(f"\nText: {text[:80]}...")
        print(f"True: {'Fake' if true_label == 1 else 'Real'}")
        print(f"Pred: {'Fake' if prediction == 1 else 'Real'}")
        print(f"Confidence: {max(proba)*100:.1f}%")
        print(f"Probabilities: [Real: {proba[0]:.4f}, Fake: {proba[1]:.4f}]")

# Test samples
test_samples(model)