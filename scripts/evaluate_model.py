import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from time import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from pathlib import Path
import joblib
from data_processing import preprocess_data

# Define paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "fake_news_model.pkl"

# Access the variables from data_processing.py
train_df, test_df = preprocess_data()

# Prepare data
X_train = train_df['processed_text']
y_train = train_df['label']
X_test = test_df['processed_text']
y_test = test_df['label']

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

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    print("\n📊 Evaluating model...")
    start = time()

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    test_time = time() - start

    print(f"✅ Evaluation completed in {test_time:.2f} seconds")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_time': test_time
    }

# Evaluate
metrics = evaluate_model(model, X_test, y_test)