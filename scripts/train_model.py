from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from pathlib import Path
from time import time
from data_processing import preprocess_data
import pandas as pd

# Access the variables from data_processing.py
train_df, test_df = preprocess_data()

# Prepare data
X_train = train_df['processed_text']
y_train = train_df['label']
X_test = test_df['processed_text']
y_test = test_df['label']

# Training logic
def train_model(X_train, y_train):
    print("\n🤖 Training model...")
    start = time()

    # Create pipeline
    model = make_pipeline(
    TfidfVectorizer(max_features=10000, ngram_range=(1, 2)),
    LogisticRegression(
        C=0.5,
        max_iter=100,
        random_state=42
    )
    )

    model.fit(X_train, y_train)

    train_time = time() - start
    print(f"✅ Training completed in {train_time:.2f} seconds")
    return model, train_time


# Train the model
model, train_time = train_model(X_train, y_train)