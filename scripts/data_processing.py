import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from time import time
from pathlib import Path
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Download NLTK dependencies
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# Define project structure paths
PROJECT_ROOT = Path.cwd()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Create directories if they don't exist
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# Step 1: Load datasets directly from Kaggle
def load_dataset():
    print("⏳ Loading datasets from Kaggle...")

    fake_ds = kagglehub.load_dataset(
        KaggleDatasetAdapter.HUGGING_FACE,
        "clmentbisaillon/fake-and-real-news-dataset",
        "Fake.csv",
        hf_kwargs={"split": "all"}
    )
    true_ds = kagglehub.load_dataset(
        KaggleDatasetAdapter.HUGGING_FACE,
        "clmentbisaillon/fake-and-real-news-dataset",
        "True.csv",
        hf_kwargs={"split": "all"}
    )

    fake_df = fake_ds.to_pandas()
    true_df = true_ds.to_pandas()

    fake_df['label'] = 1  # Fake news
    true_df['label'] = 0  # Real news

    df = pd.concat([fake_df, true_df], ignore_index=True)

    print(f"✅ Dataset loaded: {len(df)} records")
    print(f"   - Fake news: {len(fake_df)} samples")
    print(f"   - Real news: {len(true_df)} samples")

    return df


# Step 2: Preprocess individual text entries
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]

    return ' '.join(tokens)
