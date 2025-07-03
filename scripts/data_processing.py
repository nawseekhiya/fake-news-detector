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

# Step 3: Preprocessing workflow for full DataFrame
def preprocess_data(df):
    print("\n🧹 Preprocessing data...")
    start = time()

    raw_path = DATA_RAW / "raw_combined.csv"
    df.to_csv(raw_path, index=False)
    print(f"✅ Raw data saved to: {raw_path}")

    df = df.drop_duplicates(subset=['title', 'text'])
    df['text'] = df['text'].fillna('')
    df['title'] = df['title'].fillna('')
    df['full_text'] = df['title'] + ' ' + df['text']

     # Preprocess text in batches
    batch_size = 2000
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]

    processed_texts = []
    for i, batch in enumerate(batches):
        print(f"  Processing batch {i+1}/{len(batches)}")
        processed_batch = batch['full_text'].apply(preprocess_text)
        processed_texts.extend(processed_batch)

    df['processed_text'] = processed_texts

    # Create train-test split
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Save processed data
    train_path = DATA_PROCESSED / "train_processed.csv"
    test_path = DATA_PROCESSED / "test_processed.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Preprocessing completed in {time()-start:.2f} seconds")
    print(f"   - Train set ({len(train_df)} samples): {train_path}")
    print(f"   - Test set ({len(test_df)} samples): {test_path}")

    # Save additional artifacts for future reference
    sample_path = DATA_PROCESSED / "sample_processed_texts.csv"
    train_df[['processed_text', 'label']].head(100).to_csv(sample_path, index=False)
    print(f"   - Sample processed texts: {sample_path}")

    return train_df, test_df

if __name__ == "__main__":
    # Define project structure paths
    PROJECT_ROOT = Path.cwd()
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

    # Create directories if they don't exist
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    full_df = load_dataset()
    train_df, test_df = preprocess_data(full_df)