import pandas as pd
from transformers import pipeline
import torch
from datasets import Dataset

csv_file = "cleaned_poems.csv"  # Replace with your CSV filename
df = pd.read_csv(csv_file)
print(f"The DataFrame now has {len(df)} rows.")
# Comprehensive candidate labels
candidate_labels = [
    "smutek", "miłość", "natura", "zemsta",
    "wojna", "wolność", "radość",
    "samotność", "przyjaźń", "przemijanie",
    "życie", "śmierć", "ojczyzna", "wiara",
    "poświęcenie", "przemiana"
]

dataset = Dataset.from_pandas(df)

classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    device=0,  # Use GPU
    batch_size=16  # Process 8 poems at a time
)

# Function to classify a batch of texts
def classify_batch(batch):
    texts = [text for text in batch["Content"] if text]  # Filter out None or empty texts
    if not texts:
        return {"Labels": []}  # Return empty labels if no valid text

    results = classifier(
        texts,
        candidate_labels=candidate_labels,
        multi_label=True,
        truncation=True
    )

    # Extract labels with scores > 0.75 for each text
    batch["Labels"] = [
        ", ".join(
            label for label, score in zip(result["labels"], result["scores"]) if score > 0.93
        )
        for result in results
    ]
    return batch


classified_dataset = dataset.map(classify_batch, batched=True, batch_size=8)

# Convert back to a pandas DataFrame
classified_df = classified_dataset.to_pandas()

# Save the results to a new CSV file
classified_df.to_csv("classified_poems.csv", index=False, encoding="utf-8")

print("Classification completed. Results saved to 'classified_poems.csv'.")

