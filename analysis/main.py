import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from models.classifier import PoemClassifier


def prepare_input(row):
    return row['content'] + " [RYMY] " + row['rhyme_input']

def create_label_mappings(df, label_column):
    unique_labels = sorted(df[label_column].dropna().unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label

def prepare_dataset(df, tokenizer, text_column="content", label_column="type"):
    df = df.copy()
    df["labels"] = df[label_column].map(label2id).astype('int64')
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "token_type_ids", "labels"]]
    )

    tokenized_dataset.set_format("torch")

    return tokenized_dataset


if __name__ == "__main__":
    df = pd.read_csv("data/wiersze.csv")
    # df['input_text'] = df.apply(prepare_input, axis=1)
    label2id, id2label = create_label_mappings(df, label_column="type")

    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-large-cased")

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_dataset = prepare_dataset(train_df, tokenizer, text_column="content", label_column="type")
    eval_dataset = prepare_dataset(eval_df, tokenizer, text_column="content", label_column="type")
    test_dataset = prepare_dataset(test_df, tokenizer, text_column="content", label_column="type")

    classifier = PoemClassifier(
        model_name="allegro/herbert-large-cased",
        id2label=id2label,
        label2id=label2id,
        num_labels=len(id2label)
    )

    results = classifier.finetune(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        output_dir="./type_herbert_large_classifier",
        optimize_hyperparams=True,
        n_trials=1
    )

    print("----- Wyniki klasyfikatora:", results)
