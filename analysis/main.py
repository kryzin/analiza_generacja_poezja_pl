import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from models.form_classifier import PoemFormClassifier


def prepare_dataset(df, tokenizer, text_column="content", label_column="type"):
    df = df.copy()
    df["labels"] = df[label_column].map(label2id)
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
    df = pd.read_csv("data/analiza_poetycka.csv")
    unique_forms = sorted(df["type"].unique())
    label2id = {form: i for i, form in enumerate(unique_forms)}
    id2label = {i: form for form, i in label2id.items()}

    # stratify bo były puste w train
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['type'])
    eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=df['type'])

    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

    train_dataset = prepare_dataset(train_df, tokenizer)
    eval_dataset = prepare_dataset(eval_df, tokenizer)
    test_dataset = prepare_dataset(test_df, tokenizer)

    classifier = PoemFormClassifier(
        model_name="allegro/herbert-base-cased",
        id2label=id2label,
        label2id=label2id,
        num_labels=len(id2label)
    )

    results = classifier.finetune(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        output_dir="./poem_form_model",
        optimize_hyperparams=True,
        n_trials=20
    )

    print("----- results:", results)
