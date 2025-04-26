import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from models.classifier import PoemClassifier


def create_label_mappings(df, label_column):
    unique_labels = sorted(df[label_column].dropna().unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label

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
    df = pd.read_csv("data/wiersze_rhyme_col2.csv")
    label2id, id2label = create_label_mappings(df, label_column="rhyme")

    # train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    # eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

    # train_dataset = prepare_dataset(train_df, tokenizer)
    # eval_dataset = prepare_dataset(eval_df, tokenizer)
    # test_dataset = prepare_dataset(test_df, tokenizer)

    # classifier = PoemFormClassifier(
    #     model_name="allegro/herbert-base-cased",
    #     id2label=id2label,
    #     label2id=label2id,
    #     num_labels=len(id2label)
    # )

    # results = classifier.finetune(
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     test_dataset=test_dataset,
    #     output_dir="C:/Users/karol/GitHub/mgr/results/best_model_content",
    #     optimize_hyperparams=True,
    #     n_trials=10
    # )

    # train_dataset_lem = prepare_dataset(train_df, tokenizer, text_column="content_lemma")
    # eval_dataset_lem = prepare_dataset(eval_df, tokenizer, text_column="content_lemma")
    # test_dataset_lem = prepare_dataset(test_df, tokenizer, text_column="content_lemma")

    # classifier_lem = PoemFormClassifier(
    #     model_name="allegro/herbert-base-cased",
    #     id2label=id2label,
    #     label2id=label2id,
    #     num_labels=len(id2label)
    # )

    # results_lem = classifier_lem.finetune(
    #     train_dataset=train_dataset_lem,
    #     eval_dataset=eval_dataset_lem,
    #     test_dataset=test_dataset_lem,
    #     output_dir="C:/Users/karol/GitHub/mgr/results/best_model_lem",
    #     optimize_hyperparams=True,
    #     n_trials=10
    # )

    # print("----- results for regular text:", results)
    # print("----- results for lemma text:", results_lem)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["rhyme"])
    eval_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["rhyme"])

    train_dataset = prepare_dataset(train_df, tokenizer, text_column="rhyme_input", label_column="rhyme")
    eval_dataset = prepare_dataset(eval_df, tokenizer, text_column="rhyme_input", label_column="rhyme")
    test_dataset = prepare_dataset(test_df, tokenizer, text_column="rhyme_input", label_column="rhyme")

    classifier = PoemClassifier(
        model_name="allegro/herbert-base-cased",
        id2label=id2label,
        label2id=label2id,
        num_labels=len(id2label)
    )

    results = classifier.finetune(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        output_dir="C:/Users/karol/GitHub/mgr/results/best_rhyme_model",
        optimize_hyperparams=True,
        n_trials=10
    )

    print("----- Wyniki klasyfikatora rymu:", results)
