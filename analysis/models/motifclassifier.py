import torch
import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments, EarlyStoppingCallback)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import optuna
from optuna.pruners import MedianPruner
import pandas as pd
import ast
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset


class MotifClassifier:
    def __init__(self, model_name="clarin-pl/roberta-polish-kgr10", num_labels=None,
                 id2label=None, label2id=None):
        self.model_name = model_name
        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = num_labels or (len(id2label) if id2label else None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def eval_scores(self, eval_pred):
        predictions, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(predictions))
        preds = (probs >= 0.3).int().numpy()
        labels = labels.astype(int)

        return {
            'f1': f1_score(labels, preds, average='micro', zero_division=0),
            'precision': precision_score(labels, preds, average='micro', zero_division=0),
            'recall': recall_score(labels, preds, average='micro', zero_division=0)
        }

    def _initialize_model(self, trial=None):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        model.config.problem_type = 'multi_label_classification'
        return model

    def optimize_hyperparameters(self, train_dataset, eval_dataset, n_trials=10):
        def objective(trial):
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
            weight_decay = trial.suggest_float("weight_decay", 0.01, 0.3)
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
            warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.3)

            model = self._initialize_model(trial)
            training_args = TrainingArguments(
                output_dir=f"./results/hparam_search/trial_{trial.number}",
                num_train_epochs=5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=32,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                report_to="none",
                save_total_limit=1,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self.eval_scores,
                tokenizer=self.tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )

            trainer.train()
            return trainer.evaluate()["eval_f1"]

        study = optuna.create_study(direction="maximize", pruner=MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        print(f"----- best parameters: {study.best_params}")
        return study.best_params

    def finetune(self, train_dataset, eval_dataset, test_dataset=None,
                 output_dir="./motif_model", optimize_hyperparams=False,
                 n_trials=10, batch_size=16, learning_rate=2e-5, num_epochs=4):

        if optimize_hyperparams:
            print("----- hyperparameter optimization...")
            best = self.optimize_hyperparameters(train_dataset, eval_dataset, n_trials)
            learning_rate = best['learning_rate']
            batch_size = best['batch_size']
            weight_decay = best['weight_decay']
            warmup_ratio = best['warmup_ratio']
        else:
            weight_decay = 0.01
            warmup_ratio = 0.1

        # oblicz pos_weight dla strat wieloetykietowych
        # labels_np = np.array([ex['labels'].numpy() for ex in train_dataset])
        # pos_weights = torch.tensor((labels_np.shape[0] / (labels_np.sum(axis=0) + 1e-8)),
        #                            dtype=torch.float)

        self.model = self._initialize_model()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{output_dir}/logs",
            logging_strategy="epoch",
            report_to="none",
            save_total_limit=2
        )

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                labels = labels.to(logits.device)
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss

        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.eval_scores,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print("----- training...")
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        with open(f"{output_dir}/label_mappings.txt", "w") as f:
            f.write(f"id2label: {self.id2label}\n")
            f.write(f"label2id: {self.label2id}\n")

        print("----- evaluation - validation set...")
        eval_results = trainer.evaluate()

        if test_dataset:
            print("----- test set...")
            test_results = trainer.evaluate(test_dataset)
            for key, value in test_results.items():
                eval_results[f"test_{key.replace('eval_', '')}"] = value

        return eval_results


def optimal_threshold(y_true, y_prob):
    best_thresh, best_f1 = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), average='micro')
        if f1 > best_f1:
            best_thresh, best_f1 = t, f1
    return best_thresh, best_f1


if __name__ == "__main__":
    class SimpleDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx]).float()
            return item

        def __len__(self):
            return len(self.labels)

    stop_words = set([
        "a", "aby", "ach", "acz", "aczkolwiek", "aj", "albo", "ale",
        "ależ", "ani", "aż", "bardziej", "bardzo", "bez", "bo", "bowiem",
        "by", "byli", "bynajmniej", "być", "był", "była", "było", "były",
        "będzie", "będą", "cali", "cała", "cały", "ci", "cię", "ciebie", "co",
        "cokolwiek", "coś", "czasami", "czasem", "czemu", "czy", "czyli",
        "często", "daleko", "dla", "dlaczego", "dlatego", "do", "dobrze",
        "dokąd", "dość", "dr", "dużo", "dwa", "dwaj", "dwie", "dwoje",
        "dzisiaj", "dziś", "gdy", "gdyby", "gdyż", "gdzie", "gdziekolwiek",
        "gdzieś", "go", "i", "ich", "ile", "im", "inna", "inne", "inny",
        "innych", "iz", "ja", "jak", "jakaś", "jakby", "jaki", "jakichś",
        "jakie", "jakiś", "jako", "jakoś", "je", "jeden", "jedna", "jedno",
        "jednak", "jednakże", "jego", "jej", "jemu", "jest", "jestem",
        "jeszcze", "jeśli", "jeżeli", "już", "ją", "kiedy", "kilka", "kto",
        "ktokolwiek", "ktoś", "która", "które", "którego", "której",
        "który", "których", "którym", "którzy", "ku", "lat", "lecz", "lub",
        "ma", "mają", "mam", "mi", "mimo", "między", "mną", "mnie", "mogą",
        "moi", "moim", "moje", "może", "możliwe", "można", "mój", "mu",
        "musi", "my", "na", "nad", "nam", "nami", "nas", "nasi", "nasz",
        "nasza", "nasze", "naszego", "naście", "natomiast", "natychmiast",
        "nawet", "nią", "nic", "nich", "nie", "niech", "niego", "niej",
        "niemu", "nigdy", "nim", "nimi", "niż", "no", "o", "obok", "od",
        "około", "on", "ona", "one", "oni", "ono", "oraz", "oto", "owszem",
        "pan", "pana", "pani", "po", "pod", "ponad", "ponieważ", "powinien",
        "powinna", "powinni", "powinno", "poza", "prawie", "przecież",
        "przed", "przede", "przez", "przy", "roku", "również", "sam",
        "sama", "się", "skąd", "sobie", "sposób", "swoje", "ta", "tak",
        "taka", "taki", "takich", "takie", "także", "tam", "te", "tego",
        "tej", "temu", "ten", "teraz", "też", "to", "tobą", "tobie",
        "toteż", "trzeba", "tu", "tutaj", "twoi", "twoim", "twoja", "twoje",
        "twym", "twój", "ty", "tych", "tylko", "tym", "u", "w", "wam",
        "wami", "was", "wasi", "wasz", "wasza", "wasze", "we", "według",
        "wiele", "wielu", "więc", "więcej", "wszyscy", "wszystkich",
        "wszystkie", "wszystkim", "wszystko", "wtedy", "wy", "właśnie",
        "z", "za", "zapewne", "ze", "znowu", "znów", "został", "że",
        "żaden", "żadna", "żadne", "żadnych", "żeby"
    ])

    csv_path = "./analysis/data/wiersze.csv"
    if not os.path.exists(csv_path):
        print(f"----- file not found: {csv_path}")
        exit()

    print("----- loading data...")
    df = pd.read_csv(csv_path)
    df["labels"] = df["motifs_list"].apply(ast.literal_eval)

    def clean(text):
        return " ".join([w for w in text.split() if w not in stop_words])

    # df["clean_text"] = df["content_lemma"].apply(clean)
    # df["clean_text"] = df["content"]
    # df["clean_text"] = df["content_lemma"]
    df["clean_text"] = df["keywords_list"].apply(ast.literal_eval).apply(lambda lst: " ".join(lst))

    all_labels = sorted(set(label for sublist in df["labels"] for label in sublist))
    mlb = MultiLabelBinarizer(classes=all_labels)
    Y = mlb.fit_transform(df["labels"])

    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    classifier = MotifClassifier(
        # model_name="allegro/herbert-large-cased",
        model_name="allegro/herbert-base-cased",
        # model_name="FacebookAI/xlm-roberta-base",
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], Y, test_size=0.2, random_state=42
    )

    train_dataset = SimpleDataset(list(X_train), y_train, classifier.tokenizer)
    test_dataset = SimpleDataset(list(X_test), y_test, classifier.tokenizer)

    print("----- training with Optuna...")
    results = classifier.finetune(
        train_dataset,
        test_dataset,
        output_dir="./motifs_herbert_base_keywords_classifier",
        # output_dir="./motifs_roberta_classifier",
        optimize_hyperparams=True,
        n_trials=5,
        num_epochs=3
    )

    print("----- evaluation finished")
    print(results)

    print("----- threshold optimization...")
    trainer = Trainer(model=classifier.model, tokenizer=classifier.tokenizer)
    pred = trainer.predict(test_dataset)
    probs = torch.sigmoid(torch.tensor(pred.predictions)).numpy()
    true = pred.label_ids
    best_thresh, best_f1 = optimal_threshold(true, probs)
    print(f"Optimal threshold: {best_thresh:.2f}, F1: {best_f1:.4f}")
    print(results)
