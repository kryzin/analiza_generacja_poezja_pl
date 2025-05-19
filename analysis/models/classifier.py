import torch
import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments, EarlyStoppingCallback)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import optuna  # paczka do optymalizacji hiperparametrów
from optuna.pruners import MedianPruner


class PoemClassifier:
    def __init__(self, model_name="allegro/herbert-base-cased", num_labels=None,
                 id2label=None, label2id=None):
        self.model_name = model_name
        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = num_labels or (len(id2label) if id2label else None)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None  # w init modelu

    def eval_scores(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted', zero_division=0),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0)
        }

    def _initialize_model(self, trial=None):
        dropout = 0.1
        attention_dropout = 0.1

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=attention_dropout
        )
        # cross entropy vs bce logits loss (dla multi - dlatego precyzyjniej)
        model.config.problem_type = 'single_label_classification'

        return model

    def optimize_hyperparameters(self, train_dataset, eval_dataset, n_trials=100):
        def objective(trial):
            # <praca> opisać dobór przedziałów:
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
            weight_decay = trial.suggest_float("weight_decay", 0.01, 0.3)
            batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
            warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.3)

            model = self._initialize_model(trial)

            # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
            training_args = TrainingArguments(
                output_dir=f"./results/hparam_search/trial_{trial.number}",
                num_train_epochs=3,
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
            eval_results = trainer.evaluate()

            print(f"\n--- Trial {trial.number} ---")
            print(f"learning_rate: {learning_rate}")
            print(f"weight_decay: {weight_decay}")
            print(f"batch_size: {batch_size}")
            print(f"warmup_ratio: {warmup_ratio}")
            print("Evaluation metrics:")
            for k, v in eval_results.items():
                print(f"{k}: {v:.4f}")
            print("------\n")

            preds = trainer.predict(eval_dataset)
            labels = preds.label_ids
            predictions = np.argmax(preds.predictions, axis=1)
            f1_micro = f1_score(labels, predictions, average='micro')
            print(f"f1_micro: {f1_micro:.4f}")

            torch.cuda.empty_cache()

            return eval_results["eval_f1"]

        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html
        study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

        study.optimize(objective, n_trials=n_trials)
        print(f"----- best parameters: {study.best_params}")

        return study.best_params

    # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    def _compute_class_weights(self, dataset):
        labels = [item['labels'].item() for item in dataset]
        classes = np.array(sorted(set(labels)))

        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )

        return weights

    def finetune(self, train_dataset, eval_dataset, test_dataset=None,
                 output_dir="./poem_form_classifier", optimize_hyperparams=False,
                 n_trials=10, batch_size=16, learning_rate=2e-5,
                 num_epochs=4):

        if optimize_hyperparams:
            print("----- hyperparameter optimization...")
            hyperparams = self.optimize_hyperparameters(
                train_dataset,
                eval_dataset,
                n_trials=n_trials
            )

            learning_rate = hyperparams.get("learning_rate", learning_rate)
            batch_size = hyperparams.get("batch_size", batch_size)
            weight_decay = hyperparams.get("weight_decay", 0.01)
            warmup_ratio = hyperparams.get("warmup_ratio", 0.1)
        else:
            weight_decay = 0.01
            warmup_ratio = 0.1

        class_weights = self._compute_class_weights(train_dataset)

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
            save_total_limit=2,
            fp16=True,
            gradient_accumulation_steps=1,
        )

        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

        trainer = ClassWeightTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.eval_scores,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            class_weights=class_weights_tensor
        )

        print("----- training...")
        trainer.train()

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # mapping etykirt
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

    def predict(self, text_inputs, batch_size=32):
        if self.model is None:
            raise ValueError("404.")

        # już tokenized
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            compute_metrics=self.eval_scores
        )

        predictions = trainer.predict(text_inputs)
        probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)
        confidence, predicted_classes = torch.max(probs, dim=1)
        predicted_labels = predicted_classes.numpy().tolist()
        confidence_scores = confidence.numpy().tolist()
        predicted_names = [self.id2label[label] for label in predicted_labels]

        return {
            "predictions": predicted_labels,
            "confidences": confidence_scores,
            "labels": predicted_names
        }

    def load_model(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self.num_labels = len(self.id2label)

        print(f"loaded model with {self.num_labels} labels")
        return self


class ClassWeightTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None and self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
