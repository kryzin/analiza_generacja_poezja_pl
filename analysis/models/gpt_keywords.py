import pandas as pd
import ast
import openai
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_GPT = "gpt-4o-mini"
TEXT_COLUMN = "content_lemma"
MANUAL_COLUMN = "keywords_list"
TOP_N = 7
FILTER_NOUNS = True

if FILTER_NOUNS:
    nlp = spacy.load("pl_core_news_sm")

df = pd.read_csv("./analysis/data/wiersze.csv").sample(400)
df[MANUAL_COLUMN] = df[MANUAL_COLUMN].apply(ast.literal_eval)


def get_keywords_from_gpt(text):
    if pd.isna(text) or not isinstance(text, str):
        return []

    prompt = (
        f"Extract the {TOP_N} most important keywords from the following Polish poem.\n\n"
        f"TEXT:\n{text}\n\n"
        "Return a numbered list of keywords only, one per line."
    )

    try:
        response = openai.chat.completions.create(
            model=MODEL_GPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content

        keywords = [
            line.split(". ", 1)[-1].strip().lower()
            for line in content.strip().splitlines()
            if ". " in line
        ]

        if FILTER_NOUNS:
            noun_keywords = []
            for kw in keywords:
                doc = nlp(kw)
                if any(tok.pos_ == "NOUN" for tok in doc):
                    noun_keywords.append(kw)
            return noun_keywords

        return keywords

    except Exception as e:
        print(f"Błąd zapytania do GPT: {e}")
        return []


results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Przetwarzanie wierszy"):
    text = row[TEXT_COLUMN]
    manual_keywords = set(kw.lower() for kw in row[MANUAL_COLUMN] if kw)

    predicted_list = get_keywords_from_gpt(text)
    predicted_set = set(predicted_list)

    intersection = predicted_set & manual_keywords

    true_labels = [kw in manual_keywords for kw in predicted_list]
    pred_labels = [True] * len(predicted_list)

    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    results.append({
        "tekst": text,
        "manual_keywords": ", ".join(manual_keywords),
        "predicted_keywords": ", ".join(predicted_list),
        "intersection": ", ".join(intersection),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

results_df = pd.DataFrame(results)
results_df.to_csv("gpt_keywords_evaluation.csv", index=False, encoding="utf-8")

print("Średnie metryki dla całego zbioru:")
print("Precision:", round(results_df["precision"].mean(), 3))
print("Recall:", round(results_df["recall"].mean(), 3))
print("F1-score:", round(results_df["f1_score"].mean(), 3))
