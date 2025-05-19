import pandas as pd
import ast
import requests
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

MODEL_LLAMA = 'llama3.2'
TEXT_COLUMN = "content_lemma"
MANUAL_COLUMN = "keywords_list"
TOP_N = 7
FILTER_NOUNS = False

if FILTER_NOUNS:
    nlp = spacy.load("pl_core_news_sm")

df = pd.read_csv("./analysis/data/wiersze.csv").sample(n=400, random_state=42)
df[MANUAL_COLUMN] = df[MANUAL_COLUMN].apply(ast.literal_eval)


def get_keywords_from_llama(text):
    if pd.isna(text) or not isinstance(text, str):
        return []

    prompt = f"Choose keywords from this text, respond with a\
              list of {TOP_N} most important keywords.\n\nTEXT: {text}"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_LLAMA,
            "prompt": prompt,
            "stream": False
        }
    )
    data = response.json()
    if "response" not in data:
        return []

    raw = data["response"]
    keywords = [
        line.split(". ", 1)[-1].strip().lower()
        for line in raw.strip().splitlines()
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


results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Przetwarzanie wierszy"):
    text = row[TEXT_COLUMN]
    manual_keywords = set(kw.lower() for kw in row[MANUAL_COLUMN] if kw)

    predicted_list = get_keywords_from_llama(text)
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
results_df.to_csv("llama_keywords_evaluation.csv", index=False, encoding="utf-8")

print("Średnie metryki dla całego zbioru:")
print("Precision:", round(results_df["precision"].mean(), 3))
print("Recall:", round(results_df["recall"].mean(), 3))
print("F1-score:", round(results_df["f1_score"].mean(), 3))
