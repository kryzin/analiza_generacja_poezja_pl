import pandas as pd
import ast
from keybert import KeyBERT
from sklearn.metrics import precision_score, recall_score, f1_score
import spacy
from tqdm import tqdm

nlp = spacy.load('pl_core_news_sm')

kw_model = KeyBERT(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
with open("stop_words_polish.txt", "r", encoding="utf-8") as f:
    polish_stopwords = [line.strip().lower() for line in f if line.strip()]

TOP_N = 7
text_column = "content_lemma"
manual_column = "keywords_list"

df = pd.read_csv("wiersze.csv")
df['keywords_list'] = df["keywords_list"].apply(ast.literal_eval)


def extract_keywords(text):
    if pd.isna(text) or not isinstance(text, str):
        return []

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words=polish_stopwords,
        use_maxsum=True,
        nr_candidates=20,
        top_n=TOP_N
    )

    noun_keywords = []
    for kw, score in keywords:
        doc = nlp(kw)
        if any(tok.pos_ == "NOUN" for tok in doc):
            noun_keywords.append(kw.lower())

    return noun_keywords


results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Przetwarzanie wierszy"):
    text = row[text_column]
    manual_set = set(kw.lower() for kw in row.get(manual_column, []) if kw)

    predicted_list = extract_keywords(text)
    predicted_set = set(predicted_list)

    intersection = predicted_set & manual_set

    true_labels = [kw in manual_set for kw in predicted_list]
    pred_labels = [True] * len(predicted_list)

    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    results.append({
        "tekst": text,
        "manual_keywords": ", ".join(manual_set),
        "predicted_keywords": ", ".join(predicted_list),
        "intersection": ", ".join(intersection),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

results_df = pd.DataFrame(results)
results_df.to_csv("keywords_test1.csv", index=False, encoding="utf-8")

print("Średnie metryki dla całego zbioru:")
print("Precision:", results_df["precision"].mean())
print("Recall:", results_df["recall"].mean())
print("F1-score:", results_df["f1_score"].mean())
