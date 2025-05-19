import pandas as pd
import ast
import spacy
from sklearn.metrics import precision_score, recall_score, f1_score
from yake import KeywordExtractor
from tqdm import tqdm


nlp = spacy.load("pl_core_news_sm")


def tokenize_keywords(keyword_str):
    try:
        parsed = ast.literal_eval(keyword_str)
        keywords = [kw.strip().lower() for kw in parsed if isinstance(kw, str)]

        noun_keywords = []
        for phrase in keywords:
            doc = nlp(phrase)
            if all(token.pos_ == "NOUN" for token in doc):
                noun_keywords.append(phrase)
        return noun_keywords

    except Exception as e:
        print("Błąd parsowania słów kluczowych:", e)
        return []


def filter_nouns_with_spacy(phrases):
    noun_keywords = []
    for phrase in phrases:
        doc = nlp(phrase)
        if all(token.pos_ == "NOUN" for token in doc):
            noun_keywords.append(phrase)
    return noun_keywords


def evaluate_keywords(predicted, ground_truth):
    pred_set = set(predicted)
    gt_set = set(ground_truth)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def main(csv_path, language="pl", max_ngram_size=1, top_k=10):
    df = pd.read_csv(csv_path).head(3)

    kw_extractor = KeywordExtractor(lan=language, n=max_ngram_size, top=top_k)

    all_prec, all_rec, all_f1 = [], [], []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Przetwarzanie"):
        text = row['content_lemma']
        manual = tokenize_keywords(row['keywords_list'])

        keywords = kw_extractor.extract_keywords(text)
        predicted_raw = [kw.lower() for kw, _ in keywords]

        predicted = filter_nouns_with_spacy(predicted_raw)

        prec, rec, f1 = evaluate_keywords(predicted, manual)
        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        # print(f"🔹 Przykład {i+1}")
        # print(f"Tekst: {text[:100]}...")
        # print(f"Manualne: {manual}")
        # print(f"Predykcja przed filtrowaniem: {predicted_raw}")
        # print(f"Predykcja (rzeczowniki): {predicted}")
        # print(f"Precyzja: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}\n")

    print("=== ŚREDNIE METRYKI ===")
    print(f"Śr. precyzja: {sum(all_prec)/len(all_prec):.2f}")
    print(f"Śr. recall: {sum(all_rec)/len(all_rec):.2f}")
    print(f"Śr. F1: {sum(all_f1)/len(all_f1):.2f}")


if __name__ == "__main__":
    main("./analysis/data/wiersze.csv")
