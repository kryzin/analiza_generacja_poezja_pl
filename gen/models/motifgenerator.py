import os
import ast
import pandas as pd
import torch
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score

MODEL_GPT = "gpt-4o-mini"
CSV_PATH = "wiersze.csv"

motywy = [
    'miłość', 'śmierć', 'czas', 'natura', 'pamięć', 'samotność', 'przemijanie',
    'tęsknota', 'piękno', 'młodość', 'starość', 'dom', 'ojczyzna', 'podróż',
    'wolność', 'wojna', 'sacrum', 'żal', 'nadzieja', 'tożsamość', 'rozkosz'
]

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI(api_key=api_key)


def answer_with_gpt(question: str) -> str:
    stream = openai.chat.completions.create(
        model=MODEL_GPT,
        messages=[
            {"role": "system", "content": "You are a motif classifier. Respond only with 3 items from the list."},
            {"role": "user", "content": question}
        ],
        stream=True
    )

    response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        response += content
    return response


df = pd.read_csv(CSV_PATH).sample(n=400, random_state=42)
df["motifs_list"] = df["motifs_list"].apply(ast.literal_eval)
df["motywy_set"] = df["motifs_list"].apply(lambda lst: set(m.strip().lower() for m in lst if m.strip().lower() in motywy))

predicted = []
true = []
predicted_labels = []
decoded_outputs = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    tekst = row["content"]
    true_motywy = row["motywy_set"]

    prompt = (
        f"Z listy poniżej wybierz dokładnie trzy motywy, które najlepiej pasują do podanego tekstu. "
        f"Odpowiedz tylko trzema motywami oddzielonymi przecinkiem, bez komentarzy.\n\n"
        f"MOTYWY: {', '.join(motywy)}\n\n"
        f"TEKST:\n{tekst}\n\n"
        f"ODPOWIEDŹ:"
    )

    try:
        decoded = answer_with_gpt(prompt)
    except Exception as e:
        decoded = f"ERROR: {e}"

    decoded_outputs.append(decoded)

    pred_motywy = [m.strip().lower() for m in decoded.split(",") if m.strip().lower() in motywy]
    pred_motywy = list(dict.fromkeys(pred_motywy))[:3]

    predicted.append([1 if m in pred_motywy else 0 for m in motywy])
    true.append([1 if m in true_motywy else 0 for m in motywy])
    predicted_labels.append("; ".join(pred_motywy))

df["odpowiedz_modelu"] = decoded_outputs

print("\n=== METRYKI OGÓLNE ===")
print("Precision:", precision_score(true, predicted, average="micro", zero_division=0))
print("Recall:   ", recall_score(true, predicted, average="micro"))
print("F1-score: ", f1_score(true, predicted, average="micro"))

df.to_csv("motywy_gpt_output.csv", index=False)
print("\nZapisano do: motywy_gpt_output.csv")
