# Analiza i Generacja Poezji w Języku Polskim

Projekt umożliwia analizę oraz generowanie poezji w języku polskim przy użyciu nowoczesnych modeli NLP. Wykorzystuje klasyfikatory do rozpoznawania formy i układu rymów oraz generatory wspierające styl, rym i metrum.

## 📁 Struktura projektu

```
analysis/
  └── main.py              # Klasyfikator formy/rymu
  └── models/              # Zapisane modele i konfiguracje

gen/
  └── defaults.json        # Domyślne parametry generatora
  └── models/              # Modele generujące poezję

data/
  └── wiersze.csv          # Dane treningowe
  └── preprocessing.py     # Czyszczenie, lematyzacja, analiza rymów
  └── morfeusz.py          # Interfejs do Morfeusza2
  └── analiza.py/.ipynb    # Notebook z eksploracją danych
```

## ⚙️ Wymagania

Python **3.12**  
Wszystkie zależności zainstalujesz komendą:

```bash
pip install -r requirements.txt
```

Główne biblioteki:
- `transformers`, `torch`, `spacy`, `optuna`
- `epitran`, `panphon`, `sentence-transformers`
- `scikit-learn`, `seaborn`, `matplotlib`, `nltk`, `yake`, `wordcloud`

## 🚀 Szybki start

1. **Klonuj repozytorium**:

```bash
git clone https://github.com/kryzin/analiza_generacja_poezja_pl.git
cd analiza_generacja_poezja_pl
```

2. **Zainstaluj wymagania**:

```bash
pip install -r requirements.txt
```

3. **Pobierz model spaCy**:

```bash
python -m spacy download pl_core_news_sm
```

4. **Wstępne przetwarzanie danych**:

```bash
python data/preprocessing.py
```

5. **Uruchom klasyfikator** (np. formy wiersza):

```bash
python analysis/main.py
```

## 🧠 Notatniki i eksperymenty

Pliki `analiza.ipynb` i `baza.ipynb` zawierają eksplorację danych oraz przykładowe testy modelowe. Znajdziesz je w folderze `data/`.

## 🧪 Morfeusz2

Lematizacja i analiza morfologiczna realizowana jest z pomocą pliku `morfeusz.py`. Wymaga zainstalowanego Morfeusza2 oraz jego biblioteki Pythonowej (`morfeusz2`).
