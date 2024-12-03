import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


nltk.download('punkt')
nltk.download('stopwords')


data = pd.read_csv('gpt_analyzed_poems.csv')
print("Podgląd danych:")
data = data.rename(columns={
    "Title": "title",
    "Author": "author",
    "Content": "content",
    "Motywy": "motifs",
    "Forma": "form",
    "Typ rymu": "rhyme",
    "Słowa kluczowe": "keywords"
})
print(data.head(5))

# PODSTAWY
print("\nBrakujące wartości w kolumnach:")

data = data.replace(r'^\s*$', float('NaN'), regex=True)
data = data.dropna(subset=['Content', 'Słowa kluczowe'])

print(data.isnull().sum())
print("\nInformacje o danych:")
print(data.info())

# AUTORZY
unique_authors = data['author'].nunique()
print(f"\nLiczba unikalnych autorów: {unique_authors}")

top_authors = data['author'].value_counts().head(10)
print("\nNajczęściej występujący autorzy:")
print(top_authors)

# FORMY
unique_forms = data['form'].nunique()
print(f"\nLiczba unikalnych form literackich: {unique_forms}")

form_counts = data['form'].value_counts()
print("\nRozkład form literackich:")
print(form_counts)

# MOTYWY
data['motywy_list'] = data['motifs'].fillna('').apply(lambda x: [motif.strip() for motif in x.split(',')])
all_motifs = [motif for sublist in data['motywy_list'] for motif in sublist if motif]
unique_motifs = set(all_motifs)
print(f"\nLiczba unikalnych motywów: {len(unique_motifs)}")

motif_counts = Counter(all_motifs)
top_motifs = motif_counts.most_common(10)
print("\nNajczęściej występujące motywy:")
for motif, count in top_motifs:
    print(f"{motif}: {count}")

# SŁOWA KLUCZOWE
data['keywords_list'] = data['keywords'].fillna('').apply(lambda x: [kw.strip() for kw in x.split(',')])
all_keywords = [kw for sublist in data['keywords_list'] for kw in sublist if kw]
keyword_counts = Counter(all_keywords)
top_keywords = keyword_counts.most_common(10)
print("\nNajczęściej występujące słowa kluczowe:")
for kw, count in top_keywords:
    print(f"{kw}: {count}")

# TEKSTY
def word_count(text):
    tokens = word_tokenize(text)
    return len(tokens)

data['word_count'] = data['content'].apply(word_count)
print("\nStatystyki opisowe długości tekstu:")
print(data['word_count'].describe())

plt.figure(figsize=(10,6))
sns.histplot(data['word_count'], bins=30, kde=True)
plt.title('Rozkład liczby słów w tekstach')
plt.xlabel('Liczba słów')
plt.ylabel('Liczba tekstów')
plt.show()

# ZALEŻNOŚCI
motifs_matrix = data['motywy_list'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
motifs_matrix['forma'] = data['form']
motifs_by_form = motifs_matrix.groupby('forma').sum()
print("\nMotywy w zależności od formy literackiej:")
print(motifs_by_form)

motifs_by_form.plot(kind='bar', stacked=True, figsize=(12,8))
plt.title('Motywy w zależności od formy literackiej')
plt.xlabel('Forma literacka')
plt.ylabel('Liczba wystąpień motywu')
plt.legend(title='Motywy', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# RYMY
rhyme_counts = data['rhyme'].value_counts()
print("\nRozkład typów rymów:")
print(rhyme_counts)

plt.figure(figsize=(8,8))
rhyme_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Procentowy udział typów rymów')
plt.ylabel('')
plt.show()

# SŁOWA + KLUCZOWE
all_text = ' '.join(data['content'].dropna())
translator = str.maketrans('', '', string.punctuation)
all_text_clean = all_text.translate(translator)
tokens = word_tokenize(all_text_clean.lower())
stop_words = set(stopwords.words('polish'))
tokens_clean = [word for word in tokens if word not in stop_words and word.isalpha()]

word_freq = Counter(tokens_clean)
most_common_words = word_freq.most_common(20)
print("\nNajczęściej występujące słowa w korpusie:")
for word, freq in most_common_words:
    print(f"{word}: {freq}")

words, frequencies = zip(*most_common_words)
plt.figure(figsize=(12,6))
sns.barplot(x=list(words), y=list(frequencies))
plt.title('Najczęściej występujące słowa w korpusie')
plt.xlabel('Słowa')
plt.ylabel('Częstotliwość')
plt.xticks(rotation=45)
plt.show()

data['motifs_count'] = data['motywy_list'].apply(len)
plt.figure(figsize=(10,6))
sns.scatterplot(x='word_count', y='motifs_count', data=data)
plt.title('Zależność między długością tekstu a liczbą motywów')
plt.xlabel('Liczba słów w tekście')
plt.ylabel('Liczba motywów')
plt.show()

corr_coeff = data[['word_count', 'motifs_count']].corr().iloc[0,1]
print(f"\nWspółczynnik korelacji Pearsona między długością tekstu a liczbą motywów: {corr_coeff:.2f}")

from itertools import combinations

motif_pairs = []
for motifs in data['motywy_list']:
    for combo in combinations(sorted(set(motifs)), 2):
        motif_pairs.append(combo)

motif_pair_counts = Counter(motif_pairs)
most_common_pairs = motif_pair_counts.most_common(10)
print("\nNajczęściej współwystępujące pary motywów:")
for pair, count in most_common_pairs:
    print(f"{pair}: {count}")

form_by_author = data.groupby(['author', 'forma']).size().unstack(fill_value=0)
print("\nCzęstotliwość form literackich według autorów:")
print(form_by_author)

form_by_author.plot(kind='bar', stacked=True, figsize=(12,8))
plt.title('Formy literackie w twórczości poszczególnych autorów')
plt.xlabel('Autor')
plt.ylabel('Liczba utworów')
plt.legend(title='Formy', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

data.to_csv('analiza_poetycka.csv', index=False)
print("\nAnaliza zakończona. Wyniki zapisano w pliku 'analiza_poetycka.csv'.")
