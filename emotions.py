import pandas as pd
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
data = pd.read_csv('nonnull_poems.csv')
data['content'] = data['content'].fillna('').astype(str)

print(data.head(2))
data['emotion_intensity'] = data['content'].apply(lambda x: sia.polarity_scores(x)['compound'])

data.to_csv('emotions_poems.csv', index=False)


