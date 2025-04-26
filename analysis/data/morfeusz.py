import pandas as pd
import morfeusz2
import re
from tqdm import tqdm


class MorfWrapper:
    def __init__(self):
        self.morf = morfeusz2.Morfeusz()

    @staticmethod
    def clean_lemma(lemma):
        return re.split(r'[:;=]', lemma)[0]

    def lemmatize_line(self, text):
        words = re.findall(r'\b\w+\b', str(text).lower())
        lemmas = []
        for word in words:
            analyses = self.morf.analyse(word)
            if analyses:
                raw_lemma = analyses[0][2][1]
                cleaned_lemma = self.clean_lemma(raw_lemma)
                lemmas.append(cleaned_lemma)
            else:
                lemmas.append(word)
        return ' '.join(lemmas)


if __name__ == "__main__":
    df = pd.read_csv("./analysis/data/analiza_poetycka.csv")
    morf = MorfWrapper()

    tqdm.pandas()
    df["content_lemma"] = df["content"].progress_apply(lambda x: morf.lemmatize_line(str(x)))

    df.to_csv("./analysis/data/wiersze_lemma.csv", index=False)
