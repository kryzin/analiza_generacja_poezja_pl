import pandas as pd
import re
import epitran  # https://pypi.org/project/epitran/


# --------------------------------------NORMALIZACJA-----------------------------
def normalize_text(self, text):
    if pd.isna(text):
        return text

    replacements = {
        '"': '„',
        '"': '"',
        '«': '„',
        '»': '"',
        '"': '„',
        '"': '"',
        '---': '—',
        '--': '–',
        '—': '—',
        '–': '–',
        '...': '…',
        '. . .': '…',
        '….': '….',
    }

    text = str(text)

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\s+([.,!?:;])', r'\1', text)
    text = re.sub(r'([^\s])—', r'\1 —', text)
    text = re.sub(r'—([^\s])', r'— \1', text)

    def replace_number(match):
        number_str = match.group()
        if re.match(r'^[12]\d{3}$', number_str):
            return number_str

        number = float(number_str) if '.' in number_str else int(number_str)
        return number_to_polish_words(number)

    text = re.sub(r'\b\d+\.?\d*\b', replace_number, text)

    def replace_year(match):
        year = int(match.group())
        if 1000 <= year <= 2099:
            return year_to_polish_words(year)
        return match.group()

    text = re.sub(r'\b(1\d{3}|20[0-9]{2})\b', replace_year, text)
    return text


def normalize_csv(self, input_path, output_path, content_column='Content'):
    df = pd.read_csv(input_path)
    df[content_column] = df[content_column].apply(normalize_text)
    df.to_csv(output_path, index=False, encoding='utf-8')

    return f"Normalization completed: {output_path}"


# --------------------------------------------------------LICZBY NA SŁOWA--------------
def number_to_polish_words(number):
    if not isinstance(number, (int, float)):
        return number

    if number == 0:
        return "zero"

    units = ["", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem",
             "osiem", "dziewięć"]
    teens = ["dziesięć", "jedenaście", "dwanaście", "trzynaście", "czternaście",
             "piętnaście", "szesnaście", "siedemnaście", "osiemnaście", "dziewiętnaście"]
    tens = ["", "", "dwadzieścia", "trzydzieści", "czterdzieści", "pięćdziesiąt",
            "sześćdziesiąt", "siedemdziesiąt", "osiemdziesiąt", "dziewięćdziesiąt"]
    hundreds = ["", "sto", "dwieście", "trzysta", "czterysta", "pięćset",
                "sześćset", "siedemset", "osiemset", "dziewięćset"]

    def convert_group(n):
        if n == 0:
            return ""
        elif n < 10:
            return units[n]
        elif n < 20:
            return teens[n-10]
        elif n < 100:
            return tens[n//10] + (" " + units[n % 10] if n % 10 != 0 else "")
        else:
            return hundreds[n//100] + (" " + convert_group(n % 100) if n % 100 != 0 else "")

    if isinstance(number, float):
        whole = int(number)
        frac = int(round((number - whole) * 100))
        if frac == 0:
            return convert_group(whole)
        else:
            return f"{convert_group(whole)} i {convert_group(frac)}/100"

    return convert_group(number)


def year_to_polish_words(year):
    if not isinstance(year, (int, str)):
        return year

    year = int(str(year))

    if year < 1000 or year > 2099:
        return str(year)

    centuries = {
        20: "dwutysięczny",
        19: "tysiąc dziewięćsetny",
        18: "tysiąc osiemsetny",
        17: "tysiąc siedemsetny",
        16: "tysiąc sześćsetny",
    }

    def get_year_end(num):
        if num == 0:
            return ""
        elif num < 10:
            return ["pierwszy", "drugi", "trzeci", "czwarty", "piąty",
                    "szósty", "siódmy", "ósmy", "dziewiąty"][num-1]
        elif num < 20:
            return ["dziesiąty", "jedenasty", "dwunasty", "trzynasty", "czternasty",
                    "piętnasty", "szesnasty", "siedemnasty", "osiemnasty",
                    "dziewiętnasty"][num-10]
        elif num < 100:
            tens = ["dwudziesty", "trzydziesty", "czterdziesty", "pięćdziesiąty",
                    "sześćdziesiąty", "siedemdziesiąty", "osiemdziesiąty",
                    "dziewięćdziesiąty"][num//10-2]
            if num % 10 == 0:
                return tens
            return tens + " " + get_year_end(num % 10)

    century = year // 100
    year_end = year % 100

    if year_end == 0:
        return centuries[century]
    else:
        base = "dwa tysiące" if century == 20 else "tysiąc dziewięćset"
        return f"{base} {get_year_end(year_end)}"
# --------------------------Ostatnie słowa do klas. rymów-------------------------


def get_last_words(text, max_lines=10):
    lines = text.strip().split("\n")
    words = []
    for line in lines[:max_lines]:
        tokens = re.findall(r"\b\w+\b", line.lower())
        if tokens:
            words.append(tokens[-1])
    return " ".join(words)


def add_rhyme_column(df, source_column="content", target_column="rhyme_input"):
    df[target_column] = df[source_column].apply(get_last_words)
    return df


def add_phonetic_column(df, column_name ='rhyme_input', new_column='rhyme_phonetic'):
    epi = epitran.Epitran('pol-Latn')

    def to_phonetic(word):
        try:
            if isinstance(word, str):
                return epi.transliterate(word.lower())
            return ''
        except Exception as e:
            print(f"error przy transkrypcji '{word}': {e}")
            return ''

    df[new_column] = df[column_name].apply(to_phonetic)
    return df


if __name__ == "__main__":
    # normalize_csv('nonnull_poems.csv', 'normalized_poems.csv', content_column='content')

    df = pd.read_csv('./data/wiersze_rhyme_col.csv')
    # add_rhyme_column(df)
    # df.to_csv('./analysis/data/wiersze_rhyme_col.csv')

    df = add_phonetic_column(df)
    df.to_csv('./data/wiersze_rhyme_phonet.csv')
