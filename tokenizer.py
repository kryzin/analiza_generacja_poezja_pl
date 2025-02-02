import re
import pandas as pd
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")


def prepare_data(file_path='nonnull_poems.csv'):
    df = pd.read_csv(file_path)

    with open('before_tokenizer_poems.csv', 'w', encoding='utf-8') as f:
        for poem in df['content']:
            if isinstance(poem, str):
                poem = poem.strip()
                poem = re.sub(r'\n\s*\n', ' <stanza> ', poem)
                poem = poem.replace('\n', ' <line> ')
                f.write(poem + '\n')


def train_tokenizer(input_file='before_tokenizer_poems.csv', vocab_size=50000):
    special_tokens = [
        '<stanza>', '<line>'
    ]

    SentencePieceTrainer.train(
        input=input_file,
        model_prefix='pl_poetry_tokenizer',
        vocab_size=vocab_size,
        character_coverage=0.9995,  # DO PRZETESTOWANIA CZY MNIEJ/WIĘCEJ
        user_defined_symbols=special_tokens,
        model_type='unigram',
        normalization_rule_name='nmt_nfkc',
        split_by_whitespace=True
    )


def validate_tokenizer(tokenizer, csv_path, num_samples=5):
    df = pd.read_csv(csv_path)
    samples = df['content'].sample(n=num_samples, random_state=42)

    for i, poem in enumerate(samples, 1):
        if isinstance(poem, str):
            tokens = tokenizer.encode_as_pieces(poem)
            print(f"\nwiersz --- {i}:")
            print("część wiersza:\n", poem[:100], "...")
            print("tokeny:\n", tokens[:20], "...")
            print(f"ilość tokenów: {len(tokens)}")


if __name__ == "__main__":
    prepare_data()
    train_tokenizer()

    sp = SentencePieceProcessor()
    sp.load('pl_poetry_tokenizer.model')

    # df = pd.read_csv('nonnull_poems.csv')
    # sample = df['content'].iloc[0]  # testowo jeden
    # tokens = sp.EncodeAsPieces(sample)
    # print(tokens)

    validate_tokenizer(sp, 'nonnull_poems.csv')
