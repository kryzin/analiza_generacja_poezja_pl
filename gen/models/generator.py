import openai
import re
import random
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_MODEL = None
HF_TOKENIZER = None


def load_defaults(path="defaults.json"):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return (
            data,
            data["themes"],
            data["keywords"],
            data["forms"],
            data["rhyme_schemes"]
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"error at {path}: {e}")
        sys.exit(1)


def build_prompt(theme, keywords, form, rhyme_scheme):
    prompt = f"""Napisz {form.lower()} na temat: '{theme}'.\n"""
    if keywords:
        prompt += f"Uwzględnij następujące słowa kluczowe: {', '.join(keywords)}.\n"
    if rhyme_scheme:
        prompt += f"Układ rymów: {rhyme_scheme.upper()}\n"
    prompt += "Zadbaj o poetycki styl, obrazowość i emocjonalny wydźwięk."
    return prompt


def generate_poem(prompt, config, max_tokens=300, temperature=1.0, top_p=0.9):
    provider = config.get("provider", "openai")

    if provider == "openai":
        client = openai.OpenAI(api_key=config.get("openai_api_key", ""))
        try:
            response = client.chat.completions.create(
                model=config.get("openai_model", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI error: {e}"

    elif provider == "huggingface":
        global HF_MODEL, HF_TOKENIZER
        model_name = config.get("hf_model", "speakleash/Bielik-4.5B-v3.0-Instruct")
        if HF_MODEL is None or HF_TOKENIZER is None:
            HF_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            HF_MODEL = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = HF_TOKENIZER(prompt, return_tensors="pt").to(device)
        try:
            output = HF_MODEL.generate(
                **input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_k=50,
                top_p=top_p,
                temperature=temperature
            )
            return HF_TOKENIZER.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            return f"HF error: {e}"

    else:
        return "error: wrong provider"


def extract_last_words(poem_text):
    lines = [line.strip() for line in poem_text.strip().split('\n') if line.strip()]
    return [re.findall(r'\b\w+\b', line)[-1].lower() for line in lines if re.findall(r'\b\w+\b', line)]


def rhyme_suffix(word, length=3):
    return word[-length:] if len(word) >= length else word


def get_first_stanza_rhyme_info(poem_text):
    lines = [line.strip() for line in poem_text.strip().split('\n') if line.strip()]
    stanza = []
    for line in lines:
        if not line:
            break
        stanza.append(line)
        if len(stanza) >= 4:
            break
    last_words = [re.findall(r'\b\w+\b', l)[-1].lower() for l in stanza if re.findall(r'\b\w+\b', l)]
    suffixes = [rhyme_suffix(w) for w in last_words]
    return suffixes


def check_rhyme_scheme(suffixes, scheme):
    scheme = scheme.replace(" ", "").upper()
    if len(suffixes) != len(scheme):
        return False
    rhyme_groups = {}
    for idx, symbol in enumerate(scheme):
        if symbol not in rhyme_groups:
            rhyme_groups[symbol] = suffixes[idx]
        elif rhyme_groups[symbol] != suffixes[idx]:
            return False
    return True


def run_interactive(config_path="defaults.json"):
    config, DEFAULT_THEMES, DEFAULT_KEYWORDS, DEFAULT_FORMS, DEFAULT_RHYME_SCHEMES = load_defaults(config_path)

    theme = input("Motyw wiersza [Enter = losowy]: ").strip()
    keywords_input = input("Słowa kluczowe (oddzielone przecinkami) [Enter = losowe]: ").strip()
    form = input("Forma wiersza [Enter = losowa]: ").strip()
    rhyme_scheme = input("Układ rymów [Enter = losowy]: ").strip()

    if not theme:
        theme = random.choice(DEFAULT_THEMES)
        print("theme\n", theme)
    else: 
        theme = [k.strip() for k in theme.split(",") if k.strip()]
    if not keywords_input:
        keywords = random.sample(DEFAULT_KEYWORDS, k=3)
        print("keywords\n", keywords)
    else:
        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
    if not form:
        form = random.choice(DEFAULT_FORMS)
        print("form\n", form)
    if not rhyme_scheme:
        rhyme_scheme = random.choice(DEFAULT_RHYME_SCHEMES)
        print("rhyme\n", rhyme_scheme)

    prompt = build_prompt(theme, keywords, form, rhyme_scheme)

    print(f"\n[🔍 MODEL: {config['provider']}] Generuję wiersz...\n")
    poem = generate_poem(prompt, config)
    print(poem)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "defaults.json"
    run_interactive(config_path)
