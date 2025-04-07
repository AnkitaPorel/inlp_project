# -*- coding: utf-8 -*-

from transformers import AutoTokenizer

tokenizer_names = [
    "ai4bharat/indic-bert",
    "sarvamai/sarvam-1",
    "aashay96/indic-gpt",
    "microsoft/Phi-4-mini-instruct",
    "google/gemma-3-1b-pt",
    "google/gemma-2b",
    "ai4bharat/indictrans2-en-indic-1B",
]

for name in tokenizer_names:
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    print(f"Tokenizer: {name}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"Mask token ID: {tokenizer.mask_token_id}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")

    print("\n")

    text = "আমি ভাত খাই। সে বাজারে যায়। তিনি কি সত্যিই ভালো মানুষ?"
    tokens = tokenizer.tokenize(text)
    print(f"Tokenizer output: {tokens}")

    print("\n\n")
