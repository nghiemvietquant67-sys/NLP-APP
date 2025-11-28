#!/usr/bin/env python3
import torch
from transformers import pipeline, AutoTokenizer, AutoModel


def run_fill_mask():
    mask_filler = pipeline("fill-mask")
    input_sentence = "Hanoi is the [MASK] of Vietnam."
    predictions = mask_filler(input_sentence, top_k=3)
    print("Fill-Mask results:")
    for i, p in enumerate(predictions, 1):
        print(i, p['token_str'], p['score'])


def run_text_generation():
    generator = pipeline("text-generation", model="gpt2")
    prompt = "The best thing about learning NLP is"
    gen = generator(prompt, max_length=50, num_return_sequences=1)
    print("\nText generation result:")
    print(gen[0]['generated_text'])


def run_sentence_embedding():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    sentence = "This is a sample sentence."
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    sentence_embedding = sum_embeddings / sum_mask
    print("\nSentence embedding shape:", sentence_embedding.shape)


if __name__ == '__main__':
    run_fill_mask()
    run_text_generation()
    run_sentence_embedding()
