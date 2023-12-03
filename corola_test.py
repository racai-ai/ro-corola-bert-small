import torch
from rwpt import load_ro_pretrained_tokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import PreTrainedTokenizer

def corola_tokenizer_model() -> tuple[PreTrainedTokenizer, AutoModelForMaskedLM]:
    tokenizer = load_ro_pretrained_tokenizer(max_sequence_len=256)
    model = AutoModelForMaskedLM.from_pretrained("model\\checkpoint-1279000")
    return tokenizer, model


def robert_tokenizer_model() -> tuple[PreTrainedTokenizer, AutoModelForMaskedLM]:
    tokenizer = AutoTokenizer.from_pretrained('dumitrescustefan/bert-base-romanian-cased-v1')
    model = AutoModelForMaskedLM.from_pretrained(
        'dumitrescustefan/bert-base-romanian-cased-v1')
    return tokenizer, model

def run_example(input_text: str, tokenizer: PreTrainedTokenizer, model: AutoModelForMaskedLM):
    inputs = tokenizer(input_text, return_tensors="pt")
    mask_token_index = torch.where(
        inputs["input_ids"] == tokenizer.mask_token_id)[1]
    logits = model(**inputs).logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_logits, k=20, dim=1).indices[0].tolist()

    for token in top_k_tokens:
        print(input_text.replace(tokenizer.mask_token, tokenizer.decode([token])))
    # end for


if __name__ == '__main__':
    tokenizer, model = corola_tokenizer_model()
    run_example('El [MASK] a urcat în tren.', tokenizer, model)
