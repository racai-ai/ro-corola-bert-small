import os
import torch
from ro_wordpiece import RoBertPreTrainedTokenizer
from transformers import AutoModelForMaskedLM

corola_vocab_file = os.path.join(
    '..', 'ro-wordpiece-tokenizer', 'model', 'vocab.txt')
tokenizer = RoBertPreTrainedTokenizer.from_pretrained(
    corola_vocab_file, model_max_length=256)
model = AutoModelForMaskedLM.from_pretrained(
    "model\\checkpoint-1279000")

input_text = "Copiii vor să se [MASK]."

inputs = tokenizer(input_text, return_tensors="pt")
mask_token_index = torch.where(
    inputs["input_ids"] == tokenizer.mask_token_id)[1]
logits = model(**inputs).logits
mask_token_logits = logits[0, mask_token_index, :]
top_k_tokens = torch.topk(mask_token_logits, k=20, dim=1).indices[0].tolist()

for token in top_k_tokens:
    print(input_text.replace(tokenizer.mask_token, tokenizer.decode([token])))
# end for
