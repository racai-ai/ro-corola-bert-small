import sys
import os
from transformers import BertForMaskedLM, BertConfig
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from ro_wordpiece import RoBertPreTrainedTokenizer
from corola_data import corola

_corola_wordpiece_vocab = os.path.join(
    '..', 'ro-wordpiece-tokenizer', 'model', 'vocab.txt')
_bert_input_size: int = 256
_tokenizer: RoBertPreTrainedTokenizer = \
    RoBertPreTrainedTokenizer.from_pretrained(
            _corola_wordpiece_vocab, model_max_length=_bert_input_size)


def text_block_function(examples: dict):
    """If an example has less than or equal to `_bert_input_size` tokens,
    it is right padded. If not, consecutive windows of `_bert_input_size` tokens
    are added to the data set."""
    
    batch_size = len(examples['input_ids'])
    result = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': []
    }

    for i in range(batch_size):
        iids = examples['input_ids'][i]
        attn = examples['attention_mask'][i]
        tyid = examples['token_type_ids'][i]

        if len(iids) <= _bert_input_size:
            result['input_ids'].append(iids)
            result['attention_mask'].append(attn)
            result['token_type_ids'].append(tyid)
        else:
            for j in range(0, len(iids) - _bert_input_size + 1):
                result['input_ids'].append(iids[j:j + _bert_input_size])
                result['attention_mask'].append(attn[j:j + _bert_input_size])
                result['token_type_ids'].append(tyid[j:j + _bert_input_size])
            # end for
        # end if
    # end for

    return result


def tokenization_function(examples: dict):
    result = _tokenizer(text=examples['text'], padding='max_length')
    return result


def check_vocabulary() -> bool:
    """There is a bug that causes tokenizers.models.WordPiece to save
    the same token multiple times. This method checks that all IDs 
    in the vocabulary are consecutive."""
    
    trained_vocab = _tokenizer.get_vocab()
    trained_terms = sorted(trained_vocab.keys(), key=lambda x: trained_vocab[x])
    prev_id = -1
    good_to_go = True

    for term in trained_terms:
        if prev_id == -1:
            prev_id = trained_vocab[term]
        elif trained_vocab[term] > prev_id + 1:
            print(f'Vocabulary term [{term}] has ID [{trained_vocab[term]}] and previous ID is [{prev_id}]')
            prev_id = trained_vocab[term]
            good_to_go = False
        else:
            prev_id = trained_vocab[term]
        # end if
    # end for

    return good_to_go


# Actual runtime parameters:
# num_proc=30
# per_device_train_batch_size=16
# per_device_eval_batch_size=16
# test_size=0.001
# GPU 0: NVIDIA Quadro RTX 8000, 48601 MiB of RAM
# 1 epoch in approx. 30 days
if __name__ == '__main__':
    print(f'Running with BERT input size of [{_bert_input_size}]', file=sys.stderr, flush=True)
    print(f'Running with vocab.txt from [{_corola_wordpiece_vocab}]', file=sys.stderr, flush=True)

    if not check_vocabulary():
        exit(1)
    # end if

    # 1. Tokenized the CoRoLa dataset
    tokenized_corola = corola.map(tokenization_function,
                                  batched=True, num_proc=6,
                                  remove_columns=['text'],
                                  new_fingerprint="tokenization_function_v1")
    
    # 2. If an example has more than _bert_input_size tokens,
    # create consecutive blocks of tokens of _bert_input_size length.
    lm_corola = tokenized_corola.map(text_block_function,
                                     batched=True, num_proc=6,
                                     new_fingerprint="text_block_function_v1")
    lm_train_ready = lm_corola.train_test_split(test_size=0.1)

    # 3. Instantitate BERT mini    
    bert_config = BertConfig(vocab_size=_tokenizer.vocab_size,
                             hidden_size=256, num_hidden_layers=4,
                             num_attention_heads=8, max_position_embeddings=_bert_input_size,
                             pad_token_id=_tokenizer.pad_token_id)
    mini_bert_model = BertForMaskedLM(config=bert_config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=_tokenizer, mlm_probability=0.15)
    
    # 4. Train
    training_args = TrainingArguments(
        output_dir="model",
        evaluation_strategy="steps",
        eval_steps=10000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=10,
        learning_rate=2e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        push_to_hub=False
    )

    trainer = Trainer(
        model=mini_bert_model,
        args=training_args,
        train_dataset=lm_train_ready["train"],
        eval_dataset=lm_train_ready["test"],
        data_collator=data_collator,
    )

    trainer.train()
