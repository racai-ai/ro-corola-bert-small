import os
from datasets import Dataset, Features, Value


_corola_sentences_folder = os.path.join(
    '..', 'ro-wordpiece-tokenizer', 'corola-sentences2')


def corola_sentences_generator():
    for txt in os.listdir(path=_corola_sentences_folder):
        if txt.endswith('.txt'):
            txt_file = os.path.join(_corola_sentences_folder, txt)

            with open(txt_file, mode='r', encoding='utf-8') as f:
                for sentece in f:
                    yield {'text': sentece.strip()}
                # end for
            # end with
        # end if
    # end for


corola = Dataset.from_generator(generator=corola_sentences_generator, features=Features(
    {'text': Value(dtype='string')}), cache_dir='corola_cache')
