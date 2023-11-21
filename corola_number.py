# This script tries to find the singular/plural encoding of
# a set of a 100 Romanian nouns in the output of the BERT model.

import sys
import os
from pathlib import Path
from random import shuffle, random
import torch
from tqdm import tqdm
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.markers import MarkerStyle
from scipy.stats import spearmanr
from ro_wordpiece import RoBertPreTrainedTokenizer
from transformers import AutoModel

corola_vocab_file = os.path.join(
    '..', 'ro-wordpiece-tokenizer', 'model', 'vocab.txt')
tokenizer = RoBertPreTrainedTokenizer.from_pretrained(
    corola_vocab_file, model_max_length=256)
model = AutoModel.from_pretrained("model\\checkpoint-1279000")


def get_word_encoding(input_text: str, word: str) -> Tensor | None:
    """Tokenizes the text using the `tokenizer` above and then
    searches for `word` (which should be a token). Returns its encoding,
    as provided by the `model` above."""

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
    # end with

    word_id = tokenizer._convert_token_to_id(word)
    word_index_tns = torch.where(inputs["input_ids"] == word_id)[1]

    if word_index_tns.nelement() > 0:
        word_index = word_index_tns[0]

        if type(word_index) is Tensor:
            word_index = word_index.item()
        # end if

        if word_index >= 0 and \
                word_index < outputs.last_hidden_state.shape[1]:
            return outputs.last_hidden_state[0, word_index]
        # end if
    # end if

    return None


def get_folder_encodings(data_folder: str) -> dict[str, list[float]]:
    result = {}

    for txt in os.listdir(path=data_folder):
        if txt.endswith('.txt'):
            word = Path(txt).stem
            txt = os.path.join(data_folder, txt)
            result[word] = []
            flt = os.path.join(data_folder, word + '.vec')

            if os.path.isfile(flt):
                print(f'Reading encodings from file {flt}', file=sys.stderr, flush=True)

                with open(flt, mode='r', encoding='utf-8') as f:
                    for line in f:
                        f_list = [float(x) for x in line.strip().split()]
                        result[word].append(f_list)
                    # end for
                #end with

                continue
            # end if

            with open(txt, mode='r', encoding='utf-8') as f:
                all_lines = f.readlines()
            # end with

            for line in tqdm(all_lines, desc=f'{txt}'):
                context = line.strip()

                if context:
                    vec = get_word_encoding(input_text=context, word=word)
                else:
                    vec = None
                # end if

                if vec is not None:
                    result[word].append(vec.tolist())
                # end if
            # end for

            with open(flt, mode='w', encoding='utf-8') as f:
                for vec in result[word]:
                    print(' '.join([str(x) for x in vec]), file=f)
                # end for
            # end with
        # end if
    # end for

    return result


def compute_dim_correlations(examples: list[float]) -> list[tuple]:
    """List of encodings, each 256 in length."""

    correlations = []
    dimension_vectors = []

    for i in range(256):
        x = []
        
        for vec in examples:
            x.append(vec[i])
        # end for

        dimension_vectors.append(x)
    # end for

    for i in range(255):
        exi = dimension_vectors[i]
        p_exi = [(x, y) for x, y in zip(exi, range(256))]
        sp_exi = sorted(p_exi, key=lambda x: x[0])
        x = [a for a, _ in sp_exi]

        for j in range(i + 1, 256):
            exj = dimension_vectors[j]
            y = [exj[b] for _, b in sp_exi]

            sr = spearmanr(a=x, b=y)

            if sr.correlation >= 0.75:
                # They say Spearman R >= 0.6 is moderate
                # and >= 0.7 is strong
                correlations.append((i, j, sr.correlation))
            # end if
        # end for
    # end for

    return correlations


def plot_samples(word: str, samples: list[float], marker: str, color: str, ax):
    x = []
    y = []
    r = []

    for i in tqdm(range(50), desc=f'{word}'):
        shuffle(samples)
        train_examples = samples[0:100]
        corr_results = compute_dim_correlations(examples=train_examples)
        corr_file = f'{word}-{i}.spr'

        with open(corr_file, mode='w', encoding='utf-8') as f:
            for cx, cy, cr in corr_results:
                if random() < 0.5:
                    x.append(cx - (1 + random()))
                else:
                    x.append(cx + (1 + random()))
                # end if

                if random() < 0.5:
                    y.append(cy - (1 + random()))
                else:
                    y.append(cy + (1 + random()))
                # end if

                r.append(cr)
                print(f'{cx}\t{cy}\t{cr:.5f}', file=f)
            # end for
        # end with
    # end for

    ax.scatter(x, y, s=200, color=color, marker=marker, label=word)


def plot_word_group(examples: dict[str, list[float]], words: list[str], fig_file: str):
    plt_markers = [
        'o', 'v', '^', '<', '>',
        '8', 's', 'p', '*', 'h', 'H',
        'D', 'd', 'P', 'X', '.']
    plt_colors = [
        'black', 'blue', 'brown', 'crimson', 'darkblue', 'yellowgreen',
        'darkgreen', 'fuchsia', 'green', 'orange', 'pink', 'magenta',
        'violet', 'aquamarine', 'lavender', 'lightgreen']
    fig, ax = plt.subplots(figsize=(21.22, 12.64), dpi=200)

    for i, word in enumerate(words):
        plot_samples(word, samples=examples[word], marker=MarkerStyle(
            plt_markers[i], fillstyle='none'), color=plt_colors[i], ax=ax)
    # end for

    ax.legend()
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which='major', color='black', linestyle='solid')
    ax.grid(which='minor', color='grey', linestyle='--')
    
    fig.savefig(fig_file, dpi=200, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    word_examples = get_folder_encodings(data_folder='data5')
    plot_word_group(examples=word_examples, words=[
                    'ani', 'autori', 'lei', 'oameni', 'medici'], fig_file='m1-50-75.png')
    plot_word_group(examples=word_examples, words=[
                    'anii', 'autorii', 'leii', 'oamenii', 'medicii'], fig_file='m2-50-75.png')
    plot_word_group(examples=word_examples, words=[
                    'anilor', 'autorilor', 'leilor', 'oamenilor', 'medicilor'], fig_file='m3-50-75.png')
