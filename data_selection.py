import os

def repl_sgml_wih_utf8(word: str) -> str:
    word = word.replace("&abreve;", "ă")
    word = word.replace("&acirc;", "â")
    word = word.replace("&icirc;", "î")
    word = word.replace("&scedil;", "ș")
    word = word.replace("&tcedil;", "ț")
    word = word.replace("&Abreve;", "Ă")
    word = word.replace("&Acirc;", "Â")
    word = word.replace("&Icirc;", "Î")
    word = word.replace("&Scedil;", "Ș")
    word = word.replace("&Tcedil;", "Ț")

    return word


def pl_noun_selection() -> list[tuple]:
    """Reads 'tbl.wordform.ro' and selects the top 10, most
    frequent plural nouns, according to the CoRoLa word frequency."""

    tbl_file = os.path.join('..', 'Rodna', 'data',
                            'resources', 'tbl.wordform.ro')
    tbl = {}
    tbl_pos = {}

    with open(tbl_file, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line.startswith('#'):
                word, _, msd = line.split()
                word = repl_sgml_wih_utf8(word)

                if word not in tbl:
                    tbl[word] = set()
                # end if

                if word not in tbl_pos:
                    tbl_pos[word] = set()
                # end if

                tbl[word].add(msd)
                tbl_pos[word].add(msd[0])
            # end if
        # end for
    # end with

    corola_freq_file = os.path.join(
        '..', 'ro-wordpiece-tokenizer', 'corola-vocabulary.txt')
    corola_freq = {}

    with open(corola_freq_file, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            word, freq = line.split()
            corola_freq[word] = int(freq)
        # end for
    # end with

    corola_pl_nouns = []

    for word in tbl_pos:
        if len(tbl_pos[word]) == 1 and 'N' in tbl_pos[word] and \
                word in corola_freq:
            # This is a noun, only, that appears in CoRoLa
            # Check if it can only be plural
            only_plural = True

            for m in tbl[word]:
                if len(m) < 4 or m[3] != 'p':
                    only_plural = False
                    break
                # end if
            # end for

            if only_plural and corola_freq[word] >= 5000:
                corola_pl_nouns.append((word, corola_freq[word]))
            # end if
    # end for

    return corola_pl_nouns


if __name__ == '__main__':
    words = pl_noun_selection()
    words = sorted(words, key=lambda x: x[1], reverse=True)

    for w, f in words:
        print(f'{w}\t{f}')
    # end for
