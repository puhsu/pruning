"""
Downloading and processing IMDB dataset
"""

import argparse
import collections
import html
import os
import pickle
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import requests
import spacy

import utils


class Tokenizer():
    def __init__(self, lang='en'):
        self.re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        self.re_rep = re.compile(r'(\S)(\1{3,})')
        self.re_word_rep = re.compile(r'(\b\w+\W+)(\1{3,})')
        self.re_spaces = re.compile(r'  +')

        self.tok = spacy.load(lang)
        for w in '<eos>', '<bos>', '<unk>':
            self.tok.tokenizer.add_special_case(w, [{spacy.symbols.ORTH: w}])

    def sub_br(self, x): 
        return self.re_br.sub("\n", x)

    def spacy_tok(self,x):
        return [t.text for t in self.tok.tokenizer(self.sub_br(x))]

    def fixup(self, x):
        x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
            'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
            '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
            ' @-@ ','-').replace('\\', ' \\ ')
        return self.re_spaces.sub(' ', html.unescape(x))

    @staticmethod
    def replace_rep(m):
        TK_REP = 'tk_rep'
        c, cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '

    @staticmethod
    def replace_wrep(m):
        TK_WREP = 'tk_wrep'
        c, cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '

    @staticmethod
    def do_caps(ss):
        TOK_UP = ' t_up '
        res = []
        for s in re.findall(r'\w+|\W+', ss):
            res += ([TOK_UP,s.lower()] if (s.isupper() and (len(s)>2)) else [s.lower()])
        return ''.join(res)

    def proc_text(self, s):
        s = self.fixup(s)
        s = self.re_rep.sub(Tokenizer.replace_rep, s)
        s = self.re_word_rep.sub(Tokenizer.replace_wrep, s)
        s = Tokenizer.do_caps(s)
        s = re.sub(r'([/#])', r' \1 ', s)
        s = re.sub(' {2,}', ' ', s)
        return self.spacy_tok(s)

    @staticmethod
    def proc_all(ss, lang):
        tok = Tokenizer(lang)
        return np.array([tok.proc_text(s) for s in ss])

    @staticmethod
    def proc_all_mp(ss, lang='en'):
        with ProcessPoolExecutor(os.cpu_count()) as e:
            return sum(e.map(Tokenizer.proc_all, ss, [lang]*len(ss)), [])


def download_imdb(save=Path('data'),
                  url='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'):
    """Download dataset.

    Args:
        save (pathlib.Path): path to directory where dataset files are saved
        url (str): link to dataset archive
    """
    archive = save/'aclImdb_v1.tar.gz'
    if not archive.exists():
        print('Downloading')
        with open(archive, 'wb') as f:
            r = requests.get(url)
            f.write(r.content)

    if not (save/'aclImdb').exists():
        print('Extracting archive')
        with tarfile.open(name=archive, mode='r:gz') as tar:
            tar.extractall(path=save)

ClASSES = ['pos', 'neg', 'unsup']

def dir2np(path, save_unsup=True):
    """Converts dataset directory to numpy arrays containing texts and labels
    
    Args:
        path (pathlib.Path): path to directory, where dataset is stored
        save (str): filename for output file
        save_unsup (bool): whether or not to include unlabeled reviews
    
    Returns:
        res (tuple): tuple with numpy arrays (texts, labels)
    """
    texts, labels = [], []
    for i, cl in enumerate(ClASSES):
        for fname in (path/cl).glob('*.*'):
            texts.append(fname.open('r').read())
            labels.append(i)
    
    np.random.seed(42)
    idx = np.random.permutation(range(len(texts)))
    texts = np.array(texts)[idx]
    labels = np.array(labels)[idx]

    if not save_unsup:
        mask = np.ones_like(labels, dtype=bool)
        mask[np.argwhere(labels == 2)] = False
        texts = texts[mask, ...]
        labels = labels[mask, ...]

    return texts, labels

def getitos(texts, max_vocab=60000, min_freq=2):
    freq = collections.Counter(t for text in texts for t in text)
    itos = [t for t, c in freq.most_common(max_vocab) if c > min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    return itos

def classifier_data():
    print('Downloading and extracting data')
    download_imdb()

    print('Processing directories')
    trn_texts, trn_labels = dir2np(Path('data/aclImdb/train/'), save_unsup=False)
    val_texts, val_labels = dir2np(Path('data/aclImdb/test/'), save_unsup=False)

    print('Tokenizing texts')
    trn_texts = Tokenizer().proc_all_mp(utils.partition_by_cores(trn_texts))
    val_texts = Tokenizer().proc_all_mp(utils.partition_by_cores(val_texts))

    print('Numericalizing tokens')
    itos = getitos(trn_texts)
    pickle.dump(itos, open('data/itos.pkl', 'wb'))

    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
    trn_ids = np.array([[stoi[t] for t in text] for text in trn_texts])
    val_ids = np.array([[stoi[t] for t in text] for text in val_texts])

    np.savez('data/train.npz', texts=trn_ids, labels=trn_labels)
    np.savez('data/test.npz', texts=val_ids, labels=val_labels)


classifier_data()
