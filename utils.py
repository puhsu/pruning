import html
import os
import re
import random
import pickle
import tarfile
import collections
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


import numpy as np
import torch
import spacy
import requests
import torch.utils.data
from ruamel.yaml import YAML


def partition(a, sz):
    return [a[i:i+sz] for i in range(0, len(a), sz)]


def partition_by_cores(a):
    return partition(a, len(a) // os.cpu_count() + 1)


# TODO refactor tokenizer (too many fixup, sub... functions which logically do same things)
# NOTE Maybe make this a function
class Tokenizer():
    def __init__(self, lang='en_core_web_sm'):
        self.re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        self.re_rep = re.compile(r'(\S)(\1{3,})')
        self.re_word_rep = re.compile(r'(\b\w+\W+)(\1{3,})')
        self.re_spaces = re.compile(r'  +')

        self.tok = spacy.load(lang)
        for w in '<eos>', '<bos>', '<unk>':
            self.tok.tokenizer.add_special_case(w, [{spacy.symbols.ORTH: w}])

    def sub_br(self, x):
        return self.re_br.sub("\n", x)

    def spacy_tok(self, x):
        return [t.text for t in self.tok.tokenizer(self.sub_br(x))]

    def fixup(self, x):
        x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
            'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
            '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
            ' @-@ ', '-').replace('\\', ' \\ ')
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
            res += ([TOK_UP, s.lower()] if (s.isupper() and (len(s) > 2)) else [s.lower()])
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
        return [tok.proc_text(s) for s in ss]

    @staticmethod
    def proc_all_mp(ss, lang='en_core_web_sm'):
        with ProcessPoolExecutor(os.cpu_count()) as e:
            return sum(e.map(Tokenizer.proc_all, ss, [lang]*len(ss)), [])


###################################################################################
#  Classiffication
###################################################################################


class TextDataset(torch.utils.data.Dataset):
    classes = ['neg', 'pos', 'unsup']
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

    def __init__(self, *,  train, load, save_path=Path('data'),
                 max_vocab=60000, min_freq=2, seed=42):
        """Initialize dataset for text classification"""
        self.save_path = save_path
        self.max_vocab = max_vocab
        self.min_freq = min_freq

        if load:
            data_path = save_path/'dataset'/'train.pkl' if train else save_path/'dataset'/'valid.pkl'
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.texts, self.labels = zip(*data)
            print('Loaded dataset of length:', len(self.texts))
            return

        random.seed(seed)
        self._download_ds()
        train_path = save_path/'aclImdb'/'train'
        valid_path = save_path/'aclImdb'/'test'
        # extract texts from files
        print('Extracting texts')
        train_texts, train_labels = self._get_texts(train_path)
        valid_texts, valid_labels = self._get_texts(valid_path)
        # tokenize texts
        print('Tokenizing train texts')
        train_texts = Tokenizer().proc_all_mp(partition_by_cores(train_texts))
        print('Tokenizing validation texts')
        valid_texts = Tokenizer().proc_all_mp(partition_by_cores(valid_texts))

        # numericalize tokens
        print('Numericalizing tokens')
        itos = self._generate_itos(train_texts + valid_texts)
        stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})
        # filter out unsup
        print('Deleting unsup labels')
        train_data = []
        for text, label in zip(train_texts, train_labels):
            if label != 2:
                train_data.append((list(map(lambda t: stoi[t], text)), label))

        valid_data = []
        for text, label in zip(valid_texts, valid_labels):
            if label != 2:
                valid_data.append((list(map(lambda t: stoi[t], text)), label))

        # train_data = [(list(map(lambda t: stoi[t], text)), label)
        #               for text, label in zip(train_texts, train_labels) if label != 2]
        # valid_data = [(list(map(lambda t: stoi[t], text)), label)
        #               for text, label in zip(valid_texts, valid_labels) if label != 2]

        self.texts, self.labels = map(list, zip(*train_data)) if train else map(list, zip(*valid_data))

        dataset_path = save_path/'dataset'
        print('Saving dataset to', dataset_path)
        if not dataset_path.exists():
            os.mkdir(dataset_path)
        with open(dataset_path/'train.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open(dataset_path/'valid.pkl', 'wb') as f:
            pickle.dump(valid_data, f)
        with open(dataset_path/'itos.pkl', 'wb') as f:
            pickle.dump(itos, f)

    def _generate_itos(self, texts):
        # get all texts in one list
        freq = collections.Counter(t for text in texts for t in text)
        print(f'Total number of tokens: {len(freq)}')
        print('Top 10 tokens:')
        print(freq.most_common(10))

        itos = [t for t, c in freq.most_common(self.max_vocab) if c > self.min_freq]
        itos.insert(0, '_pad_')
        itos.insert(0, '_unk_')
        print('Generated itos array of length', len(itos))
        return itos

    def _download_ds(self):
        archive = self.save_path/'aclImdb_v1.tar.gz'
        if not archive.exists():
            print('Downloading')
            with open(archive, 'wb') as f:
                r = requests.get(self.url)
                f.write(r.content)

        if not (self.save_path/'aclImdb').exists():
            print('Extracting archive')
            with tarfile.open(name=archive, mode='r:gz') as tar:
                tar.extractall(path=self.save_path)

    def _get_texts(self, path):
        texts, labels = [], []
        for i, cl in enumerate(self.classes):
            for fname in (path/cl).glob('*.txt'):
                texts.append(fname.open('r').read())
                labels.append(i)

        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts[:], labels[:] = zip(*combined)
        return texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


class TextSampler(torch.utils.data.Sampler):
    """Sampler for sequence data. Samples by sequence length, so that batches
    have roughly same length sequences."""

    def __init__(self, data_source, key, batch_size):
        self.data_source = data_source
        self.key = key
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data_source))
        # sort chunks by size
        chunk_size = self.batch_size * 50
        chunk_idxs = [idxs[i:i+chunk_size] for i in range(0, len(idxs), chunk_size)]
        idxs = np.concatenate([sorted(ck, key=self.key, reverse=True) for ck in chunk_idxs])

        # sort smaller chunks by size
        chunk_size = self.batch_size
        chunk_idxs = [idxs[i:i+chunk_size] for i in range(0, len(idxs), chunk_size)]
        # move chunk with longest key to the beggining
        max_chunk = np.argmax([self.key(ck[0]) for ck in chunk_idxs])
        chunk_idxs[0], chunk_idxs[max_chunk] = chunk_idxs[max_chunk], chunk_idxs[0]

        # get final order of elements
        idxs = np.concatenate(np.random.permutation(chunk_idxs[1:]))
        idxs = np.concatenate((chunk_idxs[0], idxs))
        return iter(idxs)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences given as numpy arrays.

    Note:
        Should only be used with data of form [(seq, target)],
        where seq is one dimentional array.

    Example:
        dl = DataLoader(ds, ..., collate_fn=PadCollate(pad_idx=0))
    """

    def __init__(self, pad_idx=1, max_len=None):
        self.pad_idx = pad_idx
        self.max_len = max_len

    def pad_sequence(self, seq, max_len):
        res = [self.pad_idx] * max_len
        seq = seq[:max_len]
        res[-len(seq):] = seq
        return torch.tensor(res)

    def pad_collate(self, batch):
        """
        args:
            batch - list of (ndarray, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """

        # find longest sequence and pad according to it length
        max_len = self.max_len
        if max_len is None:
            max_len = max(map(lambda x: len(x[0]), batch))

        xs = torch.stack(tuple(self.pad_sequence(x, max_len) for x, y in batch))
        ys = torch.LongTensor(tuple(y for x, y in batch))
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


###################################################################################
#  Training models
###################################################################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0

    def __call__(self):
        return self.total / float(self.steps)

    def update(self, val):
        self.total += val
        self.steps += 1

###################################################################################
#  Serialization of models and configs
###################################################################################


def save_encoder(model):
    pass


def load_encoder(model):
    """
    Used to load
    """
    pass


def parse_config(path):
    if isinstance(path, str):
        path = Path(path)
    assert path.exists()

    yaml = YAML()
    return yaml.load(path)
