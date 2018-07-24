"""
Classes for loading and preprocessing data
"""

import os
import torch
import spacy
import pickle

from tqdm import tqdm

class Dictionary():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus():
    def __init__(self, path, load=False):
        self.dictionary = Dictionary()
        self.tokenizer = spacy.load('en')
        if load:
            self.load(path)
        else:
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        print(f'Building dictionary for {path}')
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in tqdm(f.readlines()):
                words = [t.text.lower() for t in self.tokenizer(line, disable=['parser', 'tagger', 'ner'])] + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        print(f'Numericalizing text for {path}')
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in tqdm(f.readlines()):
                words = [t.text.lower() for t in self.tokenizer(line, disable=['parser', 'tagger', 'ner'])] + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
    
    def _get_save_path(self, path):
        path = os.path.join(path, 'corpus')
        return os.path.join(path, 'dictionary.pkl'), \
               os.path.join(path, 'train.tr'), \
               os.path.join(path, 'valid.tr')

    def save(self, path):
        """
        Saves all contents of current object to directory given by path.
        """
        dictpath, trainpath, validpath = self._get_save_path(path)

        with open(dictpath, 'wb') as f:
            pickle.dump(self.dictionary, f)
        
        torch.save(self.train, trainpath)
        torch.save(self.valid, validpath)


    def load(self, path):
        """
        Loads previously saved information from disk.
        """
        dictpath, trainpath, validpath = self._get_save_path(path)
        assert all(map(os.path.exists, (dictpath, trainpath, validpath)))

        with open(dictpath, 'rb') as f:
            self.dictionary = pickle.load(f)
        
        self.train = torch.load(trainpath)
        self.valid = torch.load(validpath)
