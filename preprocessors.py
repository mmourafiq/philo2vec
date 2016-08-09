# -*- coding: utf-8 -*-

import string
from collections import Counter, defaultdict
from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer

from utils import time_


class StemmingLookup(object):
    word_lookup = defaultdict(dict)
    stemmer = EnglishStemmer()

    @classmethod
    def stem(cls, word):
        """
        Stems a word and updates the reverse lookup.
        """
        # Stem the word
        stemmed = cls.stemmer.stem(word)

        # Update the word lookup
        cls.word_lookup[stemmed][word] = (cls.word_lookup[stemmed].get(word, 0) + 1)

        return stemmed

    @classmethod
    def original_form(cls, word):
        """
        Returns original form of a word given the stemmed version,
        as stored in the word lookup.
        """
        if word in cls.word_lookup:
            return max(cls.word_lookup[word].keys(),
                       key=lambda x: cls.word_lookup[word][x])
        else:
            return word


class VocabBuilder(object):
    UNK = '<UNK>'

    def __init__(self, text_steam, size=None, min_frequency=1):
        if all([size, min_frequency]) or not any([size, min_frequency]):
            raise ValueError('`size` or `min_size` is required.')

        self.words = self.get_words(text_steam)
        self.total_count = len(self.words)
        self.size = size
        self.min_frequency = min_frequency
        self.counter = self.count_words()
        self.word2idx = self.get_word2idx()
        self.data = self.get_data()
        # update the UKN count
        self.counter[0][1] = self.data.count(0)
        self.idx2word = self.get_idx2word()

    @staticmethod
    @time_
    def get_words(text_stream):
        return [StemmingLookup.stem(word.lower()) for line in text_stream for word in word_tokenize(line)
                if word not in string.punctuation]

    @time_
    def count_words(self):
        counter = [[self.UNK, 0]]

        if self.min_frequency:
            counter.extend([list(item) for item in Counter(self.words).most_common()
                            if item[1] > self.min_frequency])
            self.size = len(counter)
        else:
            counter.extend(Counter(self.words).most_common(self.size - 1))
            self.min_frequency = min(counter.values())

        return counter

    @time_
    def get_word2idx(self):
        return {word: i for i, (word, _) in enumerate(self.counter)}

    @time_
    def get_data(self):
        return [self.word2idx.get(word, 0) for word in self.words]

    @time_
    def get_idx2word(self):
        return dict(zip(self.word2idx.values(), self.word2idx.keys()))

    def info(self):
        print(self.min_frequency, self.size)

    def get_decay(self, min_learning_rate, learning_rate, window):
        return (min_learning_rate - learning_rate) / (self.total_count * window)
