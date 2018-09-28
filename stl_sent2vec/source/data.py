###############################################################################
# Author: Wasi Ahmad
# Project: Sentence pair classification
# Date Created: 7/25/2017
#
# File Description: This script contains code to read and parse input files.
###############################################################################

import os, helper


class Dictionary(object):
    """Dictionary class that stores all words of train/dev corpus."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.pad_token = '<p>'
        self.idx2word.append(self.pad_token)
        self.word2idx[self.pad_token] = len(self.idx2word) - 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Instance(object):
    """Instance that represent a sample of train/dev/test corpus."""

    def __init__(self, pairid=None):
        self.sentence1 = []
        self.sentence2 = []
        self.label = -1
        self.id = pairid if pairid else -1

    def add_sentence(self, sentence, tokenize, sentence_no, dictionary, is_test_instance):
        words = ['<s>'] + helper.tokenize(sentence, tokenize) + ['</s>']
        if not is_test_instance:
            for word in words:
                dictionary.add_word(word)
        if sentence_no == 1:
            self.sentence1 = words
        else:
            self.sentence2 = words

    def add_label(self, label):
        self.label = label


class Corpus(object):
    """Corpus class which contains all information about train/dev/test corpus."""

    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.data = []

    def parse(self, task_name, path, filename, tokenize, num_examples=-1,
              is_test_corpus=False, skip_first_line=False):
        """Parses the content of a file."""

        path = path + task_name + '/'
        assert os.path.exists(os.path.join(path, filename))
        label2id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
            for line in f:
                if skip_first_line:
                    skip_first_line = False
                    continue

                tokens = line.strip().split('\t')
                if task_name == 'sick':
                    pairid = tokens[0]
                    sent1, sent2 = tokens[1], tokens[2]
                    label = tokens[4].lower()
                elif task_name == 'sst':
                    pairid = None
                    sent1, sent2 = tokens[0], tokens[0]
                    label = tokens[1]
                else:
                    pairid = tokens[3] if len(tokens) >= 4 else None
                    sent1, sent2 = tokens[0], tokens[1]
                    label = tokens[2]

                instance = Instance(pairid)
                instance.add_sentence(sent1, tokenize, 1, self.dictionary, is_test_corpus)
                instance.add_sentence(sent2, tokenize, 2, self.dictionary, is_test_corpus)
                if label.isdigit():
                    instance.add_label(int(label))
                else:
                    instance.add_label(label2id[label])

                self.data.append(instance)
                if len(self.data) == num_examples:
                    break
