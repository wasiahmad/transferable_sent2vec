###############################################################################
# Author: Wasi Ahmad
# Project: Multitask domain adaptation for text classification
# Date Created: 9/23/2017
#
# File Description: This script provides general purpose utility functions that
# are required at different steps in the experiments.
###############################################################################

import re
import os
import pickle
import string
import math
import time
import torch
import glob
import inspect
import numpy as np
import matplotlib as mpl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

mpl.use('Agg')

from nltk import wordpunct_tokenize, word_tokenize
from torch.autograd import Variable
from torch import optim
from collections import OrderedDict
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper


def load_word_embeddings(directory, file, dictionary):
    embeddings_index = {}
    f = open(os.path.join(directory, file))
    for line in f:
        word, vec = line.split(' ', 1)
        if word in dictionary:
            embeddings_index[word] = np.array(list(map(float, vec.split())))
    f.close()
    return embeddings_index


def save_word_embeddings(directory, file, embeddings_index):
    f = open(os.path.join(directory, file), 'w')
    for word, vec in embeddings_index.items():
        f.write(word + ' ' + ' '.join(str(x) for x in vec) + '\n')
    f.close()


def save_checkpoint(state, filename='./checkpoint.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)


def get_state_dict(model, config):
    state_dict = dict()
    state_dict['model'] = model.state_dict()
    if config.classifier == 1 and config.projection == 'mask':
        domain_mask = {}
        for task_name, mask in model.domain_mask.items():
            domain_mask[task_name] = mask.data
        state_dict['mask'] = domain_mask
    return state_dict


def load_model_states_from_checkpoint(model, config, filename, tag, from_gpu=True):
    """Load model states from a previously saved checkpoint."""
    assert os.path.exists(filename)
    if from_gpu:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    print(checkpoint[tag]['model'].keys())
    model.load_state_dict(checkpoint[tag]['model'])
    if config.classifier == 1 and config.projection == 'mask':
        for task_name in model.domain_mask.keys():
            if task_name in checkpoint[tag]['mask']:
                model.domain_mask[task_name].data = checkpoint[tag]['mask'][task_name]
            elif task_name in ['snli', 'multinli'] and 'allnli' in checkpoint[tag]['mask']:
                model.domain_mask[task_name].data = checkpoint[tag]['mask']['allnli']


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = list(inspect.signature(optim_fn.__init__).parameters.keys())
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


def softmax(input, axis=1):
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size) - 1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])
    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size) - 1)


def load_model_states_without_dataparallel(model, filename, tag):
    """Load a previously saved model states."""
    assert os.path.exists(filename)
    checkpoint = torch.load(filename)
    new_state_dict = OrderedDict()
    for k, v in checkpoint[tag].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def save_object(obj, filename):
    """Save an object into file."""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def load_object(filename):
    """Load object from file."""
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def tokenize_and_normalize(s):
    """Tokenize and normalize string."""
    token_list = []
    tokens = wordpunct_tokenize(s.lower())
    token_list.extend([x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)])
    return token_list


def tokenize(s, tokenize):
    """Tokenize string."""
    if tokenize:
        return word_tokenize(s)
    else:
        return s.split()


def initialize_out_of_vocab_words(dimension, choice='zero'):
    """Returns a vector of size dimension given a specific choice."""
    if choice == 'random':
        """Returns a random vector of size dimension where mean is 0 and standard deviation is 1."""
        return np.random.normal(size=dimension)
    elif choice == 'zero':
        """Returns a vector of zeros of size dimension."""
        return np.zeros(shape=dimension)


def sentence_to_tensor(sentence, max_sent_length, dictionary):
    sen_rep = torch.LongTensor(max_sent_length).zero_()
    for i in range(len(sentence)):
        word = sentence[i]
        if word in dictionary.word2idx:
            sen_rep[i] = dictionary.word2idx[word]
    return sen_rep


def elmo_sent_mapper(sentence, max_length, pad_token):
    word_list = []
    for i in range(max_length):
        word = sentence[i] if i < len(sentence) else pad_token
        word_list.append(ELMoCharacterMapper.convert_word_to_char_ids(word))
    return word_list


def batch_to_elmo_tensors(batch, dictionary, iseval=False):
    np.random.shuffle(batch)
    max_sent_length1, max_sent_length2 = 0, 0
    for item in batch:
        if max_sent_length1 < len(item.sentence1):
            max_sent_length1 = len(item.sentence1)
        if max_sent_length2 < len(item.sentence2):
            max_sent_length2 = len(item.sentence2)

    all_sentences1, all_sentences2 = [], []
    sent_len1 = np.zeros(len(batch), dtype=np.int)
    sent_len2 = np.zeros(len(batch), dtype=np.int)
    labels = torch.LongTensor(len(batch))
    for i in range(len(batch)):
        sent_len1[i], sent_len2[i] = len(batch[i].sentence1), len(batch[i].sentence2)
        all_sentences1.append(elmo_sent_mapper(batch[i].sentence1, max_sent_length1, dictionary.pad_token))
        all_sentences2.append(elmo_sent_mapper(batch[i].sentence2, max_sent_length2, dictionary.pad_token))
        labels[i] = batch[i].label

    all_sentences1 = torch.from_numpy(np.asarray(all_sentences1, dtype=np.int))
    all_sentences2 = torch.from_numpy(np.asarray(all_sentences2, dtype=np.int))

    return all_sentences1, sent_len1, all_sentences2, sent_len2, labels


def batch_to_tensors(batch, dictionary, iseval=False):
    """Convert a list of sequences to a list of tensors."""
    np.random.shuffle(batch)
    max_sent_length1, max_sent_length2 = 0, 0
    for item in batch:
        if max_sent_length1 < len(item.sentence1):
            max_sent_length1 = len(item.sentence1)
        if max_sent_length2 < len(item.sentence2):
            max_sent_length2 = len(item.sentence2)

    all_sentences1 = torch.LongTensor(len(batch), max_sent_length1)
    sent_len1 = np.zeros(len(batch))
    all_sentences2 = torch.LongTensor(len(batch), max_sent_length2)
    sent_len2 = np.zeros(len(batch))
    labels = torch.LongTensor(len(batch))
    for i in range(len(batch)):
        sent_len1[i], sent_len2[i] = len(batch[i].sentence1), len(batch[i].sentence2)
        all_sentences1[i] = sentence_to_tensor(batch[i].sentence1, max_sent_length1, dictionary)
        all_sentences2[i] = sentence_to_tensor(batch[i].sentence2, max_sent_length2, dictionary)
        labels[i] = batch[i].label

    return all_sentences1, sent_len1, all_sentences2, sent_len2, labels


def batchify(data, bsz):
    """Transform data into batches."""
    np.random.shuffle(data)
    batched_data = []
    for i in range(len(data)):
        if i % bsz == 0:
            batched_data.append([data[i]])
        else:
            batched_data[len(batched_data) - 1].append(data[i])
    return batched_data


def save_plot(points, filepath, filetag, epoch):
    """Generate and save the plot"""
    path_prefix = os.path.join(filepath, filetag)
    path = path_prefix + 'epoch_{}.png'.format(epoch)
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.plot(points)
    fig.savefig(path)
    plt.close(fig)  # close the figure
    for f in glob.glob(path_prefix + '*'):
        if f != path:
            os.remove(f)


def convert_to_minutes(s):
    """Converts seconds to minutes and seconds"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def show_progress(since, percent):
    """Prints time elapsed and estimated time remaining given the current time and progress in %"""
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (convert_to_minutes(s), convert_to_minutes(rs))
