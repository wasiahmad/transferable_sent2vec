# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import torch.nn as nn
import numpy as np

# get models.py from InferSent repo
from sent2vec import Sent2VecSingle, Sent2Vec
from gensen import GenSen, GenSenSingle
from elmo import ELMo


class All2Vec(nn.Module):
    """Concat Gensen."""

    def __init__(self, models):
        """A wrapper class for multiple GenSen models."""
        super(All2Vec, self).__init__()
        self.models = models

    def build_vocab(self, sentences, tokenize=True):
        for name, model in self.models.items():
            if name not in ['elmo']:
                model.build_vocab(sentences, tokenize=tokenize)

    def get_representation(
            self, sentences, batch_size,
            tokenize=False, strategy='max'
    ):
        """Get model representations."""
        representations = []
        for name, model in self.models.items():
            if name == 'elmo':
                embeddings = model.encode(
                    sentences, bsize=batch_size,
                    tokenize=tokenize
                )
            elif name == 'gensen':
                sentences = [s.lower() for s in sentences]
                _, embeddings = model.get_representation(
                    sentences, pool=strategy, return_numpy=True
                )
            else:
                embeddings = model.get_representation(
                    sentences, batch_size=batch_size,
                    tokenize=tokenize
                )
            representations.append(embeddings)

        return np.concatenate(representations, axis=1)


# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_W2V = '../../glove/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2

# SENT2VEC = ['../data/models/sent2vec/mtask/adv_shared_private/shared.pth',
#             '../data/models/sent2vec/mtask/adv_shared_private/quora.pth',
#             '../data/models/sent2vec/mtask/adv_shared_private/snli.pth',
#             '../data/models/sent2vec/mtask/adv_shared_private/multinli.pth']

SENT2VEC = ['../data/models/sent2vec/mtask/shared_private/shared.pth',
            '../data/models/sent2vec/mtask/shared_private/quora.pth',
            '../data/models/sent2vec/mtask/shared_private/snli.pth',
            '../data/models/sent2vec/mtask/shared_private/multinli.pth']

ELMO_WEIGHT = '../data/models/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
ELMO_OPTIONS = '../data/models/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'

FOLDER_PATH = '../data/models/gensen/'
PRETRAIN_EMB = '../../glove/glove.840B.300d.h5'
PREFIX1 = 'nli_large_bothskip_parse'
PREFIX2 = 'nli_large_bothskip'

V = 1  # version of InferSent

assert all([os.path.isfile(path) for path in SENT2VEC]), 'Set MODEL PATHs'
assert os.path.isfile(ELMO_WEIGHT), 'Set MODEL PATHs'
assert os.path.isfile(PATH_TO_W2V), 'Set GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def gensen_prepare(params, model, samples):
    print('Preparing task : %s ' % (params.current_task))
    vocab = set()
    for sample in samples:
        if params.current_task != 'TREC':
            sample = ' '.join(sample).lower().split()
        else:
            sample = ' '.join(sample).split()
        for word in sample:
            if word not in vocab:
                vocab.add(word)

    vocab.add('<s>')
    vocab.add('<pad>')
    vocab.add('<unk>')
    vocab.add('</s>')
    # If you want to turn off vocab expansion just comment out the below line.
    model.vocab_expansion(vocab)


def prepare(params, samples):
    for name, model in params['model'].models.items():
        if name == 'gensen':
            gensen_prepare(params, model, samples)
        elif name == 'elmo':
            continue
        else:
            model.build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]

    # batch contains list of words
    max_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'ImageCaptionRetrieval']
    if params.current_task in max_tasks:
        strategy = 'max'
    else:
        strategy = 'last'

    embeddings = params['model'].get_representation(
        sentences, params.batch_size,
        tokenize=False, strategy=strategy
    )
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}

    s2vsingle = [
        Sent2VecSingle(params_model)
        for _ in range(len(SENT2VEC))
    ]

    for i in range(len(SENT2VEC)):
        s2vsingle[i].load_state(SENT2VEC[i])
        s2vsingle[i].set_w2v_path(PATH_TO_W2V)
        s2vsingle[i] = s2vsingle[i].cuda()

    sent2vec = Sent2Vec(s2vsingle, 'concat')

    params_model = {'bsize': 64, 'pool_type': 'mean',
                    'which_layer': 'all',
                    'optfile': ELMO_OPTIONS,
                    'wgtfile': ELMO_WEIGHT}

    elmo = ELMo(params_model)
    elmo = elmo.cuda()

    gensen_1 = GenSenSingle(
        model_folder=FOLDER_PATH,
        filename_prefix=PREFIX1,
        pretrained_emb=PRETRAIN_EMB,
        cuda=True
    )
    gensen_2 = GenSenSingle(
        model_folder=FOLDER_PATH,
        filename_prefix=PREFIX2,
        pretrained_emb=PRETRAIN_EMB,
        cuda=True
    )
    gensen = GenSen(gensen_1, gensen_2)

    models = {
        'sent2vec': sent2vec,
        'elmo': elmo,
        'gensen': gensen
    }

    # models = {
    #     'elmo': elmo,
    #     'sent2vec': sent2vec
    # }

    # models = {
    #     'elmo': elmo,
    #     'gensen': gensen
    # }

    all2vec = All2Vec(models)
    params_senteval['model'] = all2vec.cuda()

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results_transfer = se.eval(transfer_tasks)

    print('--------------------------------------------')
    print('MR                [Dev:%.1f/Test:%.1f]' % (results_transfer['MR']['devacc'], results_transfer['MR']['acc']))
    print('CR                [Dev:%.1f/Test:%.1f]' % (results_transfer['CR']['devacc'], results_transfer['CR']['acc']))
    print('SUBJ              [Dev:%.1f/Test:%.1f]' % (
        results_transfer['SUBJ']['devacc'], results_transfer['SUBJ']['acc']))
    print('MPQA              [Dev:%.1f/Test:%.1f]' % (
        results_transfer['MPQA']['devacc'], results_transfer['MPQA']['acc']))
    print('SST2              [Dev:%.1f/Test:%.1f]' % (
        results_transfer['SST2']['devacc'], results_transfer['SST2']['acc']))
    print('SST5              [Dev:%.1f/Test:%.1f]' % (
        results_transfer['SST5']['devacc'], results_transfer['SST5']['acc']))
    print('TREC              [Dev:%.1f/Test:%.1f]' % (
        results_transfer['TREC']['devacc'], results_transfer['TREC']['acc']))
    print('MRPC              [Dev:%.1f/TestAcc:%.1f/TestF1:%.1f]' % (
        results_transfer['MRPC']['devacc'], results_transfer['MRPC']['acc'], results_transfer['MRPC']['f1']))
    print('SICKRelatedness   [Dev:%.3f/Test:%.3f]' % (
        results_transfer['SICKRelatedness']['devpearson'], results_transfer['SICKRelatedness']['pearson']))
    print('SICKEntailment    [Dev:%.1f/Test:%.1f]' % (
        results_transfer['SICKEntailment']['devacc'], results_transfer['SICKEntailment']['acc']))
    print('--------------------------------------------')
    print('STS12             [Pearson:%.3f/Spearman:%.3f]' % (
        results_transfer['STS12']['all']['pearson']['mean'], results_transfer['STS12']['all']['spearman']['mean']))
    print('STS13             [Pearson:%.3f/Spearman:%.3f]' % (
        results_transfer['STS13']['all']['pearson']['mean'], results_transfer['STS13']['all']['spearman']['mean']))
    print('STS14             [Pearson:%.3f/Spearman:%.3f]' % (
        results_transfer['STS14']['all']['pearson']['mean'], results_transfer['STS14']['all']['spearman']['mean']))
    print('STS15             [Pearson:%.3f/Spearman:%.3f]' % (
        results_transfer['STS15']['all']['pearson']['mean'], results_transfer['STS15']['all']['spearman']['mean']))
    print('STS16             [Pearson:%.3f/Spearman:%.3f]' % (
        results_transfer['STS16']['all']['pearson']['mean'], results_transfer['STS16']['all']['spearman']['mean']))
    print('STSBenchmark      [Dev:%.5f/Pearson:%.5f/Spearman:%.5f]' % (
        results_transfer['STSBenchmark']['devpearson'], results_transfer['STSBenchmark']['pearson'],
        results_transfer['STSBenchmark']['spearman']))
    print('--------------------------------------------')
    print('Length            [Dev:%.2f/Test:%.2f]' % (
        results_transfer['Length']['devacc'], results_transfer['Length']['acc']))
    print('WordContent       [Dev:%.2f/Test:%.2f]' % (
        results_transfer['WordContent']['devacc'], results_transfer['WordContent']['acc']))
    print('Depth             [Dev:%.2f/Test:%.2f]' % (
        results_transfer['Depth']['devacc'], results_transfer['Depth']['acc']))
    print('TopConstituents   [Dev:%.2f/Test:%.2f]' % (
        results_transfer['TopConstituents']['devacc'], results_transfer['TopConstituents']['acc']))
    print('BigramShift       [Dev:%.2f/Test:%.2f]' % (
        results_transfer['BigramShift']['devacc'], results_transfer['BigramShift']['acc']))
    print('Tense             [Dev:%.2f/Test:%.2f]' % (
        results_transfer['Tense']['devacc'], results_transfer['Tense']['acc']))
    print('SubjNumber        [Dev:%.2f/Test:%.2f]' % (
        results_transfer['SubjNumber']['devacc'], results_transfer['SubjNumber']['acc']))
    print('ObjNumber         [Dev:%.2f/Test:%.2f]' % (
        results_transfer['ObjNumber']['devacc'], results_transfer['ObjNumber']['acc']))
    print('OddManOut         [Dev:%.2f/Test:%.2f]' % (
        results_transfer['OddManOut']['devacc'], results_transfer['OddManOut']['acc']))
    print('CoordInversion    [Dev:%.2f/Test:%.2f]' % (
        results_transfer['CoordinationInversion']['devacc'], results_transfer['CoordinationInversion']['acc']))
    print('--------------------------------------------')
