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

from cove import CoVe

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_W2V = '../../glove/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2
MODEL_PATH = '../data/models/cove/wmtlstm.pth'
V = 1  # version of InferSent

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    params['cove'].build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params['cove'].encode(
        sentences, params.batch_size, tokenize=False
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
    # Load InferSent model
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 300, 'nlayers': 2,
                    'pool_type': 'mean', 'dpout_model': 0.0, 'version': V}

    model = CoVe(params_model)
    model.load_state(MODEL_PATH)
    model.set_w2v_path(PATH_TO_W2V)

    params_senteval['cove'] = model.cuda()
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
