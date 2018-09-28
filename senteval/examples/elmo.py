import numpy as np
import time

import torch
from torch.autograd import Variable
import torch.nn as nn

from allennlp.modules.elmo import _ElmoBiLm
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper


class ELMo(nn.Module):
    def __init__(self, config):
        super(ELMo, self).__init__()
        self.bsize = config['bsize']
        self.pool_type = config['pool_type']
        self.which_layer = config['which_layer']
        self.version = 1 if 'version' not in config else config['version']
        self.elmo_embedder = _ElmoBiLm(config['optfile'],
                                       config['wgtfile'],
                                       requires_grad=False)

        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = '<p>'
            self.eos = '</p>'
            self.max_pad = False
            self.moses_tok = True

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: Variable(seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        sent_output = self.elmo_embedder(sent)['activations']
        if self.which_layer == 'top':
            sent_output = sent_output[-1]
        elif self.which_layer == 'all':
            sent_output = torch.cat(sent_output, dim=2)
        else:
            assert False

        sent_output = sent_output.transpose(0, 1)

        # Pooling
        if self.pool_type == "mean":
            sent_len = Variable(torch.FloatTensor(sent_len.copy())).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch), len(batch[0]), 50))
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[i, j, :] = ELMoCharacterMapper.convert_word_to_char_ids(batch[i][j])

        return torch.LongTensor(embed)

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [s.split() if not tokenize else
                     self.tokenize(s) for s in sentences]

        lengths = np.array([len(s) + 2 for s in sentences])
        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
            sentences, bsize, tokenize, verbose)

        embeddings = []
        with torch.no_grad():
            for stidx in range(0, len(sentences), bsize):
                batch = self.get_batch(sentences[stidx:stidx + bsize])
                batch = batch.cuda()
                batch = self.forward(
                    (batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
                embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                len(embeddings) / (time.time() - tic),
                'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings
