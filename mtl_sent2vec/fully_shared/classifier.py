###############################################################################
# Author: Wasi Ahmad
# Project: Multitask learning for text classification
# Date Created: 9/23/2017
#
# File Description: This script contains code related to quora duplicate
# question classifier.
###############################################################################

import torch, helper
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict
from torch.autograd import Variable
from nn_layer import EmbeddingLayer, Encoder
from allennlp.modules.elmo import _ElmoBiLm


# model details can be found at http://aclweb.org/anthology/W17-2612
class MultitaskDomainAdapter(nn.Module):
    """Class that classifies sentence pair into desired class labels."""

    def __init__(self, dictionary, embeddings_index, args, tasks):
        """"Constructor of the class."""
        super(MultitaskDomainAdapter, self).__init__()
        self.config = args
        self.tasks = tasks
        self.task_idx = {}
        for task_name, num_class in self.tasks:
            self.task_idx[task_name] = len(self.task_idx)
        self.num_directions = 2 if args.bidirection else 1

        if self.config.use_elmo:
            self.elmo_embedder = _ElmoBiLm(self.config.optfile,
                                           self.config.wgtfile,
                                           requires_grad=False)
            self.relu_network = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(1024, self.config.emsize)),
                ('dropout', nn.Dropout(p=self.config.dropout)),
                ('tanh', nn.ReLU())
            ]))
        else:
            self.embedding = EmbeddingLayer(len(dictionary), self.config)
            self.embedding.init_embedding_weights(dictionary, embeddings_index, self.config.emsize)

        self.encoder = Encoder(self.config.emsize, self.config.nhid, self.config.bidirection, self.config)

        if self.config.classifier == 1:
            if self.config.projection == 'linear':
                self.domain_projecter = nn.Linear(self.config.nhid * self.num_directions,
                                                  self.config.nhid * self.num_directions)
            elif self.config.projection == 'mask':
                self.domain_mask = {}
                domain_mask_size = self.config.nhid // (len(self.tasks) + 1)
                shared_region_size = self.config.nhid - len(self.tasks) * domain_mask_size
                i = 0
                for task_name, num_class in self.tasks:
                    start = shared_region_size + i * domain_mask_size
                    end = shared_region_size + (i + 1) * domain_mask_size
                    mask = torch.ones(self.config.nhid)
                    mask[start:end] = 0
                    if self.num_directions == 2:
                        mask = torch.cat((mask, mask), 0)
                    self.domain_mask[task_name] = mask
                    i += 1
                # create the domain mask variables and convert them to cuda [if needed]
                for task_name, var in self.domain_mask.items():
                    self.domain_mask[task_name] = Variable(var, requires_grad=False)
                    if self.config.cuda:
                        self.domain_mask[task_name] = self.domain_mask[task_name].cuda()
            else:
                raise ValueError('unknown choice!')

        feedfnn = []
        for task_name, num_class in self.tasks:
            if self.config.nonlinear_fc:
                ffnn = nn.Sequential(OrderedDict([
                    ('dropout1', nn.Dropout(self.config.dropout_fc)),
                    ('dense1', nn.Linear(self.config.nhid * self.num_directions * 4, self.config.fc_dim)),
                    ('tanh', nn.Tanh()),
                    ('dropout2', nn.Dropout(self.config.dropout_fc)),
                    ('dense2', nn.Linear(self.config.fc_dim, self.config.fc_dim)),
                    ('tanh', nn.Tanh()),
                    ('dropout3', nn.Dropout(self.config.dropout_fc)),
                    ('dense3', nn.Linear(self.config.fc_dim, num_class))
                ]))
            else:
                ffnn = nn.Sequential(OrderedDict([
                    ('dropout1', nn.Dropout(self.config.dropout_fc)),
                    ('dense1', nn.Linear(self.config.nhid * self.num_directions * 4, self.config.fc_dim)),
                    ('dropout2', nn.Dropout(self.config.dropout_fc)),
                    ('dense2', nn.Linear(self.config.fc_dim, self.config.fc_dim)),
                    ('dropout3', nn.Dropout(self.config.dropout_fc)),
                    ('dense3', nn.Linear(self.config.fc_dim, num_class))
                ]))
            feedfnn.append(ffnn)
        self.ffnn = nn.ModuleList(feedfnn)

    def forward(self, batch_sentence1, sent_len1, batch_sentence2, sent_len2, task_label):
        """"Defines the forward computation of the sentence pair classifier."""
        if self.config.use_elmo:
            embedded1 = self.elmo_embedder(batch_sentence1)['activations'][-1]
            embedded1 = self.relu_network(embedded1)
            embedded2 = self.elmo_embedder(batch_sentence2)['activations'][-1]
            embedded2 = self.relu_network(embedded2)
        else:
            embedded1 = self.embedding(batch_sentence1)
            embedded2 = self.embedding(batch_sentence2)

        # For the first sentences in batch
        output1 = self.encoder(embedded1, sent_len1)
        # For the second sentences in batch
        output2 = self.encoder(embedded2, sent_len2)

        # applying max-pooling to construct sentence representation
        encoded_sentence1 = torch.max(output1, 1)[0].squeeze()
        encoded_sentence2 = torch.max(output2, 1)[0].squeeze()

        assert encoded_sentence1.size(0) == encoded_sentence2.size(0)

        if self.config.classifier == 1:
            encoded_sentence1 = self.apply_domain_projection(encoded_sentence1, task_label)
            encoded_sentence2 = self.apply_domain_projection(encoded_sentence2, task_label)

        # compute angle between sentence representations
        angle = torch.mul(encoded_sentence1, encoded_sentence2)
        # compute distance between sentence representations
        distance = torch.abs(encoded_sentence1 - encoded_sentence2)
        # combined_representation = batch_size x (hidden_size * num_directions * 4)
        combined_representation = torch.cat((encoded_sentence1, encoded_sentence2, angle, distance), 1)

        return self.ffnn[self.task_idx[task_label]](combined_representation)

    def apply_domain_projection(self, encoded_sentence, task_label):
        # linear projection [applying domain projection for sentence representations]
        if self.config.projection == 'linear':
            return self.domain_projecter(encoded_sentence)
        # domain masking [applying domain projection for sentence representations]
        elif self.config.projection == 'mask':
            mask = self.domain_mask[task_label]
            mask = mask.unsqueeze(0).expand(encoded_sentence.size(0), mask.size(0))
            # apply domain masking
            return torch.mul(encoded_sentence, mask)
