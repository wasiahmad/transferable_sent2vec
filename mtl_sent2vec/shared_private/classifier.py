###############################################################################
# Author: Wasi Ahmad
# Project: Multitask learning for text classification
# Date Created: 10/18/2017
#
# File Description: This script contains code related to multitask text
# classifier.
###############################################################################

import torch, helper
import torch.nn as nn
import torch.nn.functional as f
from collections import OrderedDict
from allennlp.modules.elmo import _ElmoBiLm
from nn_layer import EmbeddingLayer, Encoder


# model details can be found at http://www.aclweb.org/anthology/P17-1001
class Generator(nn.Module):
    """Adversarial Multi-task Learning Classifier."""

    def __init__(self, dictionary, embeddings_index, args, tasks):
        """"Constructor of the class."""
        super(Generator, self).__init__()
        self.config = args
        self.tasks = tasks
        self.task_idx = {}
        for task_name, num_class in self.tasks:
            self.task_idx[task_name] = len(self.task_idx)
        self.num_directions = 2 if args.bidirection else 1

        if self.config.use_elmo:
            self.embedding = ELMo(args.emsize, args.optfile, args.wgtfile, args.dropout)
        else:
            self.embedding = EmbeddingLayer(len(dictionary), self.config)
            self.embedding.init_embedding_weights(dictionary, embeddings_index, self.config.emsize)

        encoder = []
        for i in range(len(self.tasks)):
            enc = Encoder(self.config.emsize, self.config.nhid, self.config.bidirection, self.config)
            encoder.append(enc)
        self.encoder = nn.ModuleList(encoder)
        self.generator = Encoder(self.config.emsize, self.config.nhid, self.config.bidirection, self.config)

        if self.config.pool_type == 'attn':
            private_attn = []
            for i in range(len(self.tasks)):
                private_attn.append(nn.Linear(self.config.nhid * self.num_directions, 1))
            self.private_attn_layer = nn.ModuleList(private_attn)
            self.shared_attn_layer = nn.Linear(self.config.nhid * self.num_directions, 1)

        feedffn = []
        for task_name, num_class in self.tasks:
            if self.config.nonlinear_fc:
                ffnn = nn.Sequential(OrderedDict([
                    ('dropout1', nn.Dropout(self.config.dropout_fc)),
                    ('dense1', nn.Linear(self.config.nhid * self.num_directions * 8, self.config.fc_dim)),
                    ('tanh1', nn.Tanh()),
                    ('dropout2', nn.Dropout(self.config.dropout_fc)),
                    ('dense2', nn.Linear(self.config.fc_dim, self.config.fc_dim)),
                    ('tanh2', nn.Tanh()),
                    ('dropout3', nn.Dropout(self.config.dropout_fc)),
                    ('dense3', nn.Linear(self.config.fc_dim, num_class))
                ]))
            else:
                ffnn = nn.Sequential(OrderedDict([
                    ('dropout1', nn.Dropout(self.config.dropout_fc)),
                    ('dense1', nn.Linear(self.config.nhid * self.num_directions * 8, self.config.fc_dim)),
                    ('dropout2', nn.Dropout(self.config.dropout_fc)),
                    ('dense2', nn.Linear(self.config.fc_dim, self.config.fc_dim)),
                    ('dropout3', nn.Dropout(self.config.dropout_fc)),
                    ('dense3', nn.Linear(self.config.fc_dim, num_class))
                ]))
            feedffn.append(ffnn)
        self.ffnn = nn.ModuleList(feedffn)

    def forward(self, batch_sentence1, sent_len1, batch_sentence2, sent_len2, task_label):
        """"Defines the forward computation of the sentence pair classifier."""
        embedded1 = self.embedding(batch_sentence1)
        embedded2 = self.embedding(batch_sentence2)

        # run task-specific rnn to get private hidden states
        private_hidden_states1 = self.encoder[self.task_idx[task_label]](embedded1, sent_len1)  # B X |S1| X D/2D
        private_hidden_states2 = self.encoder[self.task_idx[task_label]](embedded2, sent_len2)  # B X |S2| X D/2D

        if self.config.pool_type == 'max':
            # applying max-pooling over private representations
            private_sent_rep1 = torch.max(private_hidden_states1, 1)[0].squeeze()  # B X D/2D
            private_sent_rep2 = torch.max(private_hidden_states2, 1)[0].squeeze()  # B X D/2D
        elif self.config.pool_type == 'mean':
            # applying mean-pooling over shared representations
            private_sent_rep1 = torch.mean(private_hidden_states1, 1)  # B X D/2D
            private_sent_rep2 = torch.mean(private_hidden_states2, 1)  # B X D/2D
        elif self.config.pool_type == 'attn':
            # applying weighted-pooling over shared representations
            att_weights_s1 = self.private_attn_layer[self.task_idx[task_label]](
                private_hidden_states1.view(-1, private_hidden_states1.size(2)))
            att_weights_s1 = f.softmax(att_weights_s1.view(*private_hidden_states1.size()[:-1]), 1)
            private_sent_rep1 = torch.bmm(private_hidden_states1.transpose(1, 2), att_weights_s1.unsqueeze(2)).squeeze(
                2)
            att_weights_s2 = self.private_attn_layer[self.task_idx[task_label]](
                private_hidden_states2.view(-1, private_hidden_states2.size(2)))
            att_weights_s2 = f.softmax(att_weights_s2.view(*private_hidden_states2.size()[:-1]), 1)
            private_sent_rep2 = torch.bmm(private_hidden_states2.transpose(1, 2), att_weights_s2.unsqueeze(2)).squeeze(
                2)

        # run shared rnn to get hidden states
        shared_hidden_states1 = self.generator(embedded1, sent_len1)  # B X |S1| X D/2D
        shared_hidden_states2 = self.generator(embedded2, sent_len2)  # B X |S2| X D/2D

        if self.config.pool_type == 'max':
            # applying max-pooling over shared representations
            shared_sent_rep1 = torch.max(shared_hidden_states1, 1)[0].squeeze()  # B X D/2D
            shared_sent_rep2 = torch.max(shared_hidden_states2, 1)[0].squeeze()  # B X D/2D
        elif self.config.pool_type == 'mean':
            # applying mean-pooling over shared representations
            shared_sent_rep1 = torch.mean(shared_hidden_states1, 1)  # B X D/2D
            shared_sent_rep2 = torch.mean(shared_hidden_states2, 1)  # B X D/2D
        elif self.config.pool_type == 'attn':
            # applying weighted-pooling over shared representations
            att_weights_s1 = self.shared_attn_layer(shared_hidden_states1.view(-1, shared_hidden_states1.size(2)))
            att_weights_s1 = f.softmax(att_weights_s1.view(*shared_hidden_states1.size()[:-1]), 1)
            shared_sent_rep1 = torch.bmm(shared_hidden_states1.transpose(1, 2), att_weights_s1.unsqueeze(2)).squeeze(2)
            att_weights_s2 = self.shared_attn_layer(shared_hidden_states2.view(-1, shared_hidden_states2.size(2)))
            att_weights_s2 = f.softmax(att_weights_s2.view(*shared_hidden_states2.size()[:-1]), 1)
            shared_sent_rep2 = torch.bmm(shared_hidden_states2.transpose(1, 2), att_weights_s2.unsqueeze(2)).squeeze(2)

        if self.config.adversarial:
            sent1_ = torch.bmm(private_hidden_states1, torch.transpose(shared_hidden_states1, 1, 2))
            sent2_ = torch.bmm(private_hidden_states2, torch.transpose(shared_hidden_states2, 1, 2))
            norm1 = torch.sum(sent1_ * sent1_) / shared_hidden_states1.size(0)
            norm2 = torch.sum(sent2_ * sent2_) / shared_hidden_states2.size(0)
            diff_loss = norm1 + norm2

        # concatenate private and shared sentence representations
        combined_sent_rep1 = torch.cat((private_sent_rep1, shared_sent_rep1), 1)  # B X 2D/4D
        combined_sent_rep2 = torch.cat((private_sent_rep2, shared_sent_rep2), 1)  # B X 2D/4D

        # compute angle between sentence representations
        angle = torch.mul(combined_sent_rep1, combined_sent_rep2)
        # compute distance between sentence representations
        distance = torch.abs(combined_sent_rep1 - combined_sent_rep2)
        # combined_representation = batch_size x (hidden_size * num_directions * 8)
        combined_representation = torch.cat((combined_sent_rep1, combined_sent_rep2, angle, distance), 1)

        scores = self.ffnn[self.task_idx[task_label]](combined_representation)
        if self.config.adversarial:
            return scores, diff_loss, [shared_sent_rep1, shared_sent_rep2]
        else:
            return scores


class Discriminator(nn.Module):
    """Discriminator class."""

    def __init__(self, args, num_tasks):
        """"Discriminator of the class."""
        super(Discriminator, self).__init__()
        self.num_directions = 2 if args.bidirection else 1
        self.linear = nn.Linear(args.nhid * self.num_directions, args.nhid * self.num_directions)
        self.softmax = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(args.nhid * self.num_directions, num_tasks)),
            ('log_softmax', nn.LogSoftmax(dim=1))
        ]))

    def forward(self, input):
        """"Defines the forward computation of the discriminator."""
        return self.softmax(self.linear(input))


class ELMo(nn.Module):
    def __init__(self, nhid, optfile, wgtfile, dropout):
        super(ELMo, self).__init__()
        self.elmo_embedder = _ElmoBiLm(optfile, wgtfile, requires_grad=False)
        self.weight_param = nn.Parameter(torch.FloatTensor([0.0, 0.0, 0.0]))
        self.relu_network = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(1024, nhid)),
            ('dropout', nn.Dropout(dropout)),
            ('tanh', nn.ReLU())
        ]))

    def forward(self, sent_batch):
        normed_weights = f.softmax(self.weight_param, dim=0)
        # A list of activations at each layer of the network, [shape: (batch_size, timesteps + 2, embedding_dim)]
        embedded_sent = self.elmo_embedder(sent_batch)['activations']
        embedded_sent = torch.stack(embedded_sent, dim=3)  # (batch_size, num_timesteps, 1024, 3)
        emb_sent = embedded_sent.contiguous().view(-1, embedded_sent.size(2), embedded_sent.size(3))
        ext_weights = normed_weights.unsqueeze(0).expand(emb_sent.size(0), emb_sent.size(2))
        emb_sent = torch.bmm(emb_sent, ext_weights.unsqueeze(2)).squeeze(2)  # (batch_size*num_timesteps, 1024)
        # emb_sent = embedded_sent[-1]
        emb_sent = self.relu_network(emb_sent)
        # batch_size, num_timesteps, nhid
        embedded_sent = emb_sent.view(sent_batch.size(0), sent_batch.size(1) + 2, -1)
        return embedded_sent
