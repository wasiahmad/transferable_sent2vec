###############################################################################
# Author: Wasi Ahmad
# Project: Multitask domain adaptation for text classification
# Date Created: 9/23/2017
#
# File Description: This script contains code to train the model.
###############################################################################

import time, helper, numpy, torch, sys
import torch.nn as nn
from torch.nn.utils import clip_grad_norm


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, optimizer, dictionary, embeddings_index, config, best_accuracy):
        self.train_corpus = None
        self.dev_corpus = None

        self.model = model
        self.dictionary = dictionary
        self.embeddings_index = embeddings_index
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.size_average = self.config.average_loss
        if self.config.cuda:
            self.criterion = self.criterion.cuda()

        self.optimizer = optimizer
        self.best_dev_acc = best_accuracy
        self.times_no_improvement = 0
        self.stop = False
        self.train_accuracies = []
        self.dev_accuracies = []

    def set_train_dev_corpus(self, train_corpus, dev_corpus):
        self.train_corpus = train_corpus
        self.dev_corpus = dev_corpus

    def train_epochs(self, start_epoch, n_epochs):
        """Trains model for n_epochs epochs"""
        for epoch in range(start_epoch, start_epoch + n_epochs):
            if not self.stop:
                print('\nTRAINING : Epoch ' + str((epoch + 1)))
                self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr'] * self.config.lr_decay \
                    if (epoch + 1) > 1 and 'sgd' in self.config.optimizer else self.optimizer.param_groups[0]['lr']
                print('Learning rate : {0}'.format(self.optimizer.param_groups[0]['lr']))
                self.train()
                # training epoch completes, now do validation
                print('\nVALIDATING : Epoch ' + str((epoch + 1)))
                dev_acc = self.validate()
                self.dev_accuracies.append(dev_acc)
                print('validation accuracy = %.2f' % dev_acc)
                # save model if dev loss goes down
                if self.best_dev_acc < dev_acc:
                    self.best_dev_acc = dev_acc
                    helper.save_checkpoint({
                        'epoch': (epoch + 1),
                        'state_dict': helper.get_state_dict(self.model, self.config),
                        'best_acc': self.best_dev_acc,
                        'optimizer': self.optimizer.state_dict(),
                    }, self.config.save_path + 'model_best.pth')
                    self.times_no_improvement = 0
                else:
                    if 'sgd' in self.config.optimizer:
                        self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0][
                                                                   'lr'] / self.config.lrshrink
                        print('Shrinking lr by : {0}. New lr = {1}'.format(self.config.lrshrink,
                                                                           self.optimizer.param_groups[0]['lr']))
                        if self.optimizer.param_groups[0]['lr'] < self.config.minlr:
                            self.stop = True
                    if 'adam' in self.config.optimizer:
                        self.times_no_improvement += 1
                        # early stopping (at 3rd decrease in accuracy)
                        if self.times_no_improvement >= 5:
                            self.stop = True
            else:
                break

    def train(self):
        # Turn on training mode which enables dropout.
        self.model.train()

        # Splitting the data in batches
        batches, batch_labels = [], []
        for task_name, task in self.train_corpus.items():
            train_batches = helper.batchify(task.data, self.config.batch_size)
            batches.extend(train_batches)
            batch_labels.extend([task_name] * len(train_batches))

        combined = list(zip(batches, batch_labels))
        numpy.random.shuffle(combined)
        batches[:], batch_labels[:] = zip(*combined)
        print('number of train batches = ', len(batches))

        start = time.time()
        print_acc_total = 0
        plot_acc_total = 0
        num_back = 0

        num_batches = len(batches)
        for batch_no in range(1, num_batches + 1):
            # Clearing out all previous gradient computations.
            self.optimizer.zero_grad()
            if self.config.use_elmo:
                train_sentences1, sent_len1, train_sentences2, sent_len2, train_labels = helper.batch_to_elmo_tensors(
                    batches[batch_no - 1], self.dictionary)
            else:
                train_sentences1, sent_len1, train_sentences2, sent_len2, train_labels = helper.batch_to_tensors(
                    batches[batch_no - 1], self.dictionary)

            if self.config.cuda:
                train_sentences1 = train_sentences1.cuda()
                train_sentences2 = train_sentences2.cuda()
                train_labels = train_labels.cuda()
            assert train_sentences1.size(0) == train_sentences2.size(0)

            score = self.model(train_sentences1, sent_len1, train_sentences2, sent_len2, batch_labels[batch_no - 1])
            n_correct = (torch.max(score, 1)[1].view(train_labels.size()).data == train_labels.data).sum()
            loss = self.criterion(score, train_labels)

            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            clip_grad_norm(filter(lambda p: p.requires_grad, self.model.parameters()), self.config.max_norm)
            self.optimizer.step()

            print_acc_total += 100. * n_correct / len(batches[batch_no - 1])
            plot_acc_total += 100. * n_correct / len(batches[batch_no - 1])

            if batch_no % self.config.print_every == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = '%s (%d %d%%) %.2f' % (helper.show_progress(start, batch_no / num_batches), batch_no,
                                                  batch_no / num_batches * 100, print_acc_total / batch_no)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

                # if batch_no % self.config.plot_every == 0:
                #     plot_acc_avg = plot_acc_total / self.config.plot_every
                #     self.train_accuracies.append(plot_acc_avg)
                #     plot_acc_total = 0

    def validate(self):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        # Splitting the data in batches
        batches, batch_labels = [], []
        for task_name, task in self.dev_corpus.items():
            dev_batches = helper.batchify(task.data, self.config.batch_size)
            batches.extend(dev_batches)
            batch_labels.extend([task_name] * len(dev_batches))

        combined = list(zip(batches, batch_labels))
        numpy.random.shuffle(combined)
        batches[:], batch_labels[:] = zip(*combined)
        print('number of dev batches = ', len(batches))

        num_batches = len(batches)
        n_correct, n_total = 0, 0
        with torch.no_grad():
            for batch_no in range(1, num_batches + 1):
                if self.config.use_elmo:
                    dev_sentences1, sent_len1, dev_sentences2, sent_len2, dev_labels = helper.batch_to_elmo_tensors(
                        batches[batch_no - 1], self.dictionary)
                else:
                    dev_sentences1, sent_len1, dev_sentences2, sent_len2, dev_labels = helper.batch_to_tensors(
                        batches[batch_no - 1], self.dictionary, iseval=True)

                if self.config.cuda:
                    dev_sentences1 = dev_sentences1.cuda()
                    dev_sentences2 = dev_sentences2.cuda()
                    dev_labels = dev_labels.cuda()
                assert dev_sentences1.size(0) == dev_sentences2.size(0)

                score = self.model(dev_sentences1, sent_len1, dev_sentences2, sent_len2, batch_labels[batch_no - 1])
                n_correct += (torch.max(score, 1)[1].view(dev_labels.size()).data == dev_labels.data).sum()
                n_total += len(batches[batch_no - 1])

        return 100. * n_correct / n_total
