###############################################################################
# Author: Wasi Ahmad
# Project: Adversarial Multi-task Learning for Text Classification
# Date Created: 10/18/2017
#
# File Description: This script contains code to train the model.
###############################################################################

import time, helper, numpy, torch, sys
import torch.nn as nn
from torch.nn.utils import clip_grad_norm


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, optimizer, dictionary, embeddings_index, config, best_acc):
        self.train_corpus = None
        self.dev_corpus = None
        self.task_ids = dict()

        self.config = config
        if self.config.adversarial:
            self.generator, self.discriminator = model
            self.optimizerG, self.optimizerD = optimizer
        else:
            self.generator = model
            self.optimizerG = optimizer

        self.dictionary = dictionary
        self.embeddings_index = embeddings_index
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.size_average = self.config.average_loss
        if self.config.cuda:
            self.criterion = self.criterion.cuda()

        self.best_dev_acc = best_acc
        self.times_no_improvement = 0
        self.stop = False
        self.train_accuracies = []
        self.dev_accuracies = []

    def set_train_dev_corpus(self, train_corpus, dev_corpus):
        self.train_corpus = train_corpus
        self.dev_corpus = dev_corpus
        for task_name, task in self.train_corpus.items():
            self.task_ids[task_name] = len(self.task_ids)

    def train_epochs(self, start_epoch, n_epochs):
        """Trains model for n_epochs epochs"""
        for epoch in range(start_epoch, start_epoch + n_epochs):
            if not self.stop:
                print('\nTRAINING : Epoch ' + str((epoch + 1)))
                self.optimizerG.param_groups[0]['lr'] = self.optimizerG.param_groups[0]['lr'] * self.config.lr_decay \
                    if epoch > start_epoch and 'sgd' in self.config.optimizer else self.optimizerG.param_groups[0]['lr']
                if self.config.adversarial:
                    self.optimizerD.param_groups[0]['lr'] = self.optimizerD.param_groups[0]['lr'] * self.config.lr_decay \
                        if (epoch + 1) > 1 and 'sgd' in self.config.optimizer else self.optimizerD.param_groups[0]['lr']
                if 'sgd' in self.config.optimizer:
                    print('Learning rate : {0}'.format(self.optimizerG.param_groups[0]['lr']))
                self.train()
                # training epoch completes, now do validation
                print('\nVALIDATING : Epoch ' + str((epoch + 1)))
                dev_acc = self.validate()
                self.dev_accuracies.append(dev_acc)
                print('validation acc = %.2f' % dev_acc)
                # save model if dev loss goes down
                if self.best_dev_acc < dev_acc:
                    self.best_dev_acc = dev_acc
                    check_point = dict()
                    check_point['epoch'] = (epoch + 1)
                    check_point['state_dict_G'] = self.generator.state_dict()
                    check_point['best_acc'] = self.best_dev_acc
                    check_point['optimizerG'] = self.optimizerG.state_dict()
                    if self.config.adversarial:
                        check_point['state_dict_D'] = self.discriminator.state_dict()
                        check_point['optimizerD'] = self.optimizerD.state_dict()
                    helper.save_checkpoint(check_point, self.config.save_path + 'model_best.pth.tar')
                    self.times_no_improvement = 0
                else:
                    if 'sgd' in self.config.optimizer:
                        self.optimizerG.param_groups[0]['lr'] = self.optimizerG.param_groups[0][
                                                                    'lr'] / self.config.lrshrink
                        if self.config.adversarial:
                            self.optimizerD.param_groups[0]['lr'] = self.optimizerG.param_groups[0]['lr'] / \
                                                                    self.config.lrshrink
                        print('Shrinking lr by : {0}. New lr = {1}'.format(self.config.lrshrink,
                                                                           self.optimizerG.param_groups[0]['lr']))
                        if self.optimizerG.param_groups[0]['lr'] < self.config.minlr:
                            self.stop = True
                    if 'adam' in self.config.optimizer:
                        self.times_no_improvement += 1
                        # early stopping (at 3rd decrease in accuracy)
                        if self.times_no_improvement == 3:
                            self.stop = True
                # save the train and development loss plot
                helper.save_plot(self.train_accuracies, self.config.save_path, 'training_acc_plot_', epoch + 1)
                helper.save_plot(self.dev_accuracies, self.config.save_path, 'dev_acc_plot_', epoch + 1)
            else:
                break

    def train(self):
        # Turn on training mode which enables dropout.
        self.generator.train()

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
        num_back, print_acc_total, plot_acc_total = 0, 0, 0

        num_batches = len(batches)
        for batch_no in range(1, num_batches + 1):
            if self.config.use_elmo:
                train_sentences1, sent_len1, train_sentences2, sent_len2, train_labels = helper.batch_to_elmo_input(
                    batches[batch_no - 1], self.dictionary)
            else:
                train_sentences1, sent_len1, train_sentences2, sent_len2, train_labels = helper.batch_to_tensors(
                    batches[batch_no - 1], self.dictionary)

            if self.config.cuda:
                train_sentences1 = train_sentences1.cuda()
                train_sentences2 = train_sentences2.cuda()
                train_labels = train_labels.cuda()

            assert train_sentences1.size(0) == train_sentences2.size(0)

            if self.config.adversarial:
                self.optimizerD.zero_grad()
                scores, diff_loss, shared_rep = self.generator(train_sentences1, sent_len1, train_sentences2, sent_len2,
                                                               batch_labels[batch_no - 1])
                n_correct = (torch.max(scores, 1)[1].view(train_labels.size()).data == train_labels.data).sum()
                shared_sent_rep1 = shared_rep[0]
                shared_sent_rep2 = shared_rep[1]
                # runt the discriminator to distinguish tasks
                task_prob1 = self.discriminator(shared_sent_rep1.detach())  # B X num_tasks
                task_prob2 = self.discriminator(shared_sent_rep2.detach())  # B X num_tasks
                comb_prob = torch.cat((task_prob1, task_prob2), 0)  # 2B X num_tasks
                task_prob = torch.sum(comb_prob, 0).squeeze()  # size = |num_tasks|
                adv_loss = -1 * task_prob[self.task_ids[batch_labels[batch_no - 1]]]
                adv_loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
                clip_grad_norm(filter(lambda p: p.requires_grad, self.discriminator.parameters()), self.config.max_norm)
                self.optimizerD.step()

                self.optimizerG.zero_grad()
                cross_entropy_loss = self.criterion(scores, train_labels)
                # runt the discriminator to distinguish tasks
                task_prob1 = self.discriminator(shared_sent_rep1)  # B X num_tasks
                task_prob2 = self.discriminator(shared_sent_rep2)  # B X num_tasks
                comb_prob = torch.cat((task_prob1, task_prob2), 0)  # 2B X num_tasks
                task_prob = torch.sum(comb_prob, 0).squeeze()  # size = |num_tasks|
                adv_loss = -1 * task_prob[self.task_ids[batch_labels[batch_no - 1]]]
                total_loss = cross_entropy_loss + self.config.beta * adv_loss + self.config.gamma * diff_loss
                # Important if we are using nn.DataParallel()
                if total_loss.size(0) > 1:
                    total_loss = total_loss.mean()
                total_loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
                clip_grad_norm(filter(lambda p: p.requires_grad, self.generator.parameters()), self.config.max_norm)
                self.optimizerG.step()
            else:
                self.optimizerG.zero_grad()
                scores = self.generator(train_sentences1, sent_len1, train_sentences2, sent_len2,
                                        batch_labels[batch_no - 1])
                n_correct = (torch.max(scores, 1)[1].view(train_labels.size()).data == train_labels.data).sum()
                loss = self.criterion(scores, train_labels)
                # Important if we are using nn.DataParallel()
                if loss.size(0) > 1:
                    loss = loss.mean()
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
                clip_grad_norm(filter(lambda p: p.requires_grad, self.generator.parameters()), self.config.max_norm)
                self.optimizerG.step()

            print_acc_total += 100. * n_correct / len(batches[batch_no - 1])
            plot_acc_total += 100. * n_correct / len(batches[batch_no - 1])

            if batch_no % self.config.print_every == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = '%s (%d %d%%) %.2f%%' % (helper.show_progress(start, batch_no / num_batches), batch_no,
                                                    batch_no / num_batches * 100, print_acc_total / batch_no)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

            if batch_no % self.config.plot_every == 0:
                plot_acc_avg = plot_acc_total / self.config.plot_every
                self.train_accuracies.append(plot_acc_avg)
                plot_acc_total = 0

            # this releases all cache memory and becomes visible to other applications
            torch.cuda.empty_cache()

    def validate(self):
        # Turn on evaluation mode which disables dropout.
        self.generator.eval()

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
        for batch_no in range(1, num_batches + 1):
            if self.config.use_elmo:
                dev_sentences1, sent_len1, dev_sentences2, sent_len2, dev_labels = helper.batch_to_elmo_input(
                    batches[batch_no - 1], self.dictionary, iseval=True)
            else:
                dev_sentences1, sent_len1, dev_sentences2, sent_len2, dev_labels = helper.batch_to_tensors(
                    batches[batch_no - 1], self.dictionary, iseval=True)

            if self.config.cuda:
                dev_sentences1 = dev_sentences1.cuda()
                dev_sentences2 = dev_sentences2.cuda()
                dev_labels = dev_labels.cuda()

            assert dev_sentences1.size(0) == dev_sentences2.size(0)

            if self.config.adversarial:
                scores, adv_loss, diff_loss = self.generator(dev_sentences1, sent_len1, dev_sentences2, sent_len2,
                                                             batch_labels[batch_no - 1])
            else:
                scores = self.generator(dev_sentences1, sent_len1, dev_sentences2, sent_len2,
                                        batch_labels[batch_no - 1])

            n_correct += (torch.max(scores, 1)[1].view(dev_labels.size()).data == dev_labels.data).sum()
            n_total += len(batches[batch_no - 1])

        return 100. * n_correct / n_total
