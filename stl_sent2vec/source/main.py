###############################################################################
# Author: Wasi Ahmad
# Project: Sentence pair classification
# Date Created: 7/25/2017
#
# File Description: This is the main script from where all experimental
# execution begins.
###############################################################################

import util
import data
import helper
import torch
import os
import numpy
from classifier import SentenceClassifier
from train import Train
from test import evaluate

args = util.get_args()
# if output directory doesn't exist, create it
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


def main():
    ###############################################################################
    # Load data
    ###############################################################################

    dictionary = data.Dictionary()
    train_corpus = data.Corpus(dictionary)
    dev_corpus = data.Corpus(dictionary)
    test_corpus = data.Corpus(dictionary)

    task_names = ['snli', 'multinli'] if args.task == 'allnli' else [args.task]
    for task in task_names:
        skip_first_line = True if task == 'sick' else False
        train_corpus.parse(task, args.data, 'train.txt',
                           args.tokenize, num_examples=args.max_example,
                           skip_first_line=skip_first_line)
        if task == 'multinli':
            dev_corpus.parse(task, args.data, 'dev_matched.txt', args.tokenize)
            dev_corpus.parse(task, args.data, 'dev_mismatched.txt', args.tokenize)
            test_corpus.parse(task, args.data, 'test_matched.txt',
                              args.tokenize, is_test_corpus=False)
            test_corpus.parse(task, args.data, 'test_mismatched.txt',
                              args.tokenize, is_test_corpus=False)
        else:
            dev_corpus.parse(task, args.data, 'dev.txt', args.tokenize,
                             skip_first_line=skip_first_line)
            test_corpus.parse(task, args.data, 'test.txt', args.tokenize,
                              is_test_corpus=False, skip_first_line=skip_first_line)

    print('train set size = ', len(train_corpus.data))
    print('development set size = ', len(dev_corpus.data))
    print('test set size = ', len(test_corpus.data))
    print('vocabulary size = ', len(dictionary))

    # save the dictionary object to use during testing
    helper.save_object(dictionary, args.save_path + args.task + '_dictionary.pkl')

    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                   dictionary.word2idx)
    print('number of OOV words = ', len(dictionary) - len(embeddings_index))

    # ###############################################################################
    # # Build the model
    # ###############################################################################

    model = SentenceClassifier(dictionary, embeddings_index, args)
    optim_fn, optim_params = helper.get_optimizer(args.optimizer)
    optimizer = optim_fn(filter(lambda p: p.requires_grad, model.parameters()), **optim_params)
    best_acc = 0

    if args.cuda:
        model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # ###############################################################################
    # # Train the model
    # ###############################################################################

    train = Train(model, optimizer, dictionary, embeddings_index, args, best_acc)
    bestmodel = train.train_epochs(train_corpus, dev_corpus, args.start_epoch, args.epochs)
    test_batches = helper.batchify(test_corpus.data, args.batch_size)
    if 'multinli' in task_names:
        print('Skipping evaluating best model. Evaluate using the test script.')
    else:
        test_accuracy, test_f1 = evaluate(bestmodel, test_batches, dictionary)
        print('accuracy: %.2f%%' % test_accuracy)
        print('f1: %.2f%%' % test_f1)


if __name__ == '__main__':
    main()
