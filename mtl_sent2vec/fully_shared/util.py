###############################################################################
# Author: Wasi Ahmad
# Project: Multitask domain adaptation for text classification
# Date Created: 9/23/2017
#
# File Description: This is the main script from where all experimental
# execution begins.
###############################################################################

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='multitask_domain_adaptation')
    parser.add_argument('--task', nargs='+', required=True,
                        help='list of task names')
    parser.add_argument('--data', type=str, default='../data/',
                        help='location of the training data')
    parser.add_argument('--max_example', type=int, default=-1,
                        help='number of training examples (-1 = all examples)')
    parser.add_argument('--classifier', type=int, default=0,
                        help='classifier choice')
    parser.add_argument('--tokenize', action='store_true',
                        help='tokenize instances using word_tokenize')
    parser.add_argument('--use_elmo', action='store_true',
                        help='use elmo as embedding layer')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_Tanh, RNN_RELU, LSTM, GRU)')
    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1",
                        help="adam or sgd,lr=0.1")
    parser.add_argument("--lrshrink", type=float, default=5,
                        help="shrink factor for sgd")
    parser.add_argument("--minlr", type=float, default=1e-5,
                        help="minimum lr")
    parser.add_argument("--average_loss", action='store_true',
                        help="consider average loss for mini-batches")
    parser.add_argument('--bidirection', action='store_true',
                        help='use bidirectional recurrent unit')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--emtraining', action='store_true',
                        help='train embedding layer')
    parser.add_argument('--nhid', type=int, default=2048,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument("--fc_dim", type=int, default=512,
                        help="number of hidden units per fc layer")
    parser.add_argument('--lr_decay', type=float, default=.99,
                        help='decay ratio for learning rate')
    parser.add_argument("--nonlinear_fc", action='store_true',
                        help="use nonlinear fully connected layers")
    parser.add_argument('--max_norm', type=float, default=5.0,
                        help='max norm for gradient clipping')
    parser.add_argument('--projection', type=str, default='mask',
                        help='type of domain projection')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper limit of epoch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to embedding and RNN layers (0 = no dropout)')
    parser.add_argument('--dropout_fc', type=float, default=0,
                        help='dropout applied to fully connected layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed for reproducibility')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA for computation')
    parser.add_argument('--print_every', type=int, default=20, metavar='N',
                        help='training report interval')
    parser.add_argument('--plot_every', type=int, default=2000,
                        help='plotting interval')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='resume from last checkpoint (default: none)')
    parser.add_argument('--save_path', type=str, default='../output/',
                        help='path to save the final model')
    parser.add_argument('--word_vectors_file', type=str, default='glove.840B.300d.txt',
                        help='GloVe word embedding version')
    parser.add_argument('--word_vectors_directory', type=str, default='../data/glove/',
                        help='Path of GloVe word embeddings')
    parser.add_argument('--optfile', default='./models/elmo/elmo_options.json', type=str,
                        help='options file for ELMo')
    parser.add_argument('--wgtfile', default='./models/elmo/elmo_weights.hdf5', type=str,
                        help='weight file for ELMo')

    args = parser.parse_args()
    return args
