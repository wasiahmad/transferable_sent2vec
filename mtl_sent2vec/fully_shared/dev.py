###############################################################################
# Author: Wasi Ahmad
# Project: Multitask domain adaptation for text classification
# Date Created: 9/23/2017
#
# File Description: This script contains code to test the model.
###############################################################################

import util, helper, numpy, data, torch
from classifier import MultitaskDomainAdapter
from sklearn.metrics import classification_report

args = util.get_args()

# set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


def evaluate(model, batches, batch_label, dictionary):
    """Evaluate question classifier model on test data."""
    # Turn on evaluation mode which disables dropout.
    model.eval()

    n_correct, n_total = 0, 0
    y_preds = []
    y_true = []
    for batch_no in range(len(batches)):
        test_sentences1, sent_len1, test_sentences2, sent_len2, test_labels = helper.batch_to_tensors(batches[batch_no], dictionary)
        if args.cuda:
            test_sentences1 = test_sentences1.cuda()
            test_sentences2 = test_sentences2.cuda()
            test_labels = test_labels.cuda()
        assert test_sentences1.size(0) == test_sentences1.size(0)

        softmax_prob = model(test_sentences1, sent_len1, test_sentences2, sent_len2, batch_label)
        preds = torch.max(softmax_prob, 1)[1]
        y_preds.extend(preds.data.cpu().tolist())
        y_true.extend(test_labels.data.cpu().tolist())
        n_correct += (preds.view(test_labels.size()).data == test_labels.data).sum()
        n_total += len(batches[batch_no])

    if batch_label == 'quora':
        target_names = ['non_duplicate', 'duplicate']
    elif batch_label == 'snli' or batch_label == 'multinli':
        target_names = ['entailment', 'neutral', 'contradiction']
    print(classification_report(numpy.asarray(y_true), numpy.asarray(y_preds), target_names=target_names))

    return 100. * n_correct / n_total


if __name__ == "__main__":
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                   dictionary.word2idx)
    tasks = []
    if 'quora' in args.task:
        quora_dev = data.Corpus(args.data + 'quora/', dictionary)
        quora_dev.parse('dev.txt', 'quora', args.tokenize, is_test_corpus=True)
        print('quora dev set size = ', len(quora_dev.data))
        tasks.append(('quora', 2))

    if 'snli' in args.task:
        snli_dev = data.Corpus(args.data + 'snli/', dictionary)
        snli_dev.parse('dev.txt', 'snli', args.tokenize, is_test_corpus=True)
        print('snli dev set size = ', len(snli_dev.data))
        tasks.append(('snli', 3))

    if 'multinli' in args.task:
        # test matched part
        multinli_dev_matched = data.Corpus(args.data + 'multinli/', dictionary)
        multinli_dev_matched.parse('dev_matched.txt', 'multinli', args.tokenize, is_test_corpus=True)
        print('mutinli dev[matched] set size = ', len(multinli_dev_matched.data))
        # test mismatched part
        multinli_dev_mismatched = data.Corpus(args.data + 'multinli/', dictionary)
        multinli_dev_mismatched.parse('dev_mismatched.txt', 'multinli', args.tokenize, is_test_corpus=True)
        print('mutinli dev[mismatched] set size = ', len(multinli_dev_mismatched.data))
        tasks.append(('multinli', 3))

    if tasks:
        model = MultitaskDomainAdapter(dictionary, embeddings_index, args, tasks)
        if args.cuda:
            model = model.cuda()
        helper.load_model_states_from_checkpoint(model, args, args.save_path + 'model_best.pth.tar', 'state_dict', args.cuda)
        print('vocabulary size = ', len(dictionary))

        if 'quora' in args.task:
            dev_batches = helper.batchify(quora_dev.data, args.batch_size)
            dev_accuracy = evaluate(model, dev_batches, 'quora', dictionary)
            print('quora dev accuracy: %f%%' % dev_accuracy)

        if 'snli' in args.task:
            dev_batches = helper.batchify(snli_dev.data, args.batch_size)
            dev_accuracy = evaluate(model, dev_batches, 'snli', dictionary)
            print('snli dev accuracy: %f%%' % dev_accuracy)

        if 'multinli' in args.task:
            # test matched part
            dev_batches = helper.batchify(multinli_dev_matched.data, args.batch_size)
            dev_accuracy = evaluate(model, dev_batches, 'multinli', dictionary)
            print('mutinli [matched] dev accuracy: %f%%' % dev_accuracy)
            # test mismatched part
            dev_batches = helper.batchify(multinli_dev_mismatched.data, args.batch_size)
            dev_accuracy = evaluate(model, dev_batches, 'multinli', dictionary)
            print('mutinli [mismatched] dev accuracy: %f%%' % dev_accuracy)
