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


def evaluate(model, batches, batch_label, dictionary, outfile=None):
    """Evaluate question classifier model on test data."""
    # Turn on evaluation mode which disables dropout.
    model.eval()

    n_correct, n_total = 0, 0
    y_preds = []
    y_true = []
    output = []
    with torch.no_grad():
        for batch_no in range(len(batches)):
            if args.use_elmo:
                test_sentences1, sent_len1, test_sentences2, sent_len2, test_labels = helper.batch_to_elmo_tensors(
                    batches[batch_no], dictionary, iseval=True)
            else:
                test_sentences1, sent_len1, test_sentences2, sent_len2, test_labels = helper.batch_to_tensors(
                    batches[batch_no], dictionary, iseval=True)
                if args.cuda:
                    test_sentences1 = test_sentences1.cuda()
                    test_sentences2 = test_sentences2.cuda()
                    test_labels = test_labels.cuda()
                assert test_sentences1.size(0) == test_sentences1.size(0)

            softmax_prob = model(test_sentences1, sent_len1, test_sentences2, sent_len2, batch_label)
            preds = torch.max(softmax_prob, 1)[1]
            y_preds.extend(preds.data.cpu().tolist())
            if not outfile:
                y_true.extend(test_labels.data.cpu().tolist())
                n_correct += (preds.view(test_labels.size()).data == test_labels.data).sum()
                n_total += len(batches[batch_no])
            else:
                current_y_preds = preds.data.cpu().tolist()
                for i in range(len(batches[batch_no])):
                    output.append([batches[batch_no][i].id, current_y_preds[i]])

    if batch_label == 'quora':
        target_names = ['non_duplicate', 'duplicate']
    elif batch_label == 'snli' or batch_label == 'multinli':
        target_names = ['entailment', 'neutral', 'contradiction']

    if outfile:
        with open(outfile, 'w') as f:
            f.write('pairID,gold_label' + '\n')
            for item in output:
                f.write(str(item[0]) + ',' + target_names[item[1]] + '\n')
    else:
        print(classification_report(numpy.asarray(y_true), numpy.asarray(y_preds), target_names=target_names))
        return 100. * n_correct / n_total


if __name__ == "__main__":
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                   dictionary.word2idx)
    tasks = []
    if 'quora' in args.task:
        quora_test = data.Corpus(args.data + 'quora/', dictionary)
        quora_test.parse('test.txt', 'quora', args.tokenize, is_test_corpus=True)
        print('quora test set size = ', len(quora_test.data))
        tasks.append(('quora', 2))

    if 'snli' in args.task:
        snli_test = data.Corpus(args.data + 'snli/', dictionary)
        snli_test.parse('test.txt', 'snli', args.tokenize, is_test_corpus=True)
        print('snli test set size = ', len(snli_test.data))
        tasks.append(('snli', 3))

    if 'multinli' in args.task:
        # test matched part
        multinli_test_matched = data.Corpus(args.data + 'multinli/', dictionary)
        multinli_test_matched.parse('test_matched.txt', 'multinli', args.tokenize, is_test_corpus=True)
        print('mutinli test[matched] set size = ', len(multinli_test_matched.data))
        # test mismatched part
        multinli_test_mismatched = data.Corpus(args.data + 'multinli/', dictionary)
        multinli_test_mismatched.parse('test_mismatched.txt', 'multinli', args.tokenize, is_test_corpus=True)
        print('mutinli test[mismatched] set size = ', len(multinli_test_mismatched.data))
        tasks.append(('multinli', 3))

    if tasks:
        model = MultitaskDomainAdapter(dictionary, embeddings_index, args, tasks)
        if args.cuda:
            model = model.cuda()
        helper.load_model_states_from_checkpoint(model, args, args.save_path + 'model_best.pth.tar', 'state_dict',
                                                 args.cuda)
        print('vocabulary size = ', len(dictionary))

        if 'quora' in args.task:
            test_batches = helper.batchify(quora_test.data, args.batch_size)
            test_accuracy = evaluate(model, test_batches, 'quora', dictionary)
            print('quora test accuracy: %f%%' % test_accuracy)

        if 'snli' in args.task:
            test_batches = helper.batchify(snli_test.data, args.batch_size)
            test_accuracy = evaluate(model, test_batches, 'snli', dictionary)
            print('snli test accuracy: %f%%' % test_accuracy)

        if 'multinli' in args.task:
            # test matched part
            test_batches = helper.batchify(multinli_test_matched.data, args.batch_size)
            evaluate(model, test_batches, 'multinli', dictionary, args.save_path + 'multinli_matched.csv')
            # test mismatched part
            test_batches = helper.batchify(multinli_test_mismatched.data, args.batch_size)
            evaluate(model, test_batches, 'multinli', dictionary, args.save_path + 'multinli_mismatched.csv')
