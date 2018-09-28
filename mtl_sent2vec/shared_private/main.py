###############################################################################
# Author: Wasi Ahmad
# Project: Adversarial Multi-task Learning for Text Classification
# Date Created: 10/18/2017
#
# File Description: This is the main script from where all experimental
# execution begins.
###############################################################################

import util, data, train, helper, torch, numpy, os, sys
from train import Train
from classifier import Generator, Discriminator

args = util.get_args()


def main():
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

    print('\ncommand-line params : {0}\n'.format(sys.argv[1:]))
    print('{0}\n'.format(args))

    ###############################################################################
    # Load data
    ###############################################################################

    dictionary = data.Dictionary()
    tasks = []
    train_dict, dev_dict = {}, {}

    if 'quora' in args.task:
        print('**Task name : Quora**')
        # load quora dataset
        quora_train = data.Corpus(args.data, dictionary)
        quora_train.parse('quora/train.txt', 'quora', args.tokenize, args.max_example)
        print('Found {} pairs of train sentences.'.format(len(quora_train.data)))

        quora_dev = data.Corpus(args.data, dictionary)
        quora_dev.parse('quora/dev.txt', 'quora', args.tokenize)
        print('Found {} pairs of dev sentences.'.format(len(quora_dev.data)))

        quora_test = data.Corpus(args.data, dictionary)
        quora_test.parse('quora/test.txt', 'quora', args.tokenize)
        print('Found {} pairs of test sentences.'.format(len(quora_test.data)))

        tasks.append(('quora', 2))
        train_dict['quora'] = quora_train
        dev_dict['quora'] = quora_dev

    if 'snli' in args.task:
        print('**Task name : SNLI**')
        # load snli dataset
        snli_train = data.Corpus(args.data, dictionary)
        snli_train.parse('snli/train.txt', 'snli', args.tokenize, args.max_example)
        print('Found {} pairs of train sentences.'.format(len(snli_train.data)))

        snli_dev = data.Corpus(args.data, dictionary)
        snli_dev.parse('snli/dev.txt', 'snli', args.tokenize)
        print('Found {} pairs of dev sentences.'.format(len(snli_dev.data)))

        snli_test = data.Corpus(args.data, dictionary)
        snli_test.parse('snli/test.txt', 'snli', args.tokenize)
        print('Found {} pairs of test sentences.'.format(len(snli_test.data)))

        tasks.append(('snli', 3))
        train_dict['snli'] = snli_train
        dev_dict['snli'] = snli_dev

    if 'multinli' in args.task:
        print('**Task name : Multi-NLI**')
        # load multinli dataset
        multinli_train = data.Corpus(args.data, dictionary)
        multinli_train.parse('multinli/train.txt', 'multinli', args.tokenize, args.max_example)
        print('Found {} pairs of train sentences.'.format(len(multinli_train.data)))

        multinli_dev = data.Corpus(args.data, dictionary)
        multinli_dev.parse('multinli/dev_matched.txt', 'multinli', args.tokenize)
        # multinli_dev.parse('multinli/dev_mismatched.txt', 'multinli', args.tokenize)
        print('Found {} pairs of dev sentences.'.format(len(multinli_dev.data)))

        multinli_test = data.Corpus(args.data, dictionary)
        multinli_test.parse('multinli/test_matched.txt', 'multinli', args.tokenize)
        # multinli_test.parse('multinli/test_mismatched.txt', 'multinli', args.tokenize)
        print('Found {} pairs of test sentences.'.format(len(multinli_test.data)))

        tasks.append(('multinli', 3))
        train_dict['multinli'] = multinli_train
        dev_dict['multinli'] = multinli_dev

    if 'allnli' in args.task:
        print('**Task name : AllNLI**')
        # load allnli dataset
        allnli_train = data.Corpus(args.data, dictionary)
        allnli_train.parse('snli/train.txt', 'snli', args.tokenize, args.max_example)
        allnli_train.parse('multinli/train.txt', 'multinli', args.tokenize, args.max_example)
        print('Found {} pairs of train sentences.'.format(len(allnli_train.data)))

        allnli_dev = data.Corpus(args.data, dictionary)
        allnli_dev.parse('snli/dev.txt', 'snli', args.tokenize)
        allnli_dev.parse('multinli/dev_matched.txt', 'multinli', args.tokenize)
        # allnli_dev.parse('multinli/dev_mismatched.txt', 'multinli', args.tokenize)
        print('Found {} pairs of dev sentences.'.format(len(allnli_dev.data)))

        allnli_test = data.Corpus(args.data, dictionary)
        allnli_test.parse('snli/test.txt', 'snli', args.tokenize)
        allnli_test.parse('multinli/test_matched.txt', 'multinli', args.tokenize)
        # allnli_test.parse('multinli/test_mismatched.txt', 'multinli', args.tokenize)
        print('Found {} pairs of test sentences.'.format(len(allnli_test.data)))

        tasks.append(('allnli', 3))
        train_dict['allnli'] = allnli_train
        dev_dict['allnli'] = allnli_dev

    print('\nvocabulary size = ', len(dictionary))
    # save the dictionary object to use during testing
    helper.save_object(dictionary, args.save_path + 'dictionary.p')

    if not args.use_elmo:
        embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file,
                                                       dictionary.word2idx)
        print('number of OOV words = ', len(dictionary) - len(embeddings_index))
    else:
        embeddings_index = None

    # ###############################################################################
    # # Build the model
    # ###############################################################################

    if not tasks:
        return

    optim_fn, optim_params = helper.get_optimizer(args.optimizer)
    generator = Generator(dictionary, embeddings_index, args, tasks)
    # print(generator)
    optimizerG = optim_fn(filter(lambda p: p.requires_grad, generator.parameters()), **optim_params)
    if args.adversarial:
        discriminator = Discriminator(args, len(tasks))
        # print(discriminator)
        optimizerD = optim_fn(discriminator.parameters(), **optim_params)

    best_acc = 0
    # for training on multiple GPUs. use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        if len(cuda_visible_devices) > 1:
            generator = torch.nn.DataParallel(generator, device_ids=cuda_visible_devices)
            discriminator = torch.nn.DataParallel(discriminator, device_ids=cuda_visible_devices)
    if args.cuda:
        generator = generator.cuda()
        if args.adversarial:
            discriminator = discriminator.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            generator.load_state_dict(checkpoint['state_dict_G'])
            optimizerG.load_state_dict(checkpoint['optimizerG'])
            if args.adversarial:
                discriminator.load_state_dict(checkpoint['state_dict_D'])
                optimizerD.load_state_dict(checkpoint['optimizerD'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # ###############################################################################
    # # Train the model
    # ###############################################################################

    if args.adversarial:
        train = Train((generator, discriminator), (optimizerG, optimizerD), dictionary, embeddings_index, args,
                      best_acc)
    else:
        train = Train(generator, optimizerG, dictionary, embeddings_index, args, best_acc)
    train.set_train_dev_corpus(train_dict, dev_dict)
    train.train_epochs(args.start_epoch, args.epochs)


if __name__ == '__main__':
    if args.task:
        main()
