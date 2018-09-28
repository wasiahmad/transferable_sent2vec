## [Bi-LSTM with Max-pooling for Sentence Classification]()

This is the pytorch implementation of [InferSent](https://github.com/facebookresearch/InferSent) for a set of sentence pair classification tasks. This implementation covers the following tasks.

 - SNLI
 - Quora
 - Multi-NLI
 - AllNLI


##### Requirement

* python 3.5
* pytorch (0.2.0)
* numpy 1.12.1
* [GloVe 300d word embeddings (840B)](https://nlp.stanford.edu/projects/glove/)

##### Command Line Arguments

For the domain adapatation model the `main.py` script accepts the following arguments:

```
optional arguments:
  -h, --help            show this help message and exit
  --data                location of the data corpus (default: '../data/')
  --task                name of the task [any one of snli, quora, multinli and allnli], required
  --num_classes         number of classes associated with the task, required
  --test                data partition on which test performance should be measured [default: 'test']
  --max_example         number of training examples (default: -1 [all examples])
  --tokenize            use NLTK's word_tokenize for tokenizing data
  --model               type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --optimizer           adam or sgd,lr=0.1 (default = "sgd,lr=0.1")
  --lrshrink            shrink factor for sgd (default = 5)
  --minlr               minimum lr (default = 1e-5 [only applicable for sgd])
  --average_loss        consider average loss over mini-batches for backpropagation
  --bidirection         use bidirectional recurrent unit
  --emsize              size of word embeddings
  --emtraining          turn embedding training on (default: off)
  --nhid                humber of hidden units per layer
  --fc_dim              number of hidden units in fully connected layers
  --nlayers             number of layers
  --pool_type           pooling type [any one of max, mean and last] (default: 'max')
  --lr_decay            decay ratio for learning rate (default: 0.99)
  --max_norm            max norm for gradient clipping (default: 5.0)
  --epochs              upper epoch limit (default: 25)
  --batch_size N        batch size (default: 64)
  --dropout DROPOUT     dropout applied to layers (0 = no dropout, default: 0.2)
  --dropout_fc          dropout applied to fully connected layers (default: 0)
  --seed SEED           random seed
  --cuda                use CUDA
  --print_every         training report interval
  --plot_every          plotting interval
  --save_path           path to save the final model
  --resume              path of previously saved model to resume (default: none) 
  --word_vectors_file   GloVe word embedding file name (default: '../data/glove/')
  --word_vectors_directory  Path of GloVe word embeddings
  ```
  
#### How to train and test the model?

For example, if you want to run the model on Quora task, you can use the following command.

> CUDA_VISIBLE_DEVICES=GPU_ID python main.py --cuda --bidirection --task quora --num_classes 2

Similarly, to test the model, you need to run the following command (run the `test.py` file).

> CUDA_VISIBLE_DEVICES=GPU_ID python test.py --cuda --bidirection --task quora --num_classes 2
