## [Multi-task Learning for Text Classification]()

Two different multi-task learning (MTL) framework is implemented in this project. They are:

 - **Fully shared model** - In this MTL framework, all participating tasks share a single sentence encoder.
 - **Shared private model** - In this MTL framework, all participating tasks have private sentence encoder. In addition, a global encoder is shared accross all tasks.

-----

### Fully Shared Model

Two kinds of fully shared model is implemented in this project.

 - **Simple shared model** - This architecture consists of a bi-RNN with max-pooling sentence encoder and task specific feed forward neural networks.
 - **Domain adaptation model** - This architecture consists of a bi-RNN with max-pooling sentence encoder, domain adaptation layer and task specific feed forward neural networks.
 
More details about the domain adaptation model can be found in this [paper](http://www.aclweb.org/anthology/W17-2612).

To run the above two models, the command line argument `--classifier` value needs to be passed.

 - if `--classifier` is set to 0 (which is by default set to 0), then simple shared model will be executed.
 - if `--classifier` is set to 1, then domain adaptation model will be executed.

-----

### Shared Private Model
 
Two kinds of shared private model is implemented in this project.

 - **Non-adversarial model** - This architecture consists of task specific bi-RNN sentence encoders with max-pooling, a shared bi-RNN sentence encoder with max-pooling and task specific feed forward neural networks.
 - **Adversarial model** - Similar to the previous model, this model is trained in adversarial fashion.
 
More details about the above two models can be found in this [paper](http://www.aclweb.org/anthology/P17-1001).

**Important**

Along with other command line parameters, the shared private model also accepts the following command line argument values.

```
  --adversarial         turn on adversarial training (default:off)
  --beta                hyper-parameter for adversarial loss
  --gamma               hyper-parameter for difference loss
```

-----

#### Requirement

* python 3.5+
* pytorch (0.2.0+)
* numpy 1.12.1
* [GloVe 300d word embeddings (840B)](https://nlp.stanford.edu/projects/glove/)

#### Command Line Arguments

For the domain adapatation model the `main.py` script accepts the following arguments:

```
optional arguments:
  -h, --help            show this help message and exit
  --task                list of task names [required]
  --data                location of the data corpus (default = '../data/')
  --max_example         number of training examples (default = -1 [all examples])
  --classifier          classifier choice (default: 0)
  --tokenize            use NLTK's word_tokenize for tokenizing data
  --model               type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --optimizer           adam or sgd,lr=0.1 (default = "sgd,lr=0.1")
  --lrshrink            shrink factor for sgd (default = 5)
  --minlr               minimum lr (default = 1e-5 [only applicable for sgd])
  --average_loss        consider average loss over mini-batches
  --bidirection         use bidirectional recurrent unit
  --emsize              size of word embeddings
  --emtraining          turn embedding training on (default: off)
  --nhid                humber of hidden units per layer
  --fc_dim              number of hidden units in fully connected layers
  --nlayers             number of layers
  --projection          type of domain projection (default: mask, must be mask/linear)
  --lr_decay            decay ratio for learning rate
  --max_norm            max norm for gradient clipping
  --epochs              upper epoch limit
  --batch_size N        batch size
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --dropout_fc          dropout applied to fully connected layers (default: 0)
  --seed SEED           random seed
  --cuda                use CUDA
  --print_every         training report interval
  --plot_every          plotting interval
  --save_path           path to save the final model
  --resume              path of previously saved model to resume (default: none) 
  --word_vectors_file   GloVe word embedding file name (default: '../data/glove/')
  --word_vectors_directory  Path of GloVe word embeddings (default: 'glove.840B.300d.txt')
  ```

#### How to train and test the model?

The following command can be used to train the fully shared model on three tasks, namely, quora, snli and multinli.

> CUDA_VISIBLE_DEVICES=GPU_ID python main.py --cuda --bidirection --task quora snli multinli

Similarly, to test the model, you need to run the following command (run the `test.py` file).

> CUDA_VISIBLE_DEVICES=GPU_ID python test.py --cuda --bidirection --task quora snli multinli

