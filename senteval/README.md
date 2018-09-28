# Transfer Learning using SentEval

We use [SentEval](https://github.com/facebookresearch/SentEval) to evaluate the quality of the experimental sentence embeddings. Please visit the [SentEval](https://github.com/facebookresearch/SentEval) github repository to learn more about the toolkit.

### Pretrained Models

To reproduce the numbers reported in our paper, you need the following pretrained models.

1. [ELMo](https://allennlp.org/elmo): 
      - Small [Weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5)|[Options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json)
      - Medium [Weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5)|[Options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json)
      - Original [Weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)|[Options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)
      - Original(5.5B) [Weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5)|[Options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json)
2. [CoVe](https://github.com/salesforce/cove):
      - [Small](https://drive.google.com/open?id=1oKhZRN2SbZTbU4l-dkldR4Vc6Somce2w), [Medium](https://drive.google.com/open?id=1wOr6LQhw1bv-M841d9bGZAT397hJwAWm), [Large](https://drive.google.com/file/d/1SCDBAmvUXO0iPXGDFP2dtMgtjEj66uw7/view?usp=sharing)
3. [GenSen](https://github.com/Maluuba/gensen)
      - Download the weight and vocab files from [here](https://github.com/Maluuba/gensen/blob/master/data/models/download_models.sh). We use the `nli_large_bothskip` and `nli_large_bothskip_parse` models from GenSen.
4. [Sent2vec](https://github.com/wasiahmad/universal_sentence_encoder)
      - We share the weights of the single-task encoders [here](https://drive.google.com/open?id=19aOTqOY-BrOBP_if-7rQhCfpyqIUjW1-).
      - We share the weights of the single-task encoders [here](https://drive.google.com/open?id=11qXHLZnhbuLw4caAi4-KyyoZ5InNxE7-).
