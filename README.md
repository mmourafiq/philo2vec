# philo2vec
A Tensorflow implementation of word2vec applied to [stanford philosophy encyclopedia](http://plato.stanford.edu/)


### The repo contains:

  * an object to crawl data from the philosophy encyclopedia; [PlatoData](https://github.com/mouradmourafiq/philo2vec/blob/master/data.py)
  * a object to build the vocabulary based on the crawled data; [VocabBuilder](https://github.com/mouradmourafiq/philo2vec/blob/master/preprocessors.py)
  * the model that computes the continuous distributed representations of words; [Philo2Vec](https://github.com/mouradmourafiq/philo2vec/blob/master/models.py)
  

### The hyperparams for the VocabBuilder:

  * min_frequency: the minimum frequency of the words to be used in the model.
  * size: the size of the data, the model then use the top size most frequenct words.

### The hyperparams of the model:
   
  * optimizer: an instance of tensorflow `Optimizer`, such as `GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.
  * model: the model to use to create the vectorized representation, possible values: `CBOW`, `SKIP_GRAM`.
  * loss_fct: the loss function used to calculate the error, possible values: `SOFTMAX`, `NCE`.
  * embedding_size: dimensionality of word embeddings.
  * neg_sample_size: number of negative samples for each positive sample
  * num_skips: numer of skips for a `SKIP_GRAM` model.
  * context_window:  window size, this window is used to create the context for calculating the vector representations [ window target window ].


### quick usage:

```python
params = {
    'model': Philo2Vec.CBOW,
    'loss_fct': Philo2Vec.NCE,
    'context_window': 5,
}
x_train = get_data()
validation_words = ['kant', 'descartes', 'human', 'natural']
x_validation = [StemmingLookup.stem(w) for w in validation_words]
vb = VocabBuilder(x_train, min_frequency=5)
pv = Philo2Vec(vb, **params)
pv.fit(epochs=30, validation_data=x_validation)
return pv
```


### installation

The dependencies used for this module can be easily installed with pip:

> pip install -r requirements.txt
