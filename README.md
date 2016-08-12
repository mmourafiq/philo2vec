# philo2vec

A Tensorflow implementation of word2vec applied to [stanford philosophy encyclopedia](http://plato.stanford.edu/)
The implementation supports both `cbow` and `skip gram`

for more reference, please have a look at this papers:
 
 * [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
 * [word2vec Parameter Learning Explained](http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf)
 * [Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method](http://arxiv.org/pdf/1402.3722v1.pdf)


### The repo contains:

  * an object to crawl data from the philosophy encyclopedia; [PlatoData](https://github.com/mouradmourafiq/philo2vec/blob/master/data.py)
  * a object to build the vocabulary based on the crawled data; [VocabBuilder](https://github.com/mouradmourafiq/philo2vec/blob/master/preprocessors.py)
  * the model that computes the continuous distributed representations of words; [Philo2Vec](https://github.com/mouradmourafiq/philo2vec/blob/master/models.py)
  

### The hyperparams for the VocabBuilder:

  * *min_frequency*: the minimum frequency of the words to be used in the model.
  * *size*: the size of the data, the model then use the top size most frequenct words.

### The hyperparams of the model:
   
  * *optimizer*: an instance of tensorflow `Optimizer`, such as `GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.
  * *model*: the model to use to create the vectorized representation, possible values: `CBOW`, `SKIP_GRAM`.
  * *loss_fct*: the loss function used to calculate the error, possible values: `SOFTMAX`, `NCE`.
  * *embedding_size*: dimensionality of word embeddings.
  * *neg_sample_size*: number of negative samples for each positive sample
  * *num_skips*: numer of skips for a `SKIP_GRAM` model.
  * *context_window*:  window size, this window is used to create the context for calculating the vector representations [ window target window ].


### Quick usage:

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

```python
params = {
    'model': Philo2Vec.SKIP_GRAM,
    'loss_fct': Philo2Vec.SOFTMAX,
    'context_window': 2,
    'num_skips': 4,
    'neg_sample_size': 2,
}
x_train = get_data()
validation_words = ['kant', 'descartes', 'human', 'natural']
x_validation = [StemmingLookup.stem(w) for w in validation_words]
vb = VocabBuilder(x_train, min_frequency=5)
pv = Philo2Vec(vb, **params)
pv.fit(epochs=30, validation_data=x_validation)
return pv
```


### Installation

The dependencies used for this module can be easily installed with pip:

```
> pip install -r requirements.txt
```

### Some interesting results

#### Similarities

Similar words to `death`:
 
```
untimely
ravages
grief
torment
```

Similar words to `god`: 

```
divine
De Providentia
christ
Hesiod
```

Similar words to `love`: 

```
friendship
affection
christ
reverence
```

Similar words to `life`:

```
career
live
lifetime
community
society
```

Similar words to `brain`:

```
neurological
senile
nerve
nervous
```

#### operations
 
Evaluating `hume - empiricist + rationalist`:

```
descartes
malebranche
spinoza
hobbes
herder
```

Evaluating `ethics - rational`:

```
hiroshima
```

Evaluating `ethic - reason`:

```
inegalitarian
anti-naturalist
austere
```

Evaluating `moral - rational`:

```
commonsense
```

Evaluating `life - death + love`:

```
self-positing
friendship
care
harmony
```

Evaluating `death + choice`:

```
regret
agony
misfortune
impending
```

Evaluating `god + human`:

```
divine
inviolable
yahweh
god-like
man
```

Evaluating `god + religion`:

```
amida
torah
scripture
buddha
sokushinbutsu
```

Evaluating `politic + moral`:

```
rights-oriented
normative
ethics
integrity
```

### Training details

#### skip_gram:
 
<img width="873" alt="skip_gram_loss" src="https://cloud.githubusercontent.com/assets/1261626/17628496/d19a0d42-60b5-11e6-8cbc-20f1aac3becc.png">

<img width="874" alt="skip_gram_embeddings" src="https://cloud.githubusercontent.com/assets/1261626/17628497/d19a811e-60b5-11e6-8e7c-733309b5249d.png">

<img width="878" alt="skip_gram_w" src="https://cloud.githubusercontent.com/assets/1261626/17628499/d1a1d00e-60b5-11e6-8638-8f68c288205b.png">

<img width="862" alt="skip_gram_b" src="https://cloud.githubusercontent.com/assets/1261626/17628498/d19b6778-60b5-11e6-8ed1-e45b0566d8c7.png">

#### cbow:


<img width="867" alt="cbow_loss" src="https://cloud.githubusercontent.com/assets/1261626/17630043/e41aae9c-60bd-11e6-9289-92d5dc58e55f.png">

<img width="885" alt="cbow_embedding" src="https://cloud.githubusercontent.com/assets/1261626/17630045/e41e2a90-60bd-11e6-8b49-891c3ba8ebf7.png">

<img width="869" alt="cbow_w" src="https://cloud.githubusercontent.com/assets/1261626/17630044/e41b5e6e-60bd-11e6-9eb7-55dd1dbfda48.png">

<img width="856" alt="cbow_b" src="https://cloud.githubusercontent.com/assets/1261626/17630042/e417f292-60bd-11e6-92e7-a8e9a5ddfb32.png">
