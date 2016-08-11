# -*- coding: utf-8 -*-

import os
import collections
import random
import math
import string

import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pylab

from sklearn.manifold import TSNE

from preprocessors import VocabBuilder, StemmingLookup
from utils import get_data


class Philo2Vec(object):
    """
    The algorithm that computes the continuous distributed representations of words,
    in this case, the words used in the stanford philosophy encyclopedia.

    Attributes
        * graph: the tensorflow graph.
        * session: the tensorflow session.
        * optimizer: an instance of tensorflow `Optimizer`, such as
                     `GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.
        * model: the model to use to create the vectorized representation,
                 possible values: `CBOW`, `SKIP_GRAM`.
        * loss_fct: the loss function used to calculate the error,
                    possible values: `SOFTMAX`, `NCE`.
        * embedding_size: dimensionality of word embeddings.
        * neg_sample_size: number of negative samples for each positive sample
        * num_skips: numer of skips for a `SKIP_GRAM` model.
        * context_window: window size, this window is used to create the
                          context for calculating the vector representations
                          [ window target window ].
        * vocab_builder: an instance of the `VocabBuilder` class.
        * batch_size: the batch size for training.
        * log_dir: the folder for storing histograms and other data related to the model.
        * current_step: current step in the training process, this is used
                        to resume training from the same place.
    """
    SOFTMAX = 'softmax'
    NCE = 'nce'

    CBOW = 'cbow'
    SKIP_GRAM = 'skip_gram'

    LOGS = './logs'

    def __init__(self, vocab_builder, model=SKIP_GRAM, graph=None, session=None,
                 optimizer=tf.train.AdagradOptimizer(1.0),
                 loss_fct=SOFTMAX, embedding_size=350, neg_sample_size=5,
                 num_skips=2, context_window=2, batch_size=128, log_dir=None):
        tf.reset_default_graph()
        self.graph = graph or tf.Graph()
        self.session = session or tf.Session(graph=self.graph)
        self.optimizer = optimizer
        self.model = model
        self.loss_fct = loss_fct
        self.embedding_size = embedding_size
        self.neg_sample_size = neg_sample_size
        self.num_skips = num_skips
        self.context_window = context_window
        self.vocab_builder = vocab_builder
        self.batch_size = batch_size
        self.log_dir = self._log_dir(log_dir)
        self.current_step = 0

        self.set_model_graph()

        assert batch_size % num_skips == 0
        assert num_skips <= 2 * context_window

    def _log_dir(self, log_dir):
        log_dir = (
            log_dir or
            self.LOGS + '/' + ''.join(random.choice(string.ascii_uppercase) for _ in range(6)))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def set_model_graph(self):
        """
        Build the graph model and other operations used for calculating similarities
        and vectors operations.
        """
        def set_params():
            self.X = (tf.placeholder(tf.int32, shape=[self.batch_size], name='X')
                      if self.model == self.SKIP_GRAM
                      else tf.placeholder(tf.int32,
                                          shape=[self.batch_size, self.context_window * 2],
                                          name='X'))
            self.y = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='y')

            init_width = 0.5 / self.embedding_size
            self.embeddings = tf.Variable(
                tf.random_uniform([self.vocab_builder.size, self.embedding_size],
                                  -init_width, init_width),
                name='embeddings')
            self.w = tf.Variable(
                tf.truncated_normal([self.vocab_builder.size, self.embedding_size],
                                    stddev=1.0 / math.sqrt(self.embedding_size)),
                name='w')
            self.b = tf.Variable(tf.zeros([self.vocab_builder.size]), name='b')

        def set_optimize():
            embed = tf.nn.embedding_lookup(self.embeddings, self.X)
            if self.model == self.CBOW:
                embed = tf.reduce_sum(embed, 1)
            if self.loss_fct == self.SOFTMAX:  # softmax
                self.loss = tf.reduce_mean(
                    tf.nn.sampled_softmax_loss(self.w, self.b, embed, self.y, self.neg_sample_size,
                                               self.vocab_builder.size))
            else:  # nce
                self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(self.w, self.b, embed, self.y, self.neg_sample_size,
                                   self.vocab_builder.size))

            self.train = self.optimizer.minimize(self.loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / norm

        def set_similarity():
            self.validate_words = tf.placeholder(tf.int32, shape=[None])
            self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings,
                                                           self.validate_words)
            self.similarity = tf.matmul(self.valid_embeddings,
                                        tf.transpose(self.normalized_embeddings))

        def set_closest_to_vec():
            self.valid_vector = tf.placeholder(tf.float32, shape=[1, self.embedding_size])
            self.closest_vec = tf.matmul(self.valid_vector,
                                         tf.transpose(self.normalized_embeddings))

        def set_histograms():
            histograms = [tf.histogram_summary('embeddings', self.embeddings),
                          tf.histogram_summary('w', self.w),
                          tf.histogram_summary('b', self.b),
                          tf.scalar_summary('loss', self.loss)]
            self.summaries = tf.merge_summary(histograms)
            self.summary_writer = tf.train.SummaryWriter(self.log_dir, self.graph)

        with self.graph.as_default():
            set_params()
            set_optimize()
            set_similarity()
            set_closest_to_vec()
            set_histograms()
            self.session.run(tf.initialize_all_variables())

    def _similarity(self, word_idxs, similar_idxs, top_k, return_words):
        top_k_nearest = [
            (
                word, [(similar_i, similar_idxs[i, similar_i])
                       for similar_i in (1 - similar_idxs[i, :]).argsort()[1:top_k + 1]]
            )
            for i, word in enumerate(word_idxs)]

        if not return_words:
            return top_k_nearest

        return [(self.vocab_builder.idx2word[word],
                 [(self.vocab_builder.idx2word[nearest_w], distance) for (nearest_w, distance) in
                  nearest_words])
                for (word, nearest_words) in top_k_nearest]

    def get_similar_words(self, words, top_k=8, return_words=True):
        word_idxs = np.array([self.vocab_builder.word2idx[word] for word in words])
        similar_idxs, = self.session.run([self.similarity], {self.validate_words: word_idxs})
        return self._similarity(word_idxs, similar_idxs, top_k, return_words)

    def get_close_to_vec(self, vec, top_k=8, return_words=True):
        close_vecs, = self.session.run([self.closest_vec], {self.valid_vector: [vec]})
        return self._similarity([0], close_vecs, top_k, return_words)

    def evaluate_operation(self, expression):
        # make a stack of words and a list of operations
        expression = expression.split()
        words, operations = expression[::-2], expression[1::2]
        # get words vectors
        word_idxs = [self.vocab_builder.word2idx[word] for word in words]
        vec_words, = self.session.run([self.valid_embeddings], {self.validate_words: word_idxs})
        # calculate the new word vector
        vec_words = list(vec_words)
        for op in operations:
            vec_words.append(vec_words.pop() + vec_words.pop()
                             if op == '+'
                             else vec_words.pop() - vec_words.pop())

        # get the closest words to the new vector
        return self.get_close_to_vec(vec_words.pop())

    def plot(self, embeddings, words, num_points=400):
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        two_d_embeddings = tsne.fit_transform(embeddings[:num_points, :])

        assert two_d_embeddings.shape[0] >= len(words), 'More labels than embeddings'
        pylab.figure(figsize=(15, 15))  # in inches
        for i, label in enumerate(words):
            x, y = two_d_embeddings[i, :]
            pylab.scatter(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                           ha='right', va='bottom')
        pylab.show()

    def get_skip_gram_batch_words(self, X, y):
        print("""
        NUMBER SKIPS = {} and CONTEXT WINDOW = {}:
        batch: {}
        labels: {}
        """.format(self.num_skips,
                   self.context_window,
                   [self.vocab_builder.idx2word[i] for i in X],
                   [self.vocab_builder.idx2word[i] for i in y.reshape(self.batch_size)]))

    def get_cbow_batch_words(self, X, y):
        print("""
        CONTEXT WINDOW = {}:
        batch: {}
        labels: {}
        """.format(self.context_window,
                   [[self.vocab_builder.idx2word[w] for w in i] for i in X],
                   [self.vocab_builder.idx2word[i] for i in y.reshape(self.batch_size)]))

    def batch_skip_gram(self, step):
        data_index = step % self.vocab_builder.size
        span = 2 * self.context_window + 1  # [ window target window ]
        X = np.ndarray(shape=(self.batch_size,), dtype=np.int32)
        y = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.vocab_builder.data[data_index])
            data_index = (data_index + 1) % self.vocab_builder.size
        for i in range(self.batch_size // self.num_skips):
            target = self.context_window  # target label at the center of the buffer
            targets_to_avoid = [self.context_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                X[i * self.num_skips + j] = buffer[self.context_window]
                y[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(self.vocab_builder.data[data_index])
        return X, y

    def batch_cbow(self, step):
        data_index = step % self.vocab_builder.size
        span = 2 * self.context_window + 1  # [ window target window ]
        X = np.ndarray(shape=(self.batch_size, span - 1), dtype=np.int32)
        y = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.vocab_builder.data[data_index])
            data_index = (data_index + 1) % len(self.vocab_builder.data)
        for i in range(self.batch_size):
            # just for testing
            buffer_list = list(buffer)
            y[i, 0] = buffer_list.pop(self.context_window)
            X[i] = buffer_list
            # iterate to the next buffer
            buffer.append(self.vocab_builder.data[data_index])
        return X, y

    def _run_epoch(self, steps, every_n_steps, validation_data):
        average_loss = 0
        start_step = self.current_step
        self.current_step += steps
        for step in range(start_step, self.current_step):
            X, y = self.batch_cbow(step) if self.model == self.CBOW else self.batch_skip_gram(step)
            feed_dict = {self.X: X, self.y: y}
            _, l = self.session.run([self.train, self.loss], feed_dict=feed_dict)

            average_loss += l
            if step % every_n_steps == 0:
                summary = self.session.run(self.summaries, feed_dict=feed_dict)
                self.summary_writer.add_summary(summary, step)

                average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0

        if validation_data:
            for v in self.get_similar_words(validation_data):
                print(v)

    def fit(self, steps=None, epochs=15, every_n_steps=1000, validation_data=None):
        steps = steps or self.vocab_builder.size // self.batch_size
        with self.graph.as_default():
            for epoch in range(1, epochs + 1):
                print('Epoch {}:'.format(epoch))
                self._run_epoch(steps, every_n_steps, validation_data)

            return self.session.run(self.normalized_embeddings)


def main():
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
