#%%

import tensorflow as tf
from tensorflow import keras
import datasets
import os


class LSTMBaseLine(object):
    def __init__(self, vocab_size, sequence_length, batch_size):
        # model parameters
        self.depth = 2
        self.hidden_states = 64
        self.embedding_dim = 256
        self.rnn_units = 1024
        self.batch_size = batch_size
        self.seq_length = sequence_length

        self.model = self.build_model(vocab_size)


    def build_model(self, vocab_size):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(vocab_size, self.embedding_dim,
                                            batch_input_shape=[self.batch_size, None]))
        for i in range(self.depth):
            if i < self.depth - 1:
                model.add(tf.keras.layers.LSTM(self.hidden_states,
                                               return_sequences=True,
                                               #return_state=True,
                                               stateful=True,
                                               recurrent_initializer='glorot_uniform'))
            else:
                # last layer of LSTM should not return
                model.add(tf.keras.layers.LSTM(self.hidden_states,
                                               return_sequences=True,
                                               stateful=True,
                                               recurrent_initializer='glorot_uniform'))
        model.add(tf.keras.layers.Dense(vocab_size))

        return model

    # def forward(self, x, state=None):
    #     # x, h_prev = self.mogrify(x, )
    #     res = self.model(x, initial_state=None)
    #     return res

    # def __call__(self, x):
    #     return

