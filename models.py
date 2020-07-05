# %%

import tensorflow as tf
from tensorflow import keras
import datasets
import os


class TiedDense(tf.keras.layers.Dense):
    def __init__(self, output_dim, input_embedding_layer, activation=None, **kwargs):
        super(TiedDense, self).__init__(output_dim, **kwargs)
        self.input_embedding_layer = input_embedding_layer
        self.activation = activation
        # self.initializer = initializer
        # self.bias_regularizer = regularizer
        # self.constraint = constraint

    def build(self, input_shape):
        self.b = self.add_weight(shape=(self.units,),
                                 name='bias',
                                 initializer=self.bias_initializer,
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

    def call(self, inputs):
        output = tf.matmul(inputs, tf.transpose(self.input_embedding_layer.embeddings)) + self.b
        if self.activation is not None:
            return self.activation(output)
        return output


class LSTMBaseLine(object):
    def __init__(self, vocab_size, sequence_length, batch_size, tie_embedding=False):
        # model parameters
        self.depth = 2
        self.hidden_states = 64
        self.embedding_dim = 256
        # this is needed to make the input embedding weight matrix compatible with the matrix multiplication of the input in the tied dense layer
        self.hidden_state_last = self.embedding_dim
        self.rnn_units = 1024
        self.batch_size = batch_size
        self.seq_length = sequence_length

        self.model = self.build_model(vocab_size, tie_embedding)

    def build_model(self, vocab_size, tie_embedding):
        model = tf.keras.Sequential()
        input_embedding = tf.keras.layers.Embedding(vocab_size, self.embedding_dim, batch_input_shape=[self.batch_size, None])
        model.add(input_embedding)
        for i in range(self.depth):
            if i < self.depth - 1:
                model.add(tf.keras.layers.LSTM(self.hidden_states,
                                               return_sequences=True,
                                               # return_state=True,
                                               stateful=True,
                                               recurrent_initializer='glorot_uniform'))
            else:
                # Usually last layer of LSTM should return only the last output for each input sequence
                # (a 2D tensor of shape (batch_size, output_features))
                # However, for char/word prediction we need output from each time step to know what the predicted
                # words or characters were of given sequence length, so we need (batch_size, seq_length, output_features)
                model.add(tf.keras.layers.LSTM(self.hidden_state_last,
                                               return_sequences=True,
                                               stateful=True,
                                               recurrent_initializer='glorot_uniform'))
        if tie_embedding:
            model.add(TiedDense(vocab_size, input_embedding))
        else:
            model.add(tf.keras.layers.Dense(vocab_size))

        return model

    # def forward(self, x, state=None):
    #     # x, h_prev = self.mogrify(x, )
    #     res = self.model(x, initial_state=None)
    #     return res

    # def __call__(self, x):
    #     return
