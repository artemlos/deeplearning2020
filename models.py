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


class LSTMBaseLine(tf.keras.Model):
    def __init__(self, vocab_size, sequence_length, batch_size, tie_embedding=False, skip_connection=False):
        super(LSTMBaseLine, self).__init__()
        self.hidden_states = 64
        self.embedding_dim = 256
        # this is needed to make the input embedding weight matrix compatible with the matrix multiplication of the input in the tied dense layer
        self.hidden_state_last = self.embedding_dim
        self.rnn_units = 1024
        self.batch_size = batch_size
        self.seq_length = sequence_length
        self.tie_embedding = tie_embedding
        self.skip_connection = skip_connection

        # model architecture
        self.input_embedding = tf.keras.layers.Embedding(vocab_size, self.embedding_dim,
                                                         batch_input_shape=[self.batch_size, None])
        self.lstm_layer1 = tf.keras.layers.LSTM(self.rnn_units,
                                                return_sequences=True,
                                                stateful=True,
                                                recurrent_initializer='glorot_uniform')
        self.lstm_layer2 = tf.keras.layers.LSTM(self.hidden_state_last,
                                                return_sequences=True,
                                                stateful=True,
                                                recurrent_initializer='glorot_uniform')
        if skip_connection:
            # dimension aligning layer to make additive skip connection work, otherwise dimensions don't align for the operation
            self.lstm_layer_dimens_aligning = tf.keras.layers.LSTM(self.lstm_layer2.units,
                                                                   return_sequences=True,
                                                                   stateful=True,
                                                                   recurrent_initializer='glorot_uniform')
        if tie_embedding:
            self.tied_dense = TiedDense(vocab_size, self.input_embedding)
        else:
            self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, **kwargs):
        x = self.input_embedding(inputs)

        lstm_layer1_output = self.lstm_layer1(x)

        if self.skip_connection:
            # align the dimens for lstm_layer1_output with output from lstm_layer_2
            lstm_layer_1_output_resized = self.lstm_layer_dimens_aligning(lstm_layer1_output)

        lstm_layer2_output = self.lstm_layer2(lstm_layer1_output)

        if self.skip_connection:
            lstm_layer2_output = tf.keras.layers.add([lstm_layer2_output, lstm_layer_1_output_resized])

        if self.tie_embedding:
            return self.tied_dense(lstm_layer2_output)

        return self.dense(lstm_layer2_output)
