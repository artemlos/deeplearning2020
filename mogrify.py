#%%

import tensorflow as tf
from tensorflow import keras
import datasets


ptb = datasets.PTB()
ptb_char_train = ptb.char_train
ptb_char_valid = ptb.char_valid
ptb_word_train = ptb.word_train
ptb_word_valid = ptb.word_valid
ptb_word_test = ptb.word_test

wikitext2 = datasets.Wikitext2()
wikitext2_train = wikitext2.train
wikitext2_valid = wikitext2.valid
wikitext2_test = wikitext2.test

enwik9 = datasets.Enwik9()
enwik9_train = enwik9.enwik9_train
enwik9_valid = enwik9.enwik9_valid
enwik9_test = enwik9.enwik9_test
print(enwik9_train[0:100])
print(enwik9_valid[0:100])
print(enwik9_test[0:100])



class Hyperparameters():
    r = 5
    k = int((90 + 40) / 2) # k < min(m,n)
    hidden_states = 4

class Model(object):
    def __init__(self, params):
        # model parameters
        self.r = params.r
        self.k = params.k
        self.lstm = tf.keras.layers.LSTM(params.hidden_states)

    def mogrify(self, x_init, h_0, q, r):
        x = x_init
        h_prev = h_0
        for i in range(1, self.r+1):
            if i % 2 != 0:
                x = 2 * tf.sigmoid(tf.matmul(q[i-1], h_prev)) * x
                # print("odd %s, %s" % (i, tf.shape(x)))
            else:
                h_prev = 2 * tf.sigmoid(tf.matmul(r[i-1], x)) * h_prev
                # print("even %s, %s" % (i, tf.shape(h_prev)))

        return x, h_prev

    def matrix_decomposition(self, m, n):
        # currently not decomposition
        # k < min(m,n)

        q_left = tf.random.normal([m, self.k])
        q_right = tf.random.normal([self.k, n])

        r_left = tf.random.normal([n, self.k])
        r_right = tf.random.normal([self.k, m])

        return tf.matmul(q_left, q_right), tf.matmul(r_left, r_right)

    def __call__(self, x):
        return


# d = 100
# m = d
# n = 70
# a = 10
#
# x = tf.random.normal([m, a]) # d x a
# h_0 = tf.random.normal([n, a]) # n x a
#
# params = Hyperparameters()
# model = Model(params)
#
# q = [] # m x n, where m == d
# r = [] # n x m, where m == d, it's just transposed Q
# for i in range(model.r):
#     q_tmp, r_tmp = model.matrix_decomposition(m, n)
#     q.append(q_tmp)
#     r.append(r_tmp)
#
# model.mogrify(x, h_0, q, r)
