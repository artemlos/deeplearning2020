#%%

import tensorflow as tf
from tensorflow import keras
import datasets


### Paths

ptb_char_train_path = "datasets\ptb\ptb.char.train.txt"
ptb_char_valid_path = "datasets\ptb\ptb.char.valid.txt"
ptb_word_train_path = "datasets\ptb\ptb.train.txt"
ptb_word_valid_path = "datasets\ptb\ptb.valid.txt"
ptb_word_test_path = "datasets\ptb\ptb.test.txt"

wikitext2_train_path = "datasets\wikitext-2-v1\wikitext-2\wiki.train.tokens"
wikitext2_valid_path = "datasets\wikitext-2-v1\wikitext-2\wiki.valid.tokens"
wikitext2_test_path = "datasets\wikitext-2-v1\wikitext-2\wiki.test.tokens"

enwik9_path = "datasets\enwik9\enwik9"

# load datasets
ptb_char_train = datasets.Dataset(ptb_char_train_path)
ptb_char_valid = datasets.Dataset(ptb_char_valid_path)
ptb_word_train = datasets.Dataset(ptb_word_train_path)
ptb_word_valid = datasets.Dataset(ptb_word_valid_path)
ptb_word_test = datasets.Dataset(ptb_word_test_path)

wikitext2_train = datasets.Dataset(wikitext2_train_path)
wikitext2_valid = datasets.Dataset(wikitext2_valid_path)
wikitext2_test = datasets.Dataset(wikitext2_test_path)

enwik9 = datasets.Dataset(enwik9_path)
train_offset = 9*10**7
valid_offset = train_offset + 5*10**6
test_offset = valid_offset + 5*10**6
enwik9_train = enwik9.dataset[:train_offset] # first 90 million for training
enwik9_valid = enwik9.dataset[train_offset: valid_offset] # 5 million for valid
enwik9_test = enwik9.dataset[valid_offset: test_offset] # 5 million for test



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


d = 100
m = d
n = 70
a = 10

x = tf.random.normal([m, a]) # d x a
h_0 = tf.random.normal([n, a]) # n x a

params = Hyperparameters()
model = Model(params)

q = [] # m x n, where m == d
r = [] # n x m, where m == d, it's just transposed Q
for i in range(model.r):
    q_tmp, r_tmp = model.matrix_decomposition(m, n)
    q.append(q_tmp)
    r.append(r_tmp)

model.mogrify(x, h_0, q, r)
