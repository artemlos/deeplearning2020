import tensorflow as tf
import models
from mogrifier import mogrify, matrix_decomposition
from datasets import Dataset, split_input_target
import time
import os

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def main():
    # Load datasets

    # ptb_char_train = Dataset("ptb.char.train")
    # ptb_char_valid = Dataset("ptb.char.valid")
    ptb_word_train = Dataset("ptb.train")
    ptb_word_valid = Dataset("ptb.valid")
    ptb_word_test = Dataset("ptb.test")
    # print(ptb_char_train.data[0:100])
    # print(ptb_char_valid.data[0:100])
    # print(ptb_word_train.data[0:100])
    # print(ptb_word_valid.data[0:100])
    # print(ptb_word_test.data[0:100])

    # wikitext2_train = Dataset("wiki.train")
    # wikitext2_valid = Dataset("wiki.valid")
    # wikitext2_test = Dataset("wiki.test")
    # print(wikitext2_train.data[0:100])
    # print(wikitext2_valid.data[0:100])
    # print(wikitext2_test.data[0:100])

    # file_dest = os.path.join("C:\\", "Users", "Harry", ".keras", "datasets")
    # enwik9_train = Dataset("enwik.train", file_dest)
    # enwik9_valid = Dataset("enwik.valid", file_dest)
    # enwik9_test = Dataset("enwik.test", file_dest)
    # print(enwik9_train.data[0:100])
    # print(enwik9_valid.data[0:100])
    # print(enwik9_test.data[0:100])


    batch_size = 64
    seq_length = 70
    buffer_size = 10000
    lstm_baseline = models.LSTMBaseLine(len(ptb_word_train.char2idx), seq_length, batch_size, tie_embedding=True)
    ptb_word_train.convert_text_to_int(). \
        convert_to_tensor_dataset(). \
        batch(seq_length + 1, drop_remainder=True). \
        map(split_input_target). \
        shuffle(buffer_size). \
        batch(batch_size, drop_remainder=True)

    # print(lstm_baseline.model.summary())

    # for input_example_batch, target_example_batch in ptb_word_train.data.take(1):
    #     print(tf.shape(input_example_batch))
    #     print(target_example_batch)
    #     example_batch_predictions = lstm_baseline.lstm(input_example_batch)
    #     print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    # print(lstm_baseline.model.summary())
    # sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    # sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    # print(sampled_indices)
    # print("Input: \n", repr("".join(ptb_word_train.idx2char[input_example_batch[0]])))
    # print()
    # print("Next Char Predictions: \n", repr("".join(ptb_word_train.idx2char[sampled_indices])))

    # example_batch_loss = loss(target_example_batch, example_batch_predictions)
    # print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    # print("scalar_loss:      ", example_batch_loss.numpy().mean())

    # for input_example_batch, target_example_batch in ptb_word_train.data.take(2):
    #     # print(tf.shape(lstm_baseline.model(input_example_batch)))
    #     # print(lstm_baseline.model(input_example_batch))
    #     lstm_baseline.model(input_example_batch)

    # print("weight matrix for input embedding: {}".format(lstm_baseline.model.layers[0].weights[0]))
    # print(len(lstm_baseline.model.layers[-1].weights)) # seems like there's an extra copy of input embedding weight matrix, length returned 3 instead of 2
    # print(tf.reduce_all(tf.equal(lstm_baseline.model.layers[-1].weights[1], lstm_baseline.model.layers[0].weights[0])))
    # print(lstm_baseline.model.layers[-1].weights[0])
    # print(lstm_baseline.model.layers[-1].weights[1])
    # lstm_baseline.model.load_weights(os.path.join('training_checkpoints', 'ckpt_10.h5'))

    optimizer = tf.keras.optimizers.Adam()
    fit(lstm_baseline.model, ptb_word_train.data, optimizer)


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
    # mogrify(x, h_0, q, r)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

@tf.function
def train_step(inp, target, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.exp(tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True)))
    grads, _ = tf.clip_by_global_norm(tape.gradient(loss, model.trainable_variables), 10.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def fit(model, dataset, optimizer):
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    # Training step
    EPOCHS = 10
    for epoch in range(EPOCHS):
        start = time.time()

        # initializing the hidden state at the start of every epoch
        # initally hidden is None
        hidden = model.reset_states()

        for (batch_n, (inp, target)) in enumerate(dataset):
            loss = train_step(inp, target, model, optimizer)

            if batch_n % 100 == 0:
                template = 'Epoch {} Batch {} Loss {}'
                print(template.format(epoch + 1, batch_n, loss))

        # saving (checkpoint) the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.save_weights(checkpoint_prefix.format(epoch=epoch))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    model.save_weights(checkpoint_prefix.format(epoch=EPOCHS))
    # check that the input embedding weight matrix is the same as the weight matrix in last dense layer
    print(tf.reduce_all(tf.equal(model.layers[-1].weights[1], model.layers[0].weights[0])))

if __name__ == '__main__':
    main()
