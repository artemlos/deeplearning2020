import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import tensorflow as tf

# use gpu
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# print(tf.config.list_physical_devices('GPU'))
print("GPU built with CUDA: %s" % tf.test.is_built_with_cuda())
print(tf.reduce_sum(tf.random.normal([1000, 1000])))


# load data
train = pd.read_csv('datasets/mitbih_train.csv', low_memory=False, header=None)
test = pd.read_csv('datasets/mitbih_test.csv', low_memory=False, header=None)

## Split into X and y

X_train = train.to_numpy()[:, 0:186]
y_train = train.to_numpy()[:, -1]

X_train = X_train.reshape(-1, 186, 1)
X_train_normal = X_train[y_train == 0]
X_train_anomaly = X_train[y_train == 1]
X_train_anomaly2 = X_train[y_train == 2]
X_train_anomaly3 = X_train[y_train == 3]
X_train_anomaly4 = X_train[y_train == 4]

# TEST data
X_test = test.to_numpy()[:, 0:186].reshape(-1, 186, 1)
y_test = test.to_numpy()[:, -1]

X_test_normal = X_test[y_test == 0]
X_test_anomaly = X_test[y_test == 1]
X_test_anomaly2 = X_test[y_test == 2]
X_test_anomaly3 = X_test[y_test == 3]
X_test_anomaly4 = X_test[y_test == 4]


def LSTM_AE(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    encoded = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)(inputs)
    encoded = tf.keras.layers.LSTM(16, activation='tanh', return_sequences=False)(encoded)

    decoded = tf.keras.layers.RepeatVector(input_shape[0])(encoded)
    decoded = tf.keras.layers.LSTM(16, activation='tanh', return_sequences=True)(decoded)
    decoded = tf.keras.layers.LSTM(32, activation='tanh', return_sequences=True)(decoded)
    decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_shape[1]))(decoded)

    sequence_autoencoder = tf.keras.Model(inputs, decoded)

    return sequence_autoencoder


def train(model):

    # Dataset
    # We use the ECG dataset available [here](https://www.kaggle.com/shayanfazeli/heartbeat/data?select=mitbih_train.csv). There is [an article](https://arxiv.org/pdf/1805.00794.pdf) that uses the dataset, which we can use as a reference. More details about the dataset can be found [here](https://physionet.org/content/apnea-ecg/1.0.0/)
    # 1. Train only on normal cases. --> determine average error.
    # 2. Use the error as the threshold for decisions
    # 3. Check if the error differs significantly for other classes.

    checkpoint_dir = "run"
    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(checkpoint_dir, "log2.csv"))  # to save epoch results in csv file
    model.fit(X_train_normal, X_train_normal, epochs=100, verbose=1, shuffle=True,
              validation_split=0.05, callbacks=[csv_logger])

    # important to change file name for new model configs, so we dont overwrite existing files
    model.save_weights(os.path.join(checkpoint_dir, "final_weights2.h5"))

    ### Apply model on train

    return model


def plot_results(model):
    # We will now compute the the loss for each class on the train set. This will guide us in setting a decision boundary, that we can later use on the test set.
    x_train_pred = model.predict(X_train_normal)
    train_mae_loss = np.mean(np.abs(x_train_pred - X_train_normal), axis=1)

    plt.hist(train_mae_loss, bins=10)
    plt.xlabel("Train MAE loss")
    plt.ylabel("No of samples")
    plt.show()

    print(np.max(train_mae_loss))

    # print(model.evaluate(X_train_normal,X_train_normal, verbose=3))
    # print(model.evaluate(X_train_anomaly2,X_train_anomaly2, verbose=3))
    # print(model.evaluate(X_train_anomaly3,X_train_anomaly3, verbose=3))
    # print(model.evaluate(X_train_anomaly4,X_train_anomaly4, verbose=3))
    ### Conclusion and test set
    # It's clear that our model (trained on fewer samples than we have) detects most of the nomalies (except class 3). Now, let's do the same on the test set. We can set the decision boundary to `0.16`.

    # print(model.evaluate(X_test_normal,X_test_normal, verbose=3))
    # print(model.evaluate(X_test_anomaly,X_test_anomaly, verbose=3))
    # print(model.evaluate(X_test_anomaly2,X_test_anomaly2, verbose=3))
    # print(model.evaluate(X_test_anomaly3,X_test_anomaly3, verbose=3))
    # print(model.evaluate(X_test_anomaly4,X_test_anomaly4, verbose=3))

    x_normal_pred = model.predict(X_test_normal)
    normal_mae_loss = np.mean(np.abs(x_normal_pred - X_test_normal), axis=1)
    x_anomaly_pred = model.predict(X_test_anomaly)
    anomaly_mae_loss = np.mean(np.abs(x_anomaly_pred - X_test_anomaly), axis=1)
    x_anomaly2_pred = model.predict(X_test_anomaly2)
    anomaly2_mae_loss = np.mean(np.abs(x_anomaly2_pred - X_test_anomaly2), axis=1)
    x_anomaly3_pred = model.predict(X_test_anomaly3)
    anomaly3_mae_loss = np.mean(np.abs(x_anomaly3_pred - X_test_anomaly3), axis=1)
    x_anomaly4_pred = model.predict(X_test_anomaly4)
    anomaly4_mae_loss = np.mean(np.abs(x_anomaly4_pred - X_test_anomaly4), axis=1)

    error_threshold = 0.15


    # test set with normal samples
    # "positive" means it's normal. "negative" means there is an anomaly.
    def compute_metrics(error_threshold):
        FN = len(normal_mae_loss[normal_mae_loss >= error_threshold])
        TP = len(normal_mae_loss[normal_mae_loss < error_threshold])

        FP = len(anomaly_mae_loss[anomaly_mae_loss < error_threshold]) + \
             len(anomaly2_mae_loss[anomaly2_mae_loss < error_threshold]) + \
             len(anomaly3_mae_loss[anomaly3_mae_loss < error_threshold]) + \
             len(anomaly4_mae_loss[anomaly4_mae_loss < error_threshold])

        TN = len(anomaly_mae_loss[anomaly_mae_loss >= error_threshold]) + \
             len(anomaly2_mae_loss[anomaly2_mae_loss >= error_threshold]) + \
             len(anomaly3_mae_loss[anomaly3_mae_loss >= error_threshold]) + \
             len(anomaly4_mae_loss[anomaly4_mae_loss >= error_threshold])

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_score = 2 * (precision * recall) / (precision + recall)
        accurracy = (TP + TN) / (TP + TN + FP + FN)

        print(precision, "precision")
        print(recall, "recall")
        print(F1_score, "F1 score")
        print(accurracy, "accurracy")


    print("Compute metrics with {} error threshold".format(error_threshold))
    compute_metrics(error_threshold)

    print()
    max_error_threshold = np.max(train_mae_loss)
    print("Compute metrics with {} error threshold".format(max_error_threshold))
    compute_metrics(max_error_threshold)

def main():
    model = LSTM_AE((186, 1))
    model.summary()
    model.compile(optimizer='adam', loss='mae', metrics=None)

    load_weights = False

    if load_weights:
        model.load_weights(os.path.join("run", "final_weights1.h5"))
    else:
        model = train(model)

    plot_results(model)

main()
