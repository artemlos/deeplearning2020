import tensorflow as tf

rounds = 5
k = int((90 + 40) / 2) # k < min(m,n)

def mogrify(x_init, h_0, q, r, rounds):
    x = x_init
    h_prev = h_0
    for i in range(1, rounds+1):
        if i % 2 != 0:
            x = 2 * tf.sigmoid(tf.matmul(q[i-1], h_prev)) * x
            # print("odd %s, %s" % (i, tf.shape(x)))
        else:
            h_prev = 2 * tf.sigmoid(tf.matmul(r[i-1], x)) * h_prev
            # print("even %s, %s" % (i, tf.shape(h_prev)))

    return x, h_prev

def matrix_decomposition(m, n, k):
    # currently not decomposition
    # k < min(m,n)

    q_left = tf.random.normal([m, k])
    q_right = tf.random.normal([k, n])

    r_left = tf.random.normal([n, k])
    r_right = tf.random.normal([k, m])

    return tf.matmul(q_left, q_right), tf.matmul(r_left, r_right)