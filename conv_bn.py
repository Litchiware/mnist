import config
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
from batch_norm import batch_norm

def conv_relu_pool(data, kernel_shape, bias_shape, is_training):
    weights = tf.get_variable("weights", kernel_shape,
            initializer=tf.truncated_normal_initializer(
                stddev=0.1, seed=config.SEED, dtype=tf.float32))
    biases = tf.get_variable("biases", bias_shape,
            initializer=tf.constant_initializer(0.0))

    conv = tf.nn.bias_add(
            tf.nn.conv2d(data, weights, strides=[1, 1, 1, 1], padding='SAME'),
            biases)
    relu = tf.nn.relu(batch_norm(conv, is_training))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return pool


def model(data, is_training):
    """The Model definition."""
    with tf.variable_scope('conv1'):
        net = conv_relu_pool(data, [5, 5, config.NUM_CHANNELS, 32], [32], is_training)

    with tf.variable_scope('conv2'):
        net = conv_relu_pool(net, [5, 5, 32, 64], [64], is_training)

    reshape = tf.reshape(net, [-1, 64 * 7 * 7])

    with tf.variable_scope('hidden1'):
        weights = tf.get_variable('weights', [7 * 7 * 64, 512],
                initializer=tf.truncated_normal_initializer(stddev=0.1, seed=config.SEED, dtype=tf.float32))
        biases = tf.get_variable('biases', [512],
                initializer=tf.constant_initializer(0.1))
    
    tf.add_to_collection('regularizers', tf.nn.l2_loss(weights))
    tf.add_to_collection('regularizers', tf.nn.l2_loss(biases))

    net = tf.nn.relu(tf.matmul(reshape, weights) + biases)

    with tf.variable_scope('hidden2'):
        weights = tf.get_variable('weights',
                [512, 10],
                initializer=tf.truncated_normal_initializer(stddev=0.1, seed=config.SEED, dtype=tf.float32))
        biases = tf.get_variable('biases', [config.NUM_LABELS],
                initializer=tf.constant_initializer(0.1))

    tf.add_to_collection('regularizers', tf.nn.l2_loss(weights))
    tf.add_to_collection('regularizers', tf.nn.l2_loss(biases))

    net = tf.matmul(net, weights) + biases

    return net


def main(argv=None): # pylint: disable=unused-argument
    mnist = read_data_sets(config.WORK_DIRECTORY, one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    is_training = tf.placeholder(tf.bool)

    images = tf.reshape(x, [-1, 28, 28, 1])
    logits = model(images, is_training)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))

    # loss += 5e-4 * tf.add_n(tf.get_collection('regularizers'))

    batch = tf.Variable(0, tf.float32)
    learning_rate = tf.train.exponential_decay(
            0.01,
            batch * config.BATCH_SIZE,
            mnist.train.num_examples,
            0.95,
            staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

    preds = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y, 1))
    accs = tf.reduce_mean(tf.cast(preds, 'float'))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    start_time = time.time()
    for epoch_i in range(config.NUM_EPOCHS):
        for batch_i in range(mnist.train.num_examples / config.BATCH_SIZE):
            batch_xs, batch_ys = mnist.train.next_batch(config.BATCH_SIZE)
            _, l, lr, train_acc = sess.run([optimizer, loss, learning_rate, accs],
                    feed_dict={x: batch_xs, y: batch_ys, is_training: True})

        eval_acc = sess.run(accs,
                feed_dict={x: mnist.validation.images, y: mnist.validation.labels, is_training: False})

        elapsed_time = time.time() - start_time
        start_time = time.time()
        print 'Minibatch loss: %.3f, learning rate: %.6f' % (l, lr)
        print 'Minibatch accuracy: %.3f' % train_acc
        print 'Epoch %d, %.3f s' % (epoch_i, elapsed_time)
        print 'Validation accuracy: %.3f' % eval_acc


if __name__ == '__main__':
    tf.app.run()
