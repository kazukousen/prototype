import numpy as np
import tensorflow as tf
import time

class VAE_BP(object):
    def __init__(self, in_dim=784, hidden_units=[512, 256], latent_dim=50, lr=0.001):
        self.in_dim = in_dim
        self.hidden_units = hidden_units
        self.latent_dim = latent_dim
        self.lr = tf.constant(lr, dtype=tf.float32)

        self.x = tf.placeholder(tf.float32, [None, self.in_dim])

        encode_hidden = []

        # Input Layer
        with tf.name_scope('input'):
            weights = tf.Variable(self._xavier_init(self.in_dim, self.hidden_units[0]), name='weights')
            biases = tf.Variable(tf.zeros([self.hidden_units[0]], dtype=tf.float32), name='biases')
            input = tf.nn.relu(tf.add(tf.matmul(self.x, weights), biases))

        # Encode Layer
        for index, n_hidden in enumerate(self.hidden_units):
            if index == len(self.hidden_units) - 1: break
            with tf.name_scope('encode_hidden{}'.format(index+1)):
                weights = tf.Variable(self._xavier_init(n_hidden, self.hidden_units[index+1]), name='weights')
                biases = tf.Variable(tf.zeros([self.hidden_units[index+1]], dtype=tf.float32), name='biases')
                inputs = input if index == 0 else encode_hidden[index-1]
                encode_hidden.append(tf.nn.relu(tf.add(tf.matmul(inputs, weights), biases)))

        # Latent Layer
        with tf.name_scope('latent_in'):
            weights = tf.Variable(self._xavier_init(self.hidden_units[-1], self.latent_dim), name='weights')
            biases = tf.Variable(tf.zeros([self.latent_dim], dtype=tf.float32), name='biases')
            inputs = encode_hidden[-1] if encode_hidden else input
            self.mean = tf.add(tf.matmul(inputs, weights), biases)
            self.log_var = tf.add(tf.matmul(inputs, weights), biases)

        decode_hidden = []
        reverse_hidden_units = [x for x in reversed(self.hidden_units)]

        # sampling
        self.epsilon = tf.random_normal(tf.shape(self.mean), mean=0, stddev=1, dtype=tf.float32)
        self.z = tf.add(self.mean, tf.mul(tf.sqrt(tf.exp(self.log_var)), self.epsilon))

        # Latent Layer
        with tf.name_scope('latent_out'):
            weights = tf.Variable(self._xavier_init(self.latent_dim, reverse_hidden_units[0]), name='weights')
            biases = tf.Variable(tf.zeros([reverse_hidden_units[0]], dtype=tf.float32), name='biases')
            latent_out = tf.nn.relu(tf.add(tf.matmul(self.z, weights), biases))

        # Decode Layer
        for index, n_hidden in enumerate(reverse_hidden_units):
            if index == len(self.hidden_units) - 1: break
            with tf.name_scope('decode_hidden{}'.format(index+1)):
                weights = tf.Variable(self._xavier_init(n_hidden, reverse_hidden_units[index+1]), name='weights')
                biases = tf.Variable(tf.zeros([reverse_hidden_units[index+1]], dtype=tf.float32), name='biases')
                inputs = latent_out if index == 0 else decode_hidden[index-1]
                decode_hidden.append(tf.nn.relu(tf.add(tf.matmul(inputs, weights), biases)))

        # Reconstruct Layer
        with tf.name_scope('reconstruct'):
            weights = tf.Variable(self._xavier_init(reverse_hidden_units[-1], self.in_dim), name='weights')
            biases = tf.Variable(tf.zeros([self.in_dim], dtype=tf.float32), name='biases')
            inputs = decode_hidden[-1] if decode_hidden else latent_out
            self.reconstruct_x = tf.nn.sigmoid(tf.add(tf.matmul(inputs, weights), biases))

        # define loss and launch session
        self._compile()

    def _xavier_init(self, fan_in, fan_out):
        low = -np.sqrt(6.0/(fan_in + fan_out))
        high = np.sqrt(6.0/(fan_in + fan_out))

        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

    def _binary_crossentropy(self, y_true, y_pred, eps=1e-20):
        return tf.reduce_sum(y_true*tf.log(y_pred + eps) + (1.0 - y_true)*tf.log(1.0 - y_pred + eps), 1)

    def _KL(self, mean, log_var):
        return -0.5*tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), 1)

    def _loss(self):
        KL = self._KL(self.mean, self.log_var)
        reconstruct_loss = self._binary_crossentropy(self.x, self.reconstruct_x)
        elbo = -KL + reconstruct_loss
        loss = tf.reduce_mean(-elbo)
        return loss

    def _compile(self):
        # optimizer
        self.loss = self._loss()
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # initialize weights
        init = tf.initialize_all_variables()

        # launch the session
        self.sess = tf.Session()
        self.sess.run(init)

    def fit(self, x_train=None, x_valid=None, batch_size=100, n_epochs=50, lr=0.001):
        # prepare batch data
        batch_xs = []
        for index in range(len(x_train)/batch_size):
            batch_xs.append(x_train[index*batch_size : (index+1)*batch_size])

        # train
        for epoch in range(1, n_epochs+1):
            start_time = time.time()
            np.random.shuffle(batch_xs)
            for i in range(len(x_train)/batch_size):
                if batch_xs[i] is None : break
                _, np_loss = self.sess.run([self.train_op, self.loss], feed_dict={self.x: batch_xs[i], self.lr: lr})
            np_val_loss = self.sess.run(self.loss, feed_dict={self.x: x_valid})
            print('Epoch {0:d}, loss: {1:0.3f}, val_loss: {2:0.3f}, time: {3:0.3f}'.format(epoch, -np_loss, -np_val_loss, time.time()-start_time))

    def predict(self, x_data=None):
        return self.sess.run(self.reconstruct_x, feed_dict={self.x: x_data})

