import tensorflow as tf


class Model(object):

    def __init__(self, FLAGS):
        self.image_width = FLAGS.image_width
        self.image_height = FLAGS.image_height
        self.num_image_channels = FLAGS.num_image_channels
        self.batch_size = FLAGS.batch_size
        self.num_epochs = FLAGS.num_epochs
        self.num_classes = FLAGS.num_classes
        self.growth_rate = FLAGS.growth_rate
        self.num_layers_in_dense_block = int(FLAGS.num_layers - 4 / 3) if int(FLAGS.num_layers - 4 / 3) > 0 else 12
        self.learning_rate = FLAGS.learning_rate
        self.dropout_prob = FLAGS.dropout_prob
        self.compression = FLAGS.compression
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.image_width * self.image_height * self.num_image_channels], name='inputs')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.num_classes])
        self.correct_predictions, self.loss, self.accuracy, self.optimizer = self.net()

    def conv_layer(self, x):
        return tf.layers.conv2d(inputs=x, filters=16, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    def dense_block(self, p):
        for i in range(self.num_layers_in_dense_block):
            with tf.variable_scope('bottle_neck{0}'.format(i)):
                x = tf.layers.batch_normalization(p)
                x = tf.nn.relu(x)
                x = tf.layers.conv2d(inputs=x, filters=self.growth_rate, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01))
                x = tf.concat([x, p], axis=3)
                p = x
        return x

    def transition_layer(self, x):
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        n_inchannels = x.get_shape().as_list()[3]
        n_outchannels = int(n_inchannels * self.compression)
        x = tf.layers.conv2d(inputs=x, filters=n_outchannels, kernel_size=1, strides=1, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        x = tf.layers.average_pooling2d(inputs=x, pool_size=2, strides=2)
        return x

    def net(self):
        x = tf.reshape(self.inputs, shape=[-1, self.image_width, self.image_height, self.num_image_channels])
        c = self.conv_layer(x)
        with tf.variable_scope('dense_block1') as scope:
            b1 = self.dense_block(c)

        with tf.variable_scope('transition1') as scope:
            t1 = self.transition_layer(b1)

        with tf.variable_scope('dense_block2') as scope:
            b2 = self.dense_block(t1)

        with tf.variable_scope('transition2') as scope:
            t2 = self.transition_layer(b2)

        with tf.variable_scope('dense_block3') as scope:
            b3 = self.dense_block(t2)

        b = tf.layers.batch_normalization(b3)
        r = tf.nn.relu(b)
        g = tf.layers.average_pooling2d(r, pool_size=[b3.get_shape().as_list()[1], b3.get_shape().as_list()[2]], strides=1)
        g = tf.squeeze(g)
        logits = tf.layers.dense(g, self.num_classes, activation='linear')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
        correct_predictions = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.labels, axis=1), tf.argmax(logits, axis=1)), tf.float32))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels, axis=1), tf.argmax(logits, axis=1)), tf.float32))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        return correct_predictions, loss, accuracy, optimizer



