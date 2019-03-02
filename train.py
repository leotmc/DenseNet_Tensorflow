import tensorflow as tf
from densenet import Model
from utils import DataProcessor

tf.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.flags.DEFINE_integer('num_epochs', 20, 'number of epochs')
tf.flags.DEFINE_integer('densenet_length', 40, 'lenght of densenet')
tf.flags.DEFINE_integer('image_width', 32, 'the width of the image')
tf.flags.DEFINE_integer('image_height', 32, 'the height of the image')
tf.flags.DEFINE_integer('num_image_channels', 3, 'number of channels of the image')
tf.flags.DEFINE_integer('num_classes', 10, 'number of classes')
tf.flags.DEFINE_integer('growth_rate', 12, 'growth rate')
tf.flags.DEFINE_integer('num_layers', 40, 'number of layers in densenet')
tf.flags.DEFINE_float('learning_rate', 1.0e-3, 'learning rate')
tf.flags.DEFINE_float('dropout_prob', 0.5, 'dropout probability')
tf.flags.DEFINE_float('compression', 0.5, 'compression ratio')
tf.flags.DEFINE_string('train_path', 'cifar-10-batches-py/data_batch_1', 'training data path')
tf.flags.DEFINE_string('test_path', 'cifar-10-batches-py/test_batch', 'test data path')
FLAGS = tf.flags.FLAGS


data_processor = DataProcessor(FLAGS)
data = data_processor.load_data(FLAGS.train_path)
test_data = data_processor.load_data(FLAGS.test_path)
model = Model(FLAGS)


def evaluate():
    total_correct_predictions = 0
    num_test_samples = len(test_data[b'labels'])
    for inputs, labels in data_processor.data_generator(test_data):
        batch_correct_predictions = sess.run(model.correct_predictions, feed_dict={model.inputs: inputs, model.labels: labels})
        total_correct_predictions += batch_correct_predictions
    test_accuracy = 1.0 * total_correct_predictions / num_test_samples
    return test_accuracy


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 0
    for inputs, labels in data_processor.data_generator(data):
        feed_dict = {
            model.inputs: inputs,
            model.labels: labels,
        }
        _, loss, accuracy = sess.run([model.optimizer, model.loss, model.accuracy], feed_dict=feed_dict)
        print(loss, accuracy)
        i += 1
        if i % 100 == 0:
            test_accuracy = evaluate()
            print(test_accuracy)
