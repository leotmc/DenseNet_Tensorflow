import pickle
import numpy as np


class DataProcessor(object):

    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size
        self.num_epochs = FLAGS.num_epochs
        self.num_classes = FLAGS.num_classes

    def load_data(self, fname):
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        return data

    def data_generator(self, data):
        num_batches = int(len(data[b'labels']) / self.batch_size)
        for epoch_id in range(self.num_epochs):
            for batch_id in range(num_batches):
                inputs = data[b'data'][batch_id * self.batch_size: (batch_id + 1) * self.batch_size]
                labels = data[b'labels'][batch_id * self.batch_size: (batch_id + 1) * self.batch_size]
                labels = np.eye(self.num_classes)[labels]
                yield inputs, labels
