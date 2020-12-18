import argparse
import tensorflow as tf
import os
import glob
import librosa
from typing import List, Tuple
import numpy as np


class PCAGAN(tf.keras.Model):

    def __init__(self, receptive_field_size: int, batch_size: int, num_units_h1: int, activation: tf.nn.relu):
        super(PCAGAN, self).__init__()
        self.batch_size = batch_size
        # Default size (10, 480086):
        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=receptive_field_size, batch_size=self.batch_size, name='input_layer'
        )
        self.hidden_layer_one = tf.keras.layers.Dense(
            units=num_units_h1, use_bias=True, kernel_initializer='random_uniform', bias_initializer='ones',
            kernel_regularizer=None, kernel_constraint=None, bias_constraint=None, name='h1', activation=activation
        )

    def call(self, inputs, training = None, mask = None):
        x = self.input_layer(inputs, input_shape=inputs.shape, batch_size=self.batch_size)
        return self.hidden_layer_one(x)



def main(args):
    data_file: str = args.data_file
    is_debug: bool = args.is_verbose
    cwd = os.getcwd()
    if not os.path.isfile(data_file):
        raise FileNotFoundError('The provided root data file path \'%s\' is invalid!' % data_file)
    with np.load(data_file) as npz_file:
        file_name: str = npz_file.files[0]
        X = npz_file[file_name]
        if is_debug:
            print('sample data shape: %s' % (X.shape, ))
    # TODO: Do the train-test-val partition here (after obtaining more data):
    # train_dataset = tf.data.Dataset.from_tensor_slices(X)
    train_data = X
    pca_gan_model = PCAGAN(
        receptive_field_size=train_data.shape[1],
        batch_size=10,
        num_units_h1=10,
        activation=tf.nn.relu
    )
    pca_gan_model.compile(
        optimizer=tf.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'),
        loss='mse', metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None
    )
    pca_gan_model.fit(
        x=train_data,
        y=None,
        batch_size=10,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )
    if is_debug:
        print(pca_gan_model.summary())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCGAN argument parser.')
    parser.add_argument(
        '--data-file', action='store', type=str, dest='data_file', required=True,
        help='The location of the transformed (down-sampled or up-sampled) audio data stored as a .npz file.'
    )
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--num-units-h1', type=int, nargs=1, action='store', dest='num_units_h1', required=True,
                        help='The number of neurons in the first hidden layer, the width of the first hidden layer.')
    # parser.add_argument('--beemon-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    args = parser.parse_args()
    main(args=args)
    # print('num_neurons: %s' % args.num_neurons)
