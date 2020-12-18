import os
import argparse
import copy
import numpy as np
import tensorflow as tf


def make_pca_gan_model(receptive_field_size: int, train_batch_size: int, num_units_h1: int, activation_h1: tf.nn):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=receptive_field_size, batch_size=train_batch_size, name='input_layer'),
        tf.keras.layers.Dense(units=num_units_h1, use_bias=True, kernel_initializer='random_uniform',
                              bias_initializer='ones', kernel_regularizer=None, kernel_constraint=None,
                              bias_constraint=None, activation=activation_h1, name='h1'),
        tf.keras.layers.Dense(units=receptive_field_size, batch_size=train_batch_size, name='output_layer')
        ]
    )
    return model


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
    train_data, train_targets = X, copy.deepcopy(X)
    # Create the model:
    pca_gan_model = make_pca_gan_model(
        receptive_field_size=train_data.shape[1],
        train_batch_size=train_data.shape[0],
        num_units_h1=2,
        activation_h1=None
    )
    if is_debug:
        pca_gan_model.summary()
    # Compile the model:
    pca_gan_model.compile(
        optimizer='rmsprop',
        loss='MSE',
        metrics=['MSE'],
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None
    )
    # Fit the model to the training data:
    pca_gan_model.fit(
        x=train_data,
        y=train_targets,
        batch_size=train_data.shape[0],
        epochs=10,
        verbose=2,
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=1,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )


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