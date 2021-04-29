import os
from typing import Optional
import argparse
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
from Utils.TensorFlow.TFRecordLoader import TFRecordLoader
from PCAGAN.Encoder import SVDAutoencoder, create_sequential_svd_autoencoder_model
# number of units in the hidden layer:
HP_NUM_UNITS_LATENT_SPACE = hp.HParam('num_units_latent', hp.Discrete([2, 3]))

HPARAMS = [
    HP_NUM_UNITS_LATENT_SPACE
]

METRIC_MSE = 'MSE'
METRIC_RMSE = 'RMSE'

METRICS = [
    hp.Metric(
        "epoch_loss",
        group='train',
        display_name='loss/MSE (train)'
    ),
    hp.Metric(
        "epoch_RMSE",
        group='train',
        display_name='RMSE (train)'
    )
]

DEFAULT_NUM_EPOCHS = 10


# class SVDAutoencoderHparamOptimizer:
#
#     def __init__(self, root_data_dir: str, output_data_dir: str, is_debug: bool, num_frequency_bins: int,
#                  share_weights: bool):
#         self._root_data_dir = root_data_dir
#         self._output_data_dir = output_data_dir
#         self._tensorboard_output_dir = os.path.join(self._output_data_dir, 'TensorBoard')
#         assert os.path.isdir(self._root_data_dir), self._root_data_dir
#         assert os.path.isdir(self._output_data_dir), self._output_data_dir
#         self._is_debug: bool = is_debug
#         self._num_frequency_bins: int = num_frequency_bins
#         self._share_weights: bool = share_weights
#         self._tf_record_loader = TFRecordLoader(
#             root_data_dir=self._root_data_dir,
#             is_debug=self._is_debug
#         )
#
#     def perform_hparams_grid_search(self, batch_size: int, num_epochs: int):
#         #
#         # Compile the model:
#         self._svd_autoencoder.compile(
#             optimizer=tf.optimizers.Adam(learning_rate=0.001),
#             loss=tf.keras.losses.MeanSquaredError(),
#             metrics=[tf.keras.metrics.RootMeanSquaredError()]
#         )
#         self._svd_autoencoder.build(input_shape=(batch_size, self._num_frequency_bins))
#         if self._is_debug:
#             print(self._svd_autoencoder.summary())
#             '''
#             Create Tensorflow HParams file writer object for tensorboard (see
#              https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams):
#             '''
#             with tf.summary.create_file_writer(self._tensorboard_output_dir).as_default():
#                 hp.hparams_config(
#                     hparams=[HP_NUM_UNITS_LATENT_SPACE],
#                     metrics=[hp.Metric(METRIC_MSE, display_name='loss'), hp.Metric(METRIC_RMSE, display_name=METRIC_RMSE)]
#                 )


def main(args):
    # Command line arguments:
    is_debug: bool = args.is_verbose
    root_data_dir: str = args.root_data_dir[0]
    output_data_dir: str = args.output_data_dir[0]
    train_batch_size: int = args.train_batch_size
    order_deterministically: bool = args.order_deterministically
    num_frequency_bins: int = args.num_freq_bins    # 4097 by default.
    num_epochs: int = args.num_epochs
    share_weights: bool = args.share_weights

    dataset_split_type: DatasetSplitType

    # Ensure that the provided arguments are valid:
    if not os.path.isdir(root_data_dir):
        raise FileNotFoundError('The provided root data directory \'%s\' is invalid!' % root_data_dir)
    else:
        os.chdir(root_data_dir)
    if not os.path.isdir(output_data_dir):
        raise FileNotFoundError('The provided output data directory: \'%s\' is invalid!' % output_data_dir)
    tensorboard_output_dir = os.path.join(output_data_dir, 'TensorBoard')
    hparam_tensorboard_output_dir = os.path.join(tensorboard_output_dir, 'hparam_tuning')

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Configure tensorboard summary writers (see: https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams#1_experiment_setup_and_the_hparams_experiment_summary)
    with tf.summary.create_file_writer(hparam_tensorboard_output_dir).as_default():
        hp.hparams_config(
            hparams=HPARAMS,
            metrics=METRICS
        )

    for i, num_units in enumerate(HP_NUM_UNITS_LATENT_SPACE.domain.values):
        # Specify the run name:
        run_name: str = 'run_%d' % i
        run_tensorboard_output_dir = os.path.join(hparam_tensorboard_output_dir, run_name)

        # Subset the dictionary of hyperparameters by the current run/iteration's parameters:
        hparams = {
            HP_NUM_UNITS_LATENT_SPACE: num_units
        }

        if is_debug:
            print('Trial: %s with hparams: %s' % (run_name, {h.name:  hparams[h] for h in hparams}))
            # Configure TensorBoard callbacks:
            hparams_keras_callback: hp.KerasCallback = hp.KerasCallback(
                writer=run_tensorboard_output_dir,
                hparams=hparams
            )
            tb_metrics_callback: tf.keras.callbacks.TensorBoard = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(hparam_tensorboard_output_dir, 'metrics'),
                # profile_batch='15, 30'
            )

        # Create the model:
        svd_autoencoder = create_sequential_svd_autoencoder_model(
            num_frequency_bins=num_frequency_bins,
            num_units_latent_dim=hparams[HP_NUM_UNITS_LATENT_SPACE],
            share_weights=share_weights
        )

        # Compile the model:
        svd_autoencoder.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

        # Build the model:
        svd_autoencoder.build(input_shape=(train_batch_size, num_frequency_bins))

        # Optionally print the model:
        if is_debug:
            # Print the model:
            print(svd_autoencoder.summary())

        # Load the data sets:
        tf_record_loader = TFRecordLoader(
            root_data_dir=root_data_dir,
            is_debug=is_debug
        )
        train_tf_record_ds: tf.data.TFRecordDataset = tf_record_loader.get_batched_tf_record_dataset(
            dataset_split_type=DatasetSplitType.TRAIN,
            order_deterministically=order_deterministically,
            batch_size=train_batch_size,
            prefetch=True,
            cache=False
        )
        val_tf_record_ds: tf.data.TFRecordDataset = tf_record_loader.get_batched_tf_record_dataset(
            dataset_split_type=DatasetSplitType.VAL,
            order_deterministically=order_deterministically,
            batch_size=train_batch_size,
            prefetch=True,
            cache=False
        )

        # Fit the model:
        '''
        Some of the below parameter choices for the .fit() function are not self explanatory, hence the rational for
        those choices are described below as well as in the documentation (here 
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).
        :param batch_size: This parameter is set to None because we are using TFRecord datasets which dictate their own
         batch size (in our case sourced from the command line arguments). 
        :param shuffle: A boolean value indicates whether the training data should be shuffled before each epoch, or 
         each batch. We already shuffle the data in the preprocessing step by permuting each of the TFRecord datasets
         randomly. Hence, we do not shuffle the data again due to performance overhead.
        :param steps_per_epoch: Total number of steps (batches of samples) before declaring one epoch finished and 
         starting the next epoch. A value of None defaults to the number of samples in the dataset divided by the 
         batch size of the dataset generator.
        :param validation_steps: The total number of steps (batches of samples) to draw before stopping when performing
         validation at the end of every epoch. We provide a value of None to indicate that validation should run until
         the entire validation dataset has been leveraged.
        :param validation_batch_size: We provide a value of None because we are using TFRecord datasets which dictate 
         their own batch size (in our case sourced from the command line arguments).
        :param validation_freq: When provided as an integer, specifies how many training epochs to run before performing
         a validation run. We specify with a value of 1 that the validation metrics should be computed after every 
         training epoch. 
        '''
        if is_debug:
            tb_callbacks = [hparams_keras_callback, tb_metrics_callback]
        else:
            tb_callbacks = None
        svd_autoencoder.fit(
            train_tf_record_ds,
            batch_size=None,
            epochs=num_epochs,
            verbose=1,
            callbacks=tb_callbacks,
            validation_data=val_tf_record_ds,
            shuffle=False,
            class_weight=None,
            sample_weight=None,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1
        )

        # Save the trained model:
        saved_model_dir = os.path.join(run_tensorboard_output_dir, 'SavedModel')
        svd_autoencoder.save(filepath=saved_model_dir, overwrite=True, include_optimizer=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder argument parser.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--root-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    parser.add_argument('--order-deterministically', type=bool, action='store', dest='order_deterministically',
                        required=True)
    parser.add_argument('--output-data-dir', type=str, nargs=1, action='store', dest='output_data_dir', required=True)
    parser.add_argument('--train-batch-size', type=int, action='store', dest='train_batch_size', required=True,
                        help='The batch size for use during the training of the Autoencoder.')
    parser.add_argument(
        '--num-freq-bins', type=int, action='store', dest='num_freq_bins', default=4097, required=False,
        help='The number of frequency bins for a single sample, by default this corresponds to: %d frequency bins per '
             'spectrum.' % 4097
    )
    parser.add_argument('--num-epochs', type=int, action='store', dest='num_epochs', required=False, default=DEFAULT_NUM_EPOCHS, help='Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.')
    parser.add_argument('--share-weights', type=bool, action='store', dest='share_weights', required=True,
                        help='A boolean flag indicating if the Encoder and Decoder in the Autoencoder should share the '
                             'same weight matrix. If set to false, the Encoder and Decoder will have their own separate'
                             ' weight matrices. If set to true, the Encoder and Decoder will share the same weight '
                             'matrix.')
    command_line_args = parser.parse_args()
    main(args=command_line_args)
