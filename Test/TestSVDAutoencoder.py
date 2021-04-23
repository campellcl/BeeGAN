import os
import sys
import argparse
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
from Utils.TensorFlow.TFRecordLoader import TFRecordLoader
import numpy as np
import tensorflow as tf
from PCAGAN.Encoder import SVDAutoencoder
from typing import Optional, Union
from tensorflow.keras import layers, losses, activations, initializers, optimizers, metrics
import shutil

DEFAULT_NUM_SAMPLES_FOR_TEST = 30000   # IMPORTANT: You must manually verify this amount of 32-bit float samples will
# fit on the system dedicated GPU memory.
DEFAULT_NUM_EPOCHS = 10


class TestSVDAutoencoder:

    def __init__(self, root_data_dir: str, output_data_dir: str, is_debug: bool, order_deterministically: bool,
                 num_frequency_bins: Optional[int] = 4097):
        self._root_data_dir = root_data_dir
        self._output_data_dir = output_data_dir
        assert os.path.isdir(self._root_data_dir), self._root_data_dir
        assert os.path.isdir(self._output_data_dir), self._output_data_dir
        self._is_debug: bool = is_debug
        self._order_deterministically: bool = order_deterministically
        self._num_frequency_bins: int = num_frequency_bins
        self._tf_record_loader = TFRecordLoader(
            root_data_dir=self._root_data_dir,
            is_debug=self._is_debug
        )

    def run_test(self, num_samples_for_test: int, num_units_latent_space: int, share_weights: bool,
                 train_batch_size: int, num_epochs: int):
        # Load a subset of the training TFRecordDataset into memory:
        x_train: tf.Tensor = self._tf_record_loader.get_dataset_subset_from_memory(
            dataset_split_type=DatasetSplitType.TRAIN,
            order_deterministically=self._order_deterministically,
            num_samples=num_samples_for_test,
            num_frequency_bins=self._num_frequency_bins
        )
        '''
        Normalize each row to unit length so that they sum to one. We do this because we previously encountered floating
         point precision loss errors during serialization and restoration of the weight values which are very close to
         zero:
        '''
        # Compute the norm of each row:
        row_norm = np.linalg.norm(x_train.numpy(), axis=1).reshape((-1, 1))
        # Normalize the training dataset:
        x_train = tf.divide(x_train, row_norm)

        if self._is_debug:
            ''' 
            Enable TensorBoard callbacks, see the following URLs for more info:
            https://www.tensorflow.org/guide/keras/train_and_evaluate#visualizing_loss_and_metrics_during_training
            https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
            https://www.tensorflow.org/tensorboard/get_started
            '''
            tensorboard_output_dir = os.path.join(self._output_data_dir, 'TensorBoard')
            if not os.path.exists(tensorboard_output_dir):
                os.makedirs(tensorboard_output_dir)
            else:
                # Clear the directory if it is already populated:
                if os.listdir(tensorboard_output_dir):
                    # Non-empty directory
                    shutil.rmtree(tensorboard_output_dir, ignore_errors=False)
            # Create the TensorBoard writer callback:
            tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_output_dir,
                histogram_freq=1,
                embeddings_freq=0,
                write_graph=True,
                write_images=True,
                update_freq="epoch",
                profile_batch=0
            )
        else:
            tb_callback = None

        '''
        Train an autoencoder on the x_train data subset in memory:
        '''
        autoencoder = SVDAutoencoder(
            input_dim=self._num_frequency_bins,
            latent_dim=num_units_latent_space,
            share_weights=share_weights
        )

        autoencoder.compile(
            optimizer=optimizers.Adam(learning_rate=0.01),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.RootMeanSquaredError()]
        )
        autoencoder.build(input_shape=(train_batch_size, self._num_frequency_bins))

        if self._is_debug:
            print(autoencoder.summary())

        autoencoder.fit(
            x_train,
            batch_size=train_batch_size,
            epochs=num_epochs,
            verbose=1,
            callbacks=[tb_callback] if tb_callback is not None else None,
            validation_data=None,
            shuffle=False,
            class_weight=None,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1
        )

        encoder = autoencoder.layers[0]
        decoder = autoencoder.layers[1]

        '''
        Compute the closed form SVD solution on the dataset:
        '''
        if self._is_debug:
            print('Computing closed form SVD solution on dataset (%s) a subset of size %d samples...'
                  % (DatasetSplitType.TRAIN.value, DEFAULT_NUM_SAMPLES_FOR_TEST))
        s, u, v = tf.linalg.svd(x_train, full_matrices=False, compute_uv=True, name='SVD')
        if self._is_debug:
            print('s shape: %s' % (s.shape,))
            print('u shape: %s' % (u.shape,))
            print('v shape: %s' % (v.shape,))

        '''
        Save the output to disk:
        '''
        # Get the latent encoding (the kernel/filter of the decoder):
        weights_encoder: tf.Variable = encoder.weights[0]
        weights_decoder: tf.Variable = decoder.weights[0]

        # Save the weights (encoder) to disk:
        output_path: str = os.path.join(self._output_data_dir, 'WeightsEncoder-train-%d.npy' % num_samples_for_test)
        np.save(output_path, weights_encoder.numpy().flatten())
        print('Saved the encoder\'s weights to location: \'%s\'' % output_path)

        # Save the weights (decoder) to disk:
        output_path: str = os.path.join(self._output_data_dir, 'WeightsDecoder-train-%d.npy' % num_samples_for_test)
        np.save(output_path, weights_decoder.numpy().flatten())
        print('Saved the decoder\'s weights to location: \'%s\'' % output_path)

        # Save the closed form SVD solution to disk:
        output_path: str = os.path.join(self._output_data_dir, 'SingularVector-train-%d.npy' % num_samples_for_test)
        right_singular_vector = v[:, 0]
        np.save(output_path, right_singular_vector)
        print('Saved the principal right singular vector (closed form SVD solution) to location: \'%s\'' % output_path)

        # Display the result to the user with an assertion:
        # np.testing.assert_allclose(right_singular_vector, weights_encoder.numpy().flatten())


def main(args):
    # Command line arguments:
    is_debug: bool = args.is_verbose
    root_data_dir: str = args.root_data_dir[0]
    output_data_dir: str = args.output_data_dir[0]
    train_batch_size: int = args.train_batch_size
    order_deterministically: bool = args.order_deterministically
    num_samples_train_set: Optional[int] = args.num_samples_train_set
    num_samples_val_set: Optional[int] = args.num_samples_val_set
    num_samples_test_set: Optional[int] = args.num_samples_test_set
    num_samples_for_test: int = args.num_samples_for_test
    num_frequency_bins: int = args.num_freq_bins    # 4097 by default.
    num_units_latent_space: int = args.num_units_latent_space   # 1 by default.
    num_epochs: int = args.num_epochs
    share_weights: bool = args.share_weights

    dataset_split_type: DatasetSplitType

    # Ensure that the provided arguments are valid:
    cwd = os.getcwd()
    if not os.path.isdir(root_data_dir):
        raise FileNotFoundError('The provided root data directory \'%s\' is invalid!' % root_data_dir)
    else:
        os.chdir(root_data_dir)
    os.chdir(cwd)
    if not os.path.isdir(output_data_dir):
        raise FileNotFoundError('The provided output data directory: \'%s\' is invalid!' % output_data_dir)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    test_svd_autoencoder = TestSVDAutoencoder(
        root_data_dir=root_data_dir,
        output_data_dir=output_data_dir,
        is_debug=is_debug,
        order_deterministically=order_deterministically,
        num_frequency_bins=num_frequency_bins
    )
    test_svd_autoencoder.run_test(
        num_samples_for_test=num_samples_for_test,
        num_units_latent_space=num_units_latent_space,
        share_weights=share_weights,
        train_batch_size=train_batch_size,
        num_epochs=num_epochs
    )

    sys.exit(0)


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
        '--num-samples-for-test', type=int, action='store', dest='num_samples_for_test', required=False,
        default=DEFAULT_NUM_SAMPLES_FOR_TEST,
        help='The number of samples to both compute closed form SVD on, and to train the neural network on. WARNING: '
             'The specified number of samples must fit into dedicated GPU memory. There are (4097,) 32 bit floats '
             'per-sample, and %d samples by default.' % DEFAULT_NUM_SAMPLES_FOR_TEST
    )
    parser.add_argument('--num-samples-train-set', type=int, action='store', dest='num_samples_train_set', required=False, default=None)
    parser.add_argument('--num-samples-val-set', type=int, action='store', dest='num_samples_val_set', required=False, default=None)
    parser.add_argument('--num-samples-test-set', type=int, action='store', dest='num_samples_test_set', required=False, default=None)
    parser.add_argument(
        '--num-freq-bins', type=int, action='store', dest='num_freq_bins', default=4097, required=False,
        help='The number of frequency bins for a single sample, by default this corresponds to: %d frequency bins per '
             'spectrum.' % 4097
    )
    parser.add_argument(
        '--num-units-latent-space', type=int, action='store', dest='num_units_latent_space', required=False, default=1,
        help='The number of units/neurons to use in the 1D latent space to encode the source data. By default: %d' % 1
    )
    parser.add_argument('--num-epochs', type=int, action='store', dest='num_epochs', required=False, default=DEFAULT_NUM_EPOCHS, help='Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.')
    parser.add_argument('--share-weights', type=bool, action='store', dest='share_weights', required=True,
                        help='A boolean flag indicating if the Encoder and Decoder in the Autoencoder should share the '
                             'same weight matrix. If set to false, the Encoder and Decoder will have their own separate'
                             ' weight matrices. If set to true, the Encoder and Decoder will share the same weight '
                             'matrix.')
   # parser.add_argument('--desired-num-samples', type=int, action='store', dest='desired_num_samples', required=False, default=100000)
    command_line_args = parser.parse_args()
    main(args=command_line_args)
