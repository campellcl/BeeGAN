import os
import tensorflow as tf
from typing import Tuple, Optional
from tensorflow.keras import initializers, constraints, regularizers, activations
import argparse
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
from Utils.TensorFlow.TFRecordLoader import TFRecordLoader


class BottleneckLayer:

    def __init__(
            self, num_neurons: int, activation: activations, kernel_initializer: initializers,
            bias_initializer: initializers, kernel_regularizer: Optional[regularizers] = None,
            bias_regularizer: Optional[regularizers] = None, activity_regularizer: Optional[regularizers] = None,
            kernel_constraint: Optional[constraints] = None, bias_constraint: Optional[constraints] = None,
            name: Optional[str] = 'bottleneck_layer'):
        self._num_neurons: int = num_neurons
        self._activation: activations = activation
        self._kernel_initializer: initializers = kernel_initializer
        self._bias_initializer: initializers = bias_initializer
        self._kernel_regularizer: Optional[regularizers] = kernel_regularizer
        self._bias_regularizer: Optional[regularizers] = bias_regularizer
        self._activity_regularizer: Optional[regularizers] = activity_regularizer
        self._kernel_constraint: Optional[constraints] = kernel_constraint
        self._bias_constraint: Optional[constraints] = bias_constraint
        self._name: Optional[str] = name
        self._dense_layer = tf.keras.layers.Dense(
            units=self._num_neurons,
            activation=self._activation,
            use_bias=True,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
            name=self._name
        )

    @property
    def dense_layer(self) -> tf.keras.layers.Dense:
        return self._dense_layer


def make_autoencoder_model(receptive_field_shape: Tuple[int, Optional[int]], bottleneck_layer: BottleneckLayer, training_batch_size: int):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=receptive_field_shape, batch_size=training_batch_size, name='encoder'),
        bottleneck_layer.dense_layer,
        tf.keras.layers.Dense(units=receptive_field_shape, batch_size=training_batch_size, name='decoder')
    ])
    return model


def main(args):
    # Command line arguments:
    is_debug: bool = args.is_verbose
    root_data_dir: str = args.root_data_dir[0]
    batch_size: int = args.batch_size
    dataset_split_str: str = args.dataset_split_str[0]
    order_deterministically: bool = args.order_deterministically

    dataset_split_type: DatasetSplitType

    # Ensure that the provided arguments are valid:
    if not os.path.isdir(root_data_dir):
        raise FileNotFoundError('The provided root data directory \'%s\' is invalid!' % root_data_dir)
    else:
        os.chdir(root_data_dir)
    # Ensure the provided dataset split can be parsed:
    if dataset_split_str == DatasetSplitType.TRAIN.value:
        dataset_split_type = DatasetSplitType.TRAIN
    elif dataset_split_str == DatasetSplitType.VAL.value:
        dataset_split_type = DatasetSplitType.VAL
    elif dataset_split_str == DatasetSplitType.TEST.value:
        dataset_split_type = DatasetSplitType.TEST
    elif dataset_split_str == DatasetSplitType.ALL.value:
        dataset_split_type = DatasetSplitType.ALL
    else:
        raise NotImplementedError('The provided dataset split type: \'%s\' was not recognized. Provide a value of: '
                                  '\'train\', \'val\', \'test\', or \'all\'.' % dataset_split_str)
    if is_debug:
        # For debugging (see:
        # https://www.tensorflow.org/guide/effective_tf2#use_tfconfigexperimental_run_functions_eagerly_when_debugging)
        tf.config.run_functions_eagerly(True)

    tf_record_loader: TFRecordLoader = TFRecordLoader(
        root_data_dir=root_data_dir,
        dataset_split_type=dataset_split_type,
        is_debug=is_debug,
        order_deterministically=order_deterministically
    )
    tf_record_ds: tf.data.TFRecordDataset = tf_record_loader.get_tf_record_dataset(
        batch_size=batch_size
    )

    # A single item from the dataset is now a batch of tensors (dataset_batch_size x 1):
    # tf_example_batch = next(iter(tf_record_ds))

    # There will be (dataset_batch_size x 1) raw/encoded ISO 8601 tensors in the tf_example_batch[0]:
    # iso_8601_tensor_batch = tf_example_batch[0]

    # There will be (dataset_batch_size x 4097 x 66) raw/encoded 2D spectrogram tensors in the tf_example_batch[1]:
    # spectrogram_tensor_batch = tf_example_batch[1]

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder argument parser.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--root-data-dir', type=str, action='store', dest='root_data_dir', required=True)
    parser.add_argument('--dataset-split', type=str, action='store', dest='dataset_split_str', required=True,
                        help='The dataset split that should be loaded (e.g. train, test, val, or all).')
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size', required=True)
    parser.add_argument('--order-deterministically', type=bool, action='store', dest='order_deterministically', required=True)
    # parser.add_argument('--train-batch-size', type=int, action='store', dest='train_batch_size', required=True,
    #                     help='The size of the input batches for the neural network during training.')
    # parser.add_argument('--val-batch-size', type=int, action='store', dest='val_batch_size', required=True,
    #                     help='The size of the input batches for the neural network during training.')
    command_line_args = parser.parse_args()
    main(args=command_line_args)
