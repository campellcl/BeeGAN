import os
import tensorflow as tf
from typing import Tuple, Optional
from tensorflow.keras import initializers, constraints, regularizers, activations, optimizers, losses
import argparse
from tensorflow.python.keras.utils import losses_utils
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
from Utils.TensorFlow.TFRecordLoader import TFRecordLoader

# SAMPLE_FREQUENCIES_SHAPE = (-1, 4097, 1)


class Autoencoder(tf.keras.Model):

    def __init__(self, encoder: tf.keras.Model, latent: tf.keras.Model, decoder: tf.keras.Model, name: str):
        super(Autoencoder, self).__init__(name=name)
        self._encoder = encoder
        self._encoder_optimizer: Optional[optimizers.Optimizer] = None
        self._latent = latent
        self._decoder = decoder
        self._decoder_optimizer: Optional[optimizers.Optimizer] = None
        self._loss_function: Optional[losses.Loss] = None

    def compile(self, encoder_optimizer: optimizers.Optimizer, decoder_optimizer: optimizers.Optimizer,
                loss_function: losses.Loss):
        super(Autoencoder, self).compile()
        self._encoder_optimizer = encoder_optimizer
        self._decoder_optimizer = decoder_optimizer
        self._loss_function = loss_function

    def call(self, x):
        encoded = self._encoder(x)
        decoded = self._decoder(encoded)
        return decoded

    # def train_step(self, dataset_record):
    #     if isinstance(dataset_record, tuple):
    #         sample_frequencies_tensor = dataset_record[0]
    #         sample_iso_8601 = dataset_record[1]


class BottleneckLayer:
    """
    BottleneckLayer: A class storing the representation of the bottleneck layer/latent-encoding.
    """

    def __init__(
            self, num_neurons: int, activation: activations, kernel_initializer: initializers.Initializer,
            bias_initializer: initializers.Initializer, kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            activity_regularizer: Optional[regularizers.Regularizer] = None,
            kernel_constraint: Optional[constraints.Constraint] = None,
            bias_constraint: Optional[constraints.Constraint] = None, name: Optional[str] = 'bottleneck_layer'):
        """
        __init__: Initializer for objects of type BottleneckLayer.
        :param num_neurons: <int> The number of units/neurons in the 1D bottleneck layer.
        :param activation: <tf.keras.activations> The activation function pertaining to units/neurons in the bottleneck
         layer (see: https://www.tensorflow.org/api_docs/python/tf/keras/activations).
        :param kernel_initializer: <tf.keras.initializers.Initializer> The initialization technique for the weights in
         the bottleneck layer of the network (see: https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
        :param bias_initializer: <tf.keras.initializers.Initializer> The initialization technique for the bias terms in
         the bottleneck layer of the network (see: https://www.tensorflow.org/api_docs/python/tf/keras/initializers).
        :param kernel_regularizer: <tf.keras.regularizers.Regularizer/None> The regularization technique for the weights
         in the bottleneck layer of the network (see: https://www.tensorflow.org/api_docs/python/tf/keras/regularizers)
         if any.
        :param bias_regularizer: <tf.keras.regularizers.Regularizer/None> The regularization technique for the biases in
         the bottleneck layer of the network (see: https://www.tensorflow.org/api_docs/python/tf/keras/regularizers) if
         any.
        :param activity_regularizer: <tf.keras.regularizers.Regularizer/None> The regularization technique for the
         activation function results in the bottleneck layer of the network (see:
         https://www.tensorflow.org/api_docs/python/tf/keras/regularizers) if any.
        :param kernel_constraint: <tf.keras.constraints.Constraint/None> The function which should be used to impose
         constraints on the weight values of the bottleneck layer in the neural network (if any). For more details, see:
         https://www.tensorflow.org/api_docs/python/tf/keras/constraints.
        :param bias_constraint: <tf.keras.constraints.Constraint/None> The function which should be used to impose
         constraints on the bias values of the bottleneck layer in the neural network (if any). For more details, see:
         https://www.tensorflow.org/api_docs/python/tf/keras/constraints.
        :param name: <str> The name of the bottleneck layer in the neural network. This field is used as a key value
         in the TensorFlow computational graph.
        """
        self._num_neurons: int = num_neurons
        self._activation: activations = activation
        self._kernel_initializer: initializers.Initializer = kernel_initializer
        self._bias_initializer: initializers.Initializer = bias_initializer
        self._kernel_regularizer: Optional[regularizers.Regularizer] = kernel_regularizer
        self._bias_regularizer: Optional[regularizers.Regularizer] = bias_regularizer
        self._activity_regularizer: Optional[regularizers.Regularizer] = activity_regularizer
        self._kernel_constraint: Optional[constraints.Constraint] = kernel_constraint
        self._bias_constraint: Optional[constraints.Constraint] = bias_constraint
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


def make_sequential_autoencoder_model(receptive_field_size: int, bottleneck_layer: BottleneckLayer, batch_size: int):
    """
    make_autoencoder_model: TODO: Docstrings.

    :param receptive_field_size: <int> The number of units/neurons in the 1D input layer/receptive field.
    :param bottleneck_layer:
    :param batch_size:
    :return:
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=receptive_field_size, batch_size=batch_size, name='encoder'),
        bottleneck_layer.dense_layer,
        tf.keras.layers.Dense(units=receptive_field_size, batch_size=batch_size, name='decoder')
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

    # Obtain the TFRecord dataset corresponding to the requested dataset split ('train', 'val', 'test', 'all'):
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

    # There will be (dataset_batch_size x 4097) raw/encoded 2D spectrogram tensors in the tf_example_batch[1]:
    # spectrogram_tensor_batch = tf_example_batch[1]

    # A single sample by default size (4097,) is hence:
    # sample = spectrogram_tensor_batch[0]
    # sample_iso_8601_str = iso_8601_tensor_batch[0]

    ''' For the SVD model: '''
    # Define the encoder:
    encoder = tf.keras.Sequential(
        layers=[
            tf.keras.layers.InputLayer(input_shape=(4097, 1), batch_size=batch_size, name='encoder')
        ],
        name='encoder'
    )
    # Define the latent dimension:
    latent = tf.keras.Sequential([
        tf.keras.layers.Dense(
            units=1,
            activation=activations.linear,
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
            bias_initializer=initializers.Zeros(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name='latent'
        )
    ])
    # Define the decoder:
    decoder = tf.keras.Sequential(
        layers=[
            tf.keras.layers.Dense(
                units=4097,
                activation=activations.linear,
                use_bias=True,
                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=initializers.Zeros(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                batch_size=batch_size,
                name='decoder'
            )
        ],
        name='decoder'
    )
    # Define the SVD auto-encoder:
    svd_autoencoder = Autoencoder(
        encoder=encoder,
        latent=latent,
        decoder=decoder,
        name='autoencoder'
    )
    # Setup the compilation arguments for the model:
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.0,
        nesterov=True,
        name='sgd_nesterov'
    )
    # see: https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy#args_1
    loss_func: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,   # We are providing y_pred as the logits tensor, and not a stand-alone probability dist.
        label_smoothing=0,
        reduction=losses_utils.ReductionV2.AUTO,    # This is primarily relevant for distributed TensorFlow
        name='binary_crossentropy'
    )
    # Build the component models:
    encoder.build(input_shape=(150, 4097, 1))
    latent.build(input_shape=())
    decoder.build(input_shape=(150, 4097, 1))
    # Compile the component models:
    encoder.compile(
        optimizer=optimizer,
        loss=loss_func
    )
    latent.compile(
        optimizer=optimizer
    )
    decoder.compile(
        optimizer=optimizer,
        loss=loss_func
    )
    # Compile the model (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile):
    svd_autoencoder.compile(
        encoder_optimizer=optimizer,
        decoder_optimizer=optimizer,
        loss_function=loss_func
    )
    svd_autoencoder.build(
        input_shape=(150, 4097, 1)
    )
    print(svd_autoencoder.summary())
    # Train/fit the model (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit):
    svd_autoencoder.fit(
        tf_record_ds,
        epochs=10,
        shuffle=False
    )
    exit(0)


    # BottleneckLayer the encoding/latent-code of the data (here for the SVD case):
    # bottleneck_layer: BottleneckLayer = BottleneckLayer(
    #     num_neurons=4097,
    #     activation=tf.keras.activations.linear,
    #     kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
    #     bias_initializer=initializers.Zeros(),
    #     kernel_regularizer=None,
    #     bias_regularizer=None,
    #     activity_regularizer=None,
    #     kernel_constraint=None,
    #     bias_constraint=None,
    #     name='bottleneck_layer'
    # )

    # Instantiate the model:
    # autoencoder = make_sequential_autoencoder_model(
    #     receptive_field_size=4097,
    #     bottleneck_layer=bottleneck_layer,
    #     batch_size=batch_size
    # )
    # # Setup the compilation arguments for the model:
    # optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD(
    #     learning_rate=0.01,
    #     momentum=0.0,
    #     nesterov=True,
    #     name='sgd_nesterov'
    # )
    # # see: https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy#args_1
    # loss_func: tf.keras.losses.Loss = tf.keras.losses.BinaryCrossentropy(
    #     from_logits=True,   # We are providing y_pred as the logits tensor, and not a stand-alone probability dist.
    #     label_smoothing=0,
    #     reduction=losses_utils.ReductionV2.AUTO,    # This is primarily relevant for distributed TensorFlow
    #     name='binary_crossentropy'
    # )
    # # Compile the model (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile):
    # autoencoder.compile(
    #     optimizer=optimizer,
    #     loss=loss_func,
    #     metrics=None,
    #     loss_weights=None,
    #     weighted_metrics=None,
    #     run_eagerly=is_debug,
    #     # The number of batches to run during each tf.function call. At most one full epoch per execution.
    #     steps_per_execution=1
    # )
    # Train/fit the model (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit):
    # autoencoder.fit(
    #     x=???
    # )
    # pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder argument parser.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--root-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    parser.add_argument('--dataset-split', type=str, nargs=1, action='store', dest='dataset_split_str', required=True,
                        help='The dataset split that should be loaded (e.g. train, test, val, or all).')
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size', required=True)
    parser.add_argument('--order-deterministically', type=bool, action='store', dest='order_deterministically',
                        required=True)
    command_line_args = parser.parse_args()
    main(args=command_line_args)
