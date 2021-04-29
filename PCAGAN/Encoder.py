import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras import layers, losses, activations, initializers, optimizers, metrics
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
from Utils.TensorFlow.TFRecordLoader import TFRecordLoader
import numpy as np
from tensorboard.plugins.hparams import api as hp

DEFAULT_NUM_FREQ_BINS = 4097


class SVDEncoder(layers.Layer):

    def __init__(self, input_dim: int, latent_dim: int, trainable: bool = True, **kwargs):
        super(SVDEncoder, self).__init__(trainable=trainable, name='encoder', dtype=tf.float32, dynamic=False, **kwargs)
        self.encoder = layers.Dense(
            input_shape=(1, input_dim),
            units=latent_dim,
            activation=activations.linear,
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
            bias_initializer=None,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            bias_constraint=None,
            name='encoder',
            # Constrain each row of weights to unit norm:
            kernel_constraint=tf.keras.constraints.unit_norm(axis=0)
        )

    def call(self, inputs, **kwargs):
        # print('inputs shape: (%s, %s)' % (inputs[0].shape, inputs[1].shape))
        return self.encoder(inputs)


class SVDDecoder(layers.Layer):
    """
    Decoder: Emulates Singular Value Decomposition (SVD) with separate weights for the both the encoder and decoder.
    """

    def __init__(self, latent_dim: int, output_dim: int, **kwargs):
        super(SVDDecoder, self).__init__(**kwargs)
        self.output_layer = layers.Dense(
            units=output_dim,
            input_shape=(latent_dim,),
            activation=activations.linear,
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
            bias_initializer=None,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            # kernel_constraint=tf.keras.constraints.unit_norm(axis=0),
            bias_constraint=None,
            name='decoder'
        )

    def call(self, latent_encoding, **kwargs):
        return self.output_layer(latent_encoding)


class SVDDecoderSharedWeightsLayer(layers.Layer):
    """
    SVDDecoderSharedWeightsLayer: Emulates Singular Value Decomposition (SVD) with a shared weights between the encoder and
     decoder. The accompanying class `Decoder` provides an alternative SVD implementation to this class, with the
     encoder and decoder each having their own respective weights.
    """
    # see: https://www.semicolonworld.com/question/61321/how-to-create-two-layers-with-shared-weights-where-one-is-the-transpose-of-the-other

    def __init__(self, encoding_layer: layers.Layer, trainable: bool = True):
        super().__init__(trainable=trainable, name='weights', dtype=tf.float32, dynamic=False)
        self.encoding_layer: layers.Layer = encoding_layer

    def call(self, inputs, **kwargs):
        """
        call: Performs W * W^T
        :param inputs:
        :param kwargs:
        :return:
        """
        weights = self.encoding_layer.weights[0]
        weights = tf.transpose(weights)
        # Linear layer (no biases):
        x = tf.linalg.matmul(inputs, weights)
        return x


class SVDAutoencoder(tf.keras.Model):

    def __init__(self, latent_dim: int, input_dim: int, share_weights: bool):
        """
        __init__: Initializer for objects of type SVDAutoencoder.
        :param latent_dim:
        :param input_dim:
        :param share_weights: <bool> A boolean flag indicating if the encoder and decoder should share the same weights
         variable, or if each should have its own distinct weights to be trained independently. Set to True for the
         former and False for the latter.
        """
        super(SVDAutoencoder, self).__init__()
        self.encoder = SVDEncoder(input_dim=input_dim, latent_dim=latent_dim)
        if share_weights:
            self.decoder: SVDDecoderSharedWeightsLayer = SVDDecoderSharedWeightsLayer(
                encoding_layer=self.encoder, trainable=True
            )
        else:
            self.decoder: SVDDecoder = SVDDecoder(latent_dim=latent_dim, output_dim=input_dim, trainable=True)

    def call(self, x, **kwargs):
        '''
        Here we unpack the data. Its structure depends on the model and what is passed to 'fit()':
        '''
        if isinstance(x, tuple):
            x = x[0]
        else:
            pass
        latent_code = self.encoder(x)
        reconstructed = self.decoder(latent_code)
        return reconstructed

    @staticmethod
    def mean_squared_error(y_pred, y_true):
        mse = tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true)))
        return mse

    @staticmethod
    def root_mean_squared_error(y_pred, y_true):
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true))))
        return rmse

    def train_step(self, data):
        """
        train_step: TODO: Docstrings.
        see: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/#introduction
        :param data:
        :return:
        """

        '''
        Here we unpack the data. Its structure depends on the model and what is passed to 'fit()':
        '''
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        # loss_fn = losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)

        with tf.GradientTape() as tape:
            # Run the input 1D frequency vector through the auto-encoder:
            latent_code = self.encoder(x)
            # Run the encoded latent representation through the decoder:
            reconstructed = self.decoder(latent_code)
            '''
            Compute the loss value (the loss function is configured in 'compile()'):
            '''
            # With the auto-encoder, what is traditionally thought of as 'y' is the original 'x'. And what is
            #  traditionally thought of as 'y_pred' is the 'reconstructed' x:
            loss = self.compiled_loss(x, reconstructed, regularization_losses=self.losses)

        # Use the gradient tape to compute the gradients of the trainable variables with respect to the loss:
        gradients = tape.gradient(loss, self.trainable_variables)
        # Run one step of gradient descent by updating the value of the weights associated with the trainable variables
        # to minimize the loss:
        self.optimizer.apply_gradients(grads_and_vars=zip(gradients, self.trainable_variables))
        # Update the metrics (including the metric that tracks the loss):
        self.compiled_metrics.update_state(x, reconstructed)
        # Prepare a dictionary mapping metric names to current values:
        metric_values = {m.name: m.result() for m in self.metrics}
        # Here we manually overwrite the loss that is computed on the backend due to a loss of precision somehow
        #  between the version computed by self.compiled_loss() and the version that is stored in self.metrics:
        metric_values['loss'] = loss
        return metric_values

    def test_step(self, data):
        """
        see: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/#providing_your_own_evaluation_step
        :param data:
        :return:
        """
        '''
        Here we unpack the data. Its structure depends on the model and what is passed to 'fit()':
        '''
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data
        # Recall that with the auto-encoder, what is usually 'y' is the original 'x'. And what is usually 'y_pred' is
        # the 'reconstructed' version of 'x':
        x_reconstructed = self(x, training=False)
        # Update the metrics tracking the loss on the validation and/or testing set:
        self.compiled_loss(x, x_reconstructed, regularization_losses=self.losses)
        # Update additional non-loss metrics specified during model compile time:
        self.compiled_metrics.update_state(x, x_reconstructed)
        # Return the dictionary mapping metric names to their current (just updated) values. Note that this will include
        # the loss metric tracked in self.metrics:
        return {m.name: m.result() for m in self.metrics}

    # @property
    # def metrics(self):
    #     # see: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/#going_lower-level
    #     return [self._loss_tracker]


def create_sequential_svd_autoencoder_model(num_frequency_bins: int, num_units_latent_dim: int, share_weights: bool) \
        -> tf.keras.Sequential:
    """
    create_sequential_model: Constructs a Sequential Keras model version of the existing SVDAutoencoder subclassed model.
     This functionality is necessary because it appears that in order to use TensorFlow HParams we have to use a
     sequential Keras model (e.g. https://www.tensorflow.org/guide/keras/sequential_model) and not a subclassed model
     (e.g. https://www.tensorflow.org/guide/keras/custom_layers_and_models). This is just an assumption though, as I
     can't seem to find any implementations with the subclassing approach. Additional supporting evidence includes
     the fact that the Keras (non-TF) Scikit-Learn API itself explicitly specifies that it allows Sequential models
     while omitting whether or not subclassed models are supported (see:
     https://faroit.com/keras-docs/2.0.0/scikit-learn-api/) and the API documentation for the TensorFlow version of
     the KerasClassifier is incredibly scarce (see:
     https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasClassifier). So in order to be
     safe we implement a sequential version of our existing subclassed model for use with TensorFlow's HParams and
     TensorFlow's tf.keras.wrappers.scikit_learn.KerasClassifier implementation of
     keras.wrappers.scikit_learn.KerasClassifier.
    :param num_frequency_bins: <int> The number of frequency bins in the source domain (by default this value is 4097
     such that each sample is a frequency bin of shape 1 x 4097).
    :param num_units_latent_dim: <int> The number of units in the 1D latent dimension to use for encoding.
    :param share_weights: <bool> A boolean flag indicating if the weights should be shared between the encoder and
     decoder. If True, the same weights variable will be used for both the auto-encoder and decoder. If False, the
     encoder and decoder will both be trainable with their own respective weight variables.
    :return sequential_svd_autoencoder: <tf.keras.Sequential> A Sequential (see:
     https://www.tensorflow.org/guide/keras/sequential_model) Keras model representing the SVD emulating auto-encoder.
     Be sure to remember to compile the model prior to use.
    """
    sequential_svd_autoencoder: tf.keras.Sequential

    # Construct encoder layer:
    encoding_layer = layers.Dense(
        input_shape=(1, num_frequency_bins),
        units=num_units_latent_dim,
        activation=activations.linear,
        use_bias=False,
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
        bias_initializer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        bias_constraint=None,
        name='encoder',
        # Constrain each row of weights to unit norm:
        kernel_constraint=tf.keras.constraints.unit_norm(axis=0),
        trainable=False if share_weights else True
    )
    # Construct decoder layer:
    if share_weights:
        decoding_layer: SVDDecoderSharedWeightsLayer = SVDDecoderSharedWeightsLayer(
            encoding_layer=encoding_layer,
            trainable=True
        )
    else:
        decoding_layer: tf.keras.layers.Layer = layers.Dense(
            units=num_frequency_bins,
            input_shape=(num_units_latent_dim,),
            activation=activations.linear,
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
            bias_initializer=None,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            # kernel_constraint=tf.keras.constraints.unit_norm(axis=0),
            bias_constraint=None,
            name='decoder'
        )
    # Construct sequential Autoencoder:
    sequential_svd_autoencoder: tf.keras.Sequential = tf.keras.Sequential([
        encoding_layer,
        decoding_layer
    ])
    return sequential_svd_autoencoder


def main(args):
    # Command line arguments:
    is_debug: bool = args.is_verbose
    root_data_dir: str = args.root_data_dir[0]
    output_data_dir: str = args.output_data_dir[0]
    batch_size: int = args.batch_size
    num_epochs: int = args.num_epochs
    # dataset_split_str: str = args.dataset_split_str[0]
    order_deterministically: bool = args.order_deterministically
    share_weights: bool = args.share_weights

    # TODO: wrap this value in hyper-parameter gird search:
    latent_dim = 1

    dataset_split_type: DatasetSplitType

    # Ensure that the provided arguments are valid:
    cwd = os.getcwd()
    if not os.path.isdir(root_data_dir):
        raise FileNotFoundError('The provided root data directory \'%s\' is invalid!' % root_data_dir)
    if not os.path.isdir(output_data_dir):
        raise FileNotFoundError('The provided output data directory \'%s\' is invalid!' % output_data_dir)
    # Ensure the provided dataset split can be parsed:
    # if dataset_split_str == DatasetSplitType.TRAIN.value:
    #     dataset_split_type = DatasetSplitType.TRAIN
    # elif dataset_split_str == DatasetSplitType.VAL.value:
    #     dataset_split_type = DatasetSplitType.VAL
    # elif dataset_split_str == DatasetSplitType.TEST.value:
    #     dataset_split_type = DatasetSplitType.TEST
    # elif dataset_split_str == DatasetSplitType.ALL.value:
    #     dataset_split_type = DatasetSplitType.ALL
    # else:
    #     raise NotImplementedError('The provided dataset split type: \'%s\' was not recognized. Provide a value of: '
    #                               '\'train\', \'val\', \'test\', or \'all\'.' % dataset_split_str)
    if is_debug:
        # For debugging (see:
        # https://www.tensorflow.org/guide/effective_tf2#use_tfconfigexperimental_run_functions_eagerly_when_debugging)
        tf.config.run_functions_eagerly(True)
        # Log tensor placement (see: https://www.tensorflow.org/guide/gpu#logging_device_placement):
        # tf.debugging.set_log_device_placement(True)

    # Obtain the TFRecord dataset corresponding to the requested dataset split ('train', 'val', 'test', 'all'):
    tf_record_loader: TFRecordLoader = TFRecordLoader(
        root_data_dir=root_data_dir,
        is_debug=is_debug
    )

    # loss_tracker = metrics.Mean(name='loss')
    autoencoder: SVDAutoencoder = SVDAutoencoder(latent_dim=latent_dim, input_dim=4097, share_weights=share_weights)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=[metrics.RootMeanSquaredError()])
    autoencoder.build(input_shape=(batch_size, 4097))
    print(autoencoder.summary())

    if is_debug:
        # tensorboard callback for profiling training process:
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), '../Output'),
                                                     profile_batch='15, 30')
    else:
        tb_callback = None

    train_tf_record_ds: tf.data.TFRecordDataset = tf_record_loader.get_batched_tf_record_dataset(
        dataset_split_type=DatasetSplitType.TRAIN,
        order_deterministically=order_deterministically,
        batch_size=batch_size,
        prefetch=True,
        cache=False
    )
    val_tf_record_ds: tf.data.TFRecordDataset = tf_record_loader.get_batched_tf_record_dataset(
        dataset_split_type=DatasetSplitType.VAL,
        order_deterministically=order_deterministically,
        batch_size=batch_size,
        prefetch=True,
        cache=False
    )
    test_tf_record_ds: tf.data.TFRecordDataset = tf_record_loader.get_batched_tf_record_dataset(
        dataset_split_type=DatasetSplitType.TEST,
        order_deterministically=order_deterministically,
        batch_size=batch_size,
        prefetch=True,
        cache=False
    )

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
    :param validation_batch_size: We provide a value of None because we are using TFRecord datasets which dictate their own
     batch size (in our case sourced from the command line arguments).
    :param validation_freq: When provided as an integer, specifies how many training epochs to run before performing
     a validation run. We specify with a value of 1 that the validation metrics should be computed after every 
     training epoch. 
    '''
    autoencoder.fit(
        train_tf_record_ds,
        batch_size=None,
        epochs=num_epochs,
        verbose=1,
        callbacks=[tb_callback] if is_debug else None,
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
    autoencoder.save(filepath=os.path.join(output_data_dir, 'SavedModel'), overwrite=True, include_optimizer=True)
    # x_val_batch = next(iter(val_tf_record_ds))
    x_pred = autoencoder.predict(val_tf_record_ds, verbose=1)
    ''' Load and reconstruct the serialized model to assert they are the same: '''
    reconstructed_model = tf.keras.models.load_model(os.path.join(output_data_dir, 'SavedModel'))
    # Run a prediction using the reconstructed model and assert they are the same:
    reconstructed_model_x_pred = reconstructed_model.predict(val_tf_record_ds, verbose=1)
    # Now assert relative equality between the in-memory version and the reconstructed model:
    np.testing.assert_allclose(x_pred, reconstructed_model_x_pred)
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder argument parser.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--root-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    parser.add_argument('--output-data-dir', type=str, nargs=1, action='store', dest='output_data_dir', required=True)
    # parser.add_argument('--dataset-split', type=str, nargs=1, action='store', dest='dataset_split_str', required=True,
    #                     help='The dataset split that should be loaded (e.g. train, test, val, or all).')
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size', required=True)
    parser.add_argument('--num-epochs', type=int, action='store', dest='num_epochs', required=True)
    parser.add_argument('--order-deterministically', type=bool, action='store', dest='order_deterministically',
                        required=True)
    parser.add_argument('--share-weights', type=bool, action='store', dest='share_weights', required=True,
                        help='A boolean flag indicating if the Encoder and Decoder in the Autoencoder should share the '
                             'same weight matrix. If set to false, the Encoder and Decoder will have their own separate'
                             ' weight matrices. If set to true, the Encoder and Decoder will share the same weight '
                             'matrix.')
    parser.add_argument(
        '--num-freq-bins', type=int, action='store', dest='num_freq_bins', default=DEFAULT_NUM_FREQ_BINS,
        required=False, help='The number of frequency bins for a single sample, by default this corresponds to: %d '
                             'frequency bins per spectrum.' % DEFAULT_NUM_FREQ_BINS
    )
    command_line_args = parser.parse_args()
    main(args=command_line_args)
