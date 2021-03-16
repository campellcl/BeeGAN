import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, losses, activations, initializers, optimizers, metrics
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
from Utils.TensorFlow.TFRecordLoader import TFRecordLoader

latent_dim = 1


class Encoder(layers.Layer):

    def __init__(self, original_dim: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.encoder = layers.Dense(
            input_shape=(1, 4097),
            units=latent_dim,
            activation=activations.linear,
            use_bias=False,
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
            bias_initializer=None,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name='encoder'
        )

    def call(self, inputs, **kwargs):
        # print('inputs shape: (%s, %s)' % (inputs[0].shape, inputs[1].shape))
        return self.encoder(inputs)


class Decoder(layers.Layer):

    def __init__(self, output_dim: int, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.output_layer = layers.Dense(
            units=output_dim,
            input_shape=(1,),
            activation=activations.linear,
            use_bias=True,
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
            bias_initializer=initializers.zeros,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name='decoder'
        )

    def call(self, latent_encoding, **kwargs):
        return self.output_layer(latent_encoding)


class Autoencoder(tf.keras.Model):

    def __init__(self, latent_dim, original_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(original_dim=original_dim, latent_dim=latent_dim)
        self.decoder = Decoder(output_dim=original_dim)
        self._loss_tracker = metrics.Mean(name='loss')

    def call(self, x, **kwargs):
        latent_code = self.encoder(x)
        reconstructed = self.decoder(latent_code)
        return reconstructed

    def reconstruction_error(self, y_pred, y_true):
        reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true)))
        return reconstruction_error

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        loss_fn = losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)

        with tf.GradientTape() as tape:
            # Run the input 1D frequency vector through the auto-encoder:
            latent_code = self.encoder(data)
            # Run the encoded latent representation through the decoder:
            reconstructed = self.decoder(latent_code)
            # Compute the reconstruction error as a loss function:
            loss = self.reconstruction_error(y_true=data, y_pred=reconstructed)

        # Use the gradient tape to compute the gradients of the trainable variables with respect to the loss:
        gradients = tape.gradient(loss, self.trainable_variables)
        # Run one step of gradient descent by updating the value of the variables to minimize the loss:
        self.optimizer.apply_gradients(grads_and_vars=zip(gradients, self.trainable_variables))
        # Compute and retain the loss metric:
        self._loss_tracker.update_state(values=loss)
        return {'loss': self._loss_tracker.result()}

    @property
    def metrics(self):
        # see: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/#going_lower-level
        return [self._loss_tracker]


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
      batch_size=1
    )

    # loss_tracker = metrics.Mean(name='loss')
    autoencoder = Autoencoder(latent_dim=latent_dim, original_dim=4097)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder.build(input_shape=(1, 4097))
    print(autoencoder.summary())
    # steps_per_epcoh: Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
    autoencoder.fit(tf_record_ds, epochs=10, shuffle=False, steps_per_epoch=None)
    # autoencoder = SVDAutoencoder(latent_dim)
    # autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    # autoencoder.fit(tf_record_ds, tf_record_ds, epochs=10, shuffle=False)


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