import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, losses, activations, initializers, optimizers, metrics
import kerastuner as kt
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
from Utils.TensorFlow.TFRecordLoader import TFRecordLoader


class Encoder(layers.Layer):

    def __init__(self, source_input_dim: int = 4097, latent_dim: int = 1):
        super(Encoder, self).__init__()
        self.encoder = layers.Dense(
            input_shape=(1, source_input_dim),
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
        latent_encoding = self.encoder(inputs)
        return latent_encoding


class Decoder(layers.Layer):

    def __init__(self, source_input_dim: int = 4097, latent_dim: int = 1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.output_layer = layers.Dense(
            units=source_input_dim,
            input_shape=(latent_dim,),
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

    def __init__(self, source_input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(source_input_dim=source_input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(source_input_dim=source_input_dim)

    def call(self, x, **kwargs):
        latent_code = self.encoder(x)
        reconstructed = self.decoder(latent_code)
        return reconstructed

    def train_step(self, data):
        """
        train_step: TODO: Docstrings.
        :param data:
        :return:
        """

        '''
        Here we unpack the data. Its structure depends on the mdoel and what is passed to 'fit()':
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
        # TODO: Here we manually overwrite the loss that is computed on the backend due to a loss of precision somehow
        #  between the version computed by self.compiled_loss() and the version that is stored in self.metrics:
        metric_values['loss'] = loss
        return metric_values


class HyperAutoencoder(kt.HyperModel):

    def __init__(self):
        super().__init__()

    def build(self, hp):

        # Tune the capacity (number of units/neurons) in the latent encoding (see: https://keras-team.github.io/keras-tuner/documentation/hyperparameters/):
        hp_num_units_latent_code = hp.Int('num_units_latent_code', min_value=1, max_value=20, step=1, default=1, sampling='linear')
        autoencoder = Autoencoder(source_input_dim=4097, latent_dim=hp_num_units_latent_code)
        autoencoder.compile(
            optimizer=optimizers.Adam(),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.RootMeanSquaredError()]
        )
        return autoencoder


def main(args):
    # Command line arguments:
    is_debug: bool = args.is_verbose
    root_data_dir: str = args.root_data_dir[0]
    batch_size: int = args.batch_size
    dataset_split_str: str = args.dataset_split_str[0]
    order_deterministically: bool = args.order_deterministically

    dataset_split_type: DatasetSplitType

    # Ensure that the provided arguments are valid:
    cwd = os.getcwd()
    if not os.path.isdir(root_data_dir):
        raise FileNotFoundError('The provided root data directory \'%s\' is invalid!' % root_data_dir)
    else:
        os.chdir(root_data_dir)
    os.chdir(cwd)
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
        # Log tensor placement (see: https://www.tensorflow.org/guide/gpu#logging_device_placement):
        # tf.debugging.set_log_device_placement(True)
        # tensorboard callback for profiling training process:
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(os.getcwd(), '../Output'),
            profile_batch='15, 30'
        )
    else:
        tb_callback = None

    # Obtain the TFRecord dataset corresponding to the requested dataset split ('train', 'val', 'test', 'all'):
    train_tf_record_loader: TFRecordLoader = TFRecordLoader(
      root_data_dir=root_data_dir,
      dataset_split_type=DatasetSplitType.TRAIN,
      is_debug=is_debug,
      order_deterministically=order_deterministically
    )
    val_tf_record_loader: TFRecordLoader = TFRecordLoader(
        root_data_dir=root_data_dir,
        dataset_split_type=DatasetSplitType.VAL,
        order_deterministically=order_deterministically,
        is_debug=is_debug
    )
    test_tf_record_loader: TFRecordLoader = TFRecordLoader(
        root_data_dir=root_data_dir,
        dataset_split_type=DatasetSplitType.TEST,
        order_deterministically=order_deterministically,
        is_debug=is_debug
    )
    # Hyper-parameter tuning wrapper for the auto-encoder provided via the Keras hyper-parameter tuner:
    autoencoder_hypermodel = HyperAutoencoder()

    hyperparam_tuner = kt.Hyperband(
        hypermodel=autoencoder_hypermodel,
        objective='val_accuracy',
        max_epochs=10,
        hyperband_iterations=1,
        seed=None
    )
    hyperparam_tuner.search_space_summary()

    if dataset_split_type == DatasetSplitType.TRAIN:
        train_tf_record_ds: tf.data.TFRecordDataset = train_tf_record_loader.get_tf_record_dataset(
            batch_size=batch_size
        )
        # Note that the search method has the same signature as tf.keras.Model.fit (see:
        # https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html):
        hyperparam_tuner.search(train_tf_record_ds, epochs=10, shuffle=False, steps_per_epoch=None)
    elif dataset_split_type == DatasetSplitType.VAL:
        val_tf_record_ds: tf.data.TFRecordDataset = val_tf_record_loader.get_tf_record_dataset(
            batch_size=batch_size
        )
        # Note that the search method has the same signature as tf.keras.Model.fit (see:
        # https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html):
        hyperparam_tuner.search(val_tf_record_ds, epochs=10, shuffle=False, steps_per_epoch=None)
    elif dataset_split_type == DatasetSplitType.TEST:
        test_tf_record_ds: tf.data.TFRecordDataset = test_tf_record_loader.get_tf_record_dataset(
            batch_size=batch_size
        )
        # Note that the search method has the same signature as tf.keras.Model.fit (see:
        # https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html):
        hyperparam_tuner.search(test_tf_record_ds, epochs=10, shuffle=False, steps_per_epoch=None)
    else:
        # DatasetSplitType == DatasetSplitType.ALL
        train_tf_record_ds: tf.data.TFRecordDataset = train_tf_record_loader.get_tf_record_dataset(
            batch_size=batch_size
        )
        val_tf_record_ds: tf.data.TFRecordDataset = val_tf_record_loader.get_tf_record_dataset(
            batch_size=batch_size
        )
        # test_tf_record_ds: tf.data.TFRecordDataset = test_tf_record_loader.get_tf_record_dataset(
        #     batch_size=batch_size
        # )
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
        # Note that the search method has the same signature as tf.keras.Model.fit (see:
        # https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html):
        if is_debug:
            callbacks = [tb_callback]
            hyperparam_tuner.search(
                train_tf_record_ds,
                batch_size=None,
                epochs=10,
                verbose=1,
                callbacks=callbacks,
                validation_data=val_tf_record_ds,
                shuffle=False,
                class_weight=None,
                sample_weight=None,
                steps_per_epoch=None,
                validation_steps=None,
                validation_batch_size=None,
                validation_freq=1
            )
        else:
            hyperparam_tuner.search(
                train_tf_record_ds,
                batch_size=None,
                epochs=10,
                verbose=1,
                validation_data=val_tf_record_ds,
                shuffle=False,
                class_weight=None,
                sample_weight=None,
                steps_per_epoch=None,
                validation_steps=None,
                validation_batch_size=None,
                validation_freq=1
            )


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
