import os
import sys
import argparse
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
from Utils.TensorFlow.TFRecordLoader import TFRecordLoader
import numpy as np
import tensorflow as tf
from PCAGAN.Encoder import Autoencoder
from typing import Optional, Union
from tensorflow.keras import layers, losses, activations, initializers, optimizers, metrics

DEFAULT_NUM_SAMPLES_FOR_TEST = 30000   # IMPORTANT: You must manually verify this amount of 32-bit float samples will
# fit on the system dedicated GPU memory.
DEFAULT_NUM_EPOCHS = 10


def determine_num_samples_in_dataset(dataset: tf.data.Dataset, dataset_split_type: DatasetSplitType) -> int:
    num_samples: int = -1
    assert dataset_split_type != DatasetSplitType.ALL, dataset_split_type

    num_samples = dataset.cardinality().numpy()
    if num_samples == tf.data.UNKNOWN_CARDINALITY:
        command_line_arg_text: str = '--num-samples-%s-set' % dataset_split_type.value
        print('Warning: Cannot implicitly determine number of samples in the provided TF dataset (%s) as '
              'tf.data reports an unknown cardinality. This value must now be computed by iterating over all samples, '
              'this could take some time. Provide the resulting number of samples as a command line argument: \'%s\' '
              'to skip this step next time...' % (dataset_split_type, command_line_arg_text))
        num_samples: int = 0
        for i, (x_0, _) in dataset.enumerate(start=0):
            if len(x_0.shape) == 1:
                # The dataset is not batched and this is a single sample.
                num_samples += 1
            else:
                # The dataset is batched and this is a batch of samples.
                num_samples += x_0.shape[0]
    print('Number of Samples in dataset (%s): %d' % (dataset_split_type, num_samples))
    return num_samples


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
    # test_tf_record_loader: TFRecordLoader = TFRecordLoader(
    #     root_data_dir=root_data_dir,
    #     dataset_split_type=DatasetSplitType.TEST,
    #     order_deterministically=order_deterministically,
    #     is_debug=is_debug
    # )
    # train_tf_record_ds: tf.data.TFRecordDataset = train_tf_record_loader.get_tf_record_dataset(
    #     prefetch=False
    # )
    # val_tf_record_ds: tf.data.TFRecordDataset = val_tf_record_loader.get_batched_tf_record_dataset(
    #     batch_size=batch_size,
    #     prefetch=False
    # )
    # test_tf_record_ds: tf.data.TFRecordDataset = test_tf_record_loader.get_tf_record_dataset(
    #     prefetch=False
    # )

    # Compute the number of training samples in the dataset if not manually provided as a command line argument.
    # if num_samples_train_set is None:
    #     num_samples_train_set = determine_num_samples_in_dataset(
    #         dataset=train_tf_record_ds,
    #         dataset_split_type=DatasetSplitType.TRAIN
    #     )
    # if num_samples_val_set is None:
    #     num_samples_val_set = determine_num_samples_in_dataset(
    #         dataset=val_tf_record_ds,
    #         dataset_split_type=DatasetSplitType.VAL
    #     )
    # if num_samples_test_set is None:
    #     num_samples_test_set = determine_num_samples_in_dataset(
    #         dataset=test_tf_record_ds,
    #         dataset_split_type=DatasetSplitType.TEST
    #     )

    # Construct a placeholder tensor to store training data in memory for SVD:
    x_train: tf.Tensor = tf.zeros(
        shape=(num_samples_for_test, 4097),
        dtype=tf.dtypes.float32,
        name='x_train_mem'
    )

    # Convert the dataset iterator into a batch iterator and retrieve the entire x_train at once:
    train_tf_record_ds = train_tf_record_loader.get_batched_tf_record_dataset(
        batch_size=num_samples_for_test, prefetch=False, cache=False
    )

    # Now get the single batch from the dataset:
    x_0, _ = next(iter(train_tf_record_ds))
    # Assign it to a tensor in memory:
    x_train = tf.add(x_train, x_0, name='x_train')
    # Normalize each row so that they sum to one:
    row_norm = np.linalg.norm(x_train.numpy(), axis=1).reshape((-1, 1))
    x_train = tf.divide(x_train, row_norm)
    # x_train = tf.multiply(x_train, 100)
    # De-allocate the tf.dataset from memory:
    del train_tf_record_ds
    # TODO: Write the training data to a numpy file if too large to keep in mem.
    print('Computing closed form SVD solution on dataset (%s) a subset of %d total samples...'
          % (DatasetSplitType.TRAIN.value, DEFAULT_NUM_SAMPLES_FOR_TEST))
    s, u, v = tf.linalg.svd(x_train, full_matrices=False, compute_uv=True, name='SVD')
    print('s shape: %s' % (s.shape,))
    print('u shape: %s' % (u.shape,))
    print('v shape: %s' % (v.shape,))
    # TODO: Serialize the resulting singular vectors to disk if too large to keep in mem.
    # Now train an autoencoder on the same in-memory data:
    autoencoder = Autoencoder(input_dim=num_frequency_bins, latent_dim=num_units_latent_space)
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.MeanSquaredError(), metrics=[metrics.RootMeanSquaredError()])
    autoencoder.build(input_shape=(train_batch_size, num_frequency_bins))
    print(autoencoder.summary())

    if is_debug:
        # TODO: Enable any TensorBoard callbacks for debugging here.
        pass

    # train_tf_record_ds: tf.data.TFRecordDataset = train_tf_record_loader.get_batched_tf_record_dataset(
    #     batch_size=train_batch_size,
    #     prefetch=True,
    #     cache=True
    # )
    # val_tf_record_ds: tf.data.TFRecordDataset = val_tf_record_loader.get_batched_tf_record_dataset(
    #     batch_size=train_batch_size,
    #     prefetch=True,
    #     cache=True
    # )
    autoencoder.fit(
        x_train,
        batch_size=train_batch_size,
        epochs=num_epochs,
        verbose=1,
        callbacks=None,
        validation_data=None,
        shuffle=False,
        class_weight=None,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1
    )
    print('Trained!')
    encoder = autoencoder.layers[0]
    decoder = autoencoder.layers[1]

    # Get the latent encoding (the kernel/filter of the decoder):
    weights_encoder: tf.Variable = encoder.weights[0]
    weights_decoder: tf.Variable = decoder.weights[0]

    # Save the weights (encoder) to disk:
    output_path: str = os.path.join(output_data_dir, 'WeightsEncoder-train-%d.npy' % num_samples_for_test)
    np.save(output_path, weights_encoder.numpy().flatten())

    # Save the weights (decoder) to disk:
    output_path: str = os.path.join(output_data_dir, 'WeightsDecoder-train-%d.npy' % num_samples_for_test)
    np.save(output_path, weights_decoder.numpy().flatten())

    # Save the closed form SVD solution to disk:
    output_path: str = os.path.join(output_data_dir, 'SingularVector-train-%d.npy' % num_samples_for_test)
    right_singular_vector = v[:, 0]
    np.save(output_path, right_singular_vector)

    # Display the result to the user with an assertion:
    np.testing.assert_allclose(right_singular_vector, weights_encoder.numpy().flatten())
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
   # parser.add_argument('--desired-num-samples', type=int, action='store', dest='desired_num_samples', required=False, default=100000)
    command_line_args = parser.parse_args()
    main(args=command_line_args)