import os
import sys
import argparse
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
from Utils.TensorFlow.TFRecordLoader import TFRecordLoader
import numpy as np
import tensorflow as tf
from PCAGAN.Encoder import Autoencoder
from typing import Optional, Union

DEFAULT_NUM_SAMPLES_FOR_TEST = 30000   # IMPORTANT: You must manually verify this amount of 32-bit float samples will
# fit on the system dedicated GPU memory.


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
    batch_size: int = args.batch_size
    order_deterministically: bool = args.order_deterministically
    num_samples_train_set: Optional[int] = args.num_samples_train_set
    num_samples_val_set: Optional[int] = args.num_samples_val_set
    num_samples_test_set: Optional[int] = args.num_samples_test_set

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
    # Determine the maximum number of batch sizes we can iterate over and still have enough memory to compute closed
    # form SVD:
    # if num_samples_train_set > DEFAULT_NUM_SAMPLES_FOR_TEST:
    #     max_num_batches_for_training_to_fit_in_mem: int = DEFAULT_NUM_SAMPLES_FOR_TEST // batch_size
    #     # Now we have the number of batches we can train on to still compute a comparable closed form solution in memory.
    #     num_elements_for_training_and_svd: int = max_num_batches_for_training_to_fit_in_mem * batch_size
    # else:
    #     max_num_batches_for_training_to_fit_in_mem: int = num_samples_train_set // batch_size
    #     num_elements_for_training_and_svd: int = num_samples_train_set

    # Read the data into memory for SVD:
    x_train: tf.Tensor = tf.zeros(
        shape=(DEFAULT_NUM_SAMPLES_FOR_TEST, 4097),
        dtype=tf.dtypes.float32,
        name='x_train_mem'
    )

    # Convert the dataset iterator into a batch iterator and retrieve the entire x_train at once:
    train_tf_record_ds = train_tf_record_loader.get_batched_tf_record_dataset(batch_size=DEFAULT_NUM_SAMPLES_FOR_TEST, prefetch=False, cache=False)

    # Now get the single batch from the dataset:
    x_0, _ = next(iter(train_tf_record_ds))
    # Assign it to a tensor in memory:
    x_train = tf.add(x_train, x_0, name='x_train')
    # De-allocate the tf.dataset from memory:
    del train_tf_record_ds
    # TODO: Write the training data to a numpy file:


    # # x_train: tf.Tensor
    # x_train_write_idx: int = 0
    # for i in range(max_num_batches_for_training_to_fit_in_mem):
    #     x_0, _ = next(iter(train_tf_record_ds))
    #     # Populate a subset of x_train tensor with the new batch read in from the dataset iterator:
    #     # indices = tf.constant(value=[m for m in ])
    #     if i == 0:
    #         x_train = x_0
    #     else:
    #         x_train = tf.stack([x_train, x_0])

    print('Computing closed form SVD solution on dataset (%s) a subset of %d total samples...'
          % (DatasetSplitType.TRAIN, DEFAULT_NUM_SAMPLES_FOR_TEST))
    s, u, v = tf.linalg.svd(x_train, full_matrices=False, compute_uv=True, name='SVD')
    print('s shape: %s' % (s.shape,))
    print('u shape: %s' % (u.shape,))
    print('v shape: %s' % (v.shape,))
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder argument parser.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--root-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    parser.add_argument('--order-deterministically', type=bool, action='store', dest='order_deterministically',
                        required=True)
    parser.add_argument('--output-data-dir', type=str, nargs=1, action='store', dest='output_data_dir', required=True)
    # TODO: Add warning that the batch size will be used to compute the
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size', required=True)
    parser.add_argument('--num-samples-train-set', type=int, action='store', dest='num_samples_train_set', required=False, default=None)
    parser.add_argument('--num-samples-val-set', type=int, action='store', dest='num_samples_val_set', required=False, default=None)
    parser.add_argument('--num-samples-test-set', type=int, action='store', dest='num_samples_test_set', required=False, default=None)
    # parser.add_argument('--desired-num-samples', type=int, action='store', dest='desired_num_samples', required=False, default=100000)
    command_line_args = parser.parse_args()
    main(args=command_line_args)