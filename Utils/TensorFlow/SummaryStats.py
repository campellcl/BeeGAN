import sys
import os
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
from Utils.TensorFlow.TFRecordLoader import TFRecordLoader
from tqdm import tqdm
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.fft import rfftfreq


class SummaryStats(tf.Module):

    def __init__(self, train_ds: tf.data.TFRecordDataset, val_ds: tf.data.TFRecordDataset,
                 test_ds: tf.data.TFRecordDataset):
        super().__init__()
        self._train_ds: tf.data.TFRecordDataset = train_ds
        self._val_ds: tf.data.TFRecordDataset = val_ds
        self._test_ds: tf.data.TFRecordDataset = test_ds
        pass
        # self._dataset_split_type: DatasetSplitType = dataset_split_type
        # self.__running_mean: tf.Variable = tf.Variable(initial_value=)

    def mean_spectrum(self, dataset_split_type: DatasetSplitType) -> Tuple[tf.Tensor, int]:
        mean_spectra: tf.Tensor

        _freq_bin_accumulator = tf.zeros(shape=(4097,), name='freq_bin_accumulator', dtype=tf.float32)
        num_spectra: int = 0
        if dataset_split_type == DatasetSplitType.TRAIN:
            ds = self._train_ds
            for i, (x_0, _) in ds.enumerate(start=0):
                assert not isinstance(x_0, tuple)
                _freq_bin_accumulator = tf.add(_freq_bin_accumulator, x_0)
                num_spectra += 1
                # print('current freq_bin_means: %s' % tf.divide(freq_bin_means, num_samples_per_bin), end="\r", flush=True)
        else:
            raise NotImplementedError
        _num_spectra_per_freq_bin = tf.constant(value=num_spectra, shape=(4097,), dtype=tf.float32)
        mean_spectra = tf.divide(_freq_bin_accumulator, _num_spectra_per_freq_bin, name='mean_spectra')
        return mean_spectra, num_spectra

    @tf.function
    def __mean_spectrum(self):
        # TODO: implement mean spectrum computation using TF-compatible logic to leverage GPU.
        pass

    @property
    def train_ds(self) -> tf.data.TFRecordDataset:
        return self._train_ds

    @property
    def val_ds(self) -> tf.data.TFRecordDataset:
        return self._val_ds

    @property
    def test_ds(self) -> tf.data.TFRecordDataset:
        return self._test_ds

    # @property
    # def dataset_split_type(self) -> DatasetSplitType:
    #     return self._dataset_split_type


def main(args):
    # Command line arguments:
    is_debug: bool = args.is_verbose
    root_data_dir: str = args.root_data_dir[0]
    output_data_dir: str = args.output_data_dir[0]
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
    if not os.path.isdir(output_data_dir):
        raise FileNotFoundError('The provided output data directory: \'%s\' is invalid!' % output_data_dir)
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
    train_tf_record_ds: tf.data.TFRecordDataset = train_tf_record_loader.get_tf_record_dataset(
        prefetch=False
    )
    val_tf_record_ds: tf.data.TFRecordDataset = val_tf_record_loader.get_batched_tf_record_dataset(
        batch_size=batch_size
    )
    test_tf_record_ds: tf.data.TFRecordDataset = test_tf_record_loader.get_batched_tf_record_dataset(
        batch_size=batch_size
    )

    # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), '../Output'), profile_batch='0, 15')
    summary_stats: SummaryStats = SummaryStats(
        train_ds=train_tf_record_ds,
        val_ds=val_tf_record_ds,
        test_ds=test_tf_record_ds
    )
    # cardinality = train_tf_record_ds.cardinality().numpy()
    # if cardinality == tf.data.UNKNOWN_CARDINALITY:
    #     print('Cannot implicitly determine number of samples in the provided TF dataset (train), tf.data reports '
    #           'an unknown cardinality.')
    # else:
    #     print('Number of Samples (train): %s' % cardinality)

    mean_spectra = np.load(os.path.join(output_data_dir, 'mean_spectra-train-1282034.npy'))
    f = rfftfreq(8192, 1/8000)
    max_idx = np.argmax(mean_spectra)
    plt.plot(f, mean_spectra)
    print('f max-idx: %s' % f[max_idx])
    plt.show()

    # TODO: Benchmarking does not appear to work here via tfds:
    # print('Benchmarking training dataset...')
    # tfds_benchmark = train_tf_record_loader.benchmark_dataset(ds=train_tf_record_ds, batch_size=batch_size)
    # print('Computing the mean spectrum ')
    # train_tf_record_loader.plot_batch_sizes(train_tf_record_ds)

    # print('Computing the mean spectra for dataset (%s)...' % dataset_split_type.value)
    # mean_spectra, num_spectra = summary_stats.mean_spectrum(dataset_split_type=DatasetSplitType.TRAIN)
    # print('mean_spectra: %s' % mean_spectra)
    # print('mean_spectra shape: %s' % (mean_spectra.shape,))
    # print('mean_spectra was computed from: %d samples encountered in dataset (%s)' % (num_spectra, dataset_split_type.value))
    # output_mean_spectra_file_path = os.path.join(output_data_dir, '{}-{}-{}.npy'.format('mean_spectra', dataset_split_type.value, num_spectra))
    # print('Serializing...')
    # mean_spectra_numpy_repr = mean_spectra.numpy()
    # np.save(output_mean_spectra_file_path, mean_spectra_numpy_repr, allow_pickle=False, fix_imports=False)
    # print('Successfully serialized to: \'%s\'' % output_mean_spectra_file_path)


    # print('Determining how many cached dataset (train) batches can fit into GPU memory before OOM errors...')
    # train_tf_record_ds = train_tf_record_loader.get_batched_tf_record_dataset(
    #     batch_size=batch_size,
    #     prefetch=True,
    #     cache=False
    # )
    # for i, batch in enumerate(train_tf_record_ds.as_numpy_iterator()):
    #     print('batch: %d' % i)

    # in_mem: list = list(train_tf_record_ds.as_numpy_iterator())
    # print('Dataset shape in memory: (%s, %s)' % (len(in_mem), len(in_mem[0])))

    # i = 0
    # for train_batch in train_tf_record_ds:
    #     print('train_batch: %3d' % i)
    #     i += 1
    # for i in range(100):
    #     print('batch: %3d' % i)
    #     train_tf_record_ds.take(batch_size).cache()
    # sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder argument parser.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--root-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    parser.add_argument('--dataset-split', type=str, nargs=1, action='store', dest='dataset_split_str', required=True,
                        help='The dataset split that should be loaded (e.g. train, test, val, or all).')
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size', required=True)
    parser.add_argument('--order-deterministically', type=bool, action='store', dest='order_deterministically',
                        required=True)
    parser.add_argument('--output-data-dir', type=str, nargs=1, action='store', dest='output_data_dir', required=True)
    command_line_args = parser.parse_args()
    main(args=command_line_args)