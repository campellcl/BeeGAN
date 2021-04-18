import argparse
import os
from typing import Tuple, Optional
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
import tensorflow as tf
import tensorflow_datasets as tfds


class TFRecordLoader:
    """
    Loads and de-serializes TFRecord files and provides access to the deserialized TFRecord Dataset objects.
    """

    def __init__(self, root_data_dir: str, is_debug: bool):
        """
        __init__: Initializer for objects of type TFRecordLoader.
        :param root_data_dir:
        :param is_debug: <bool> A boolean flag indicating if additional print statements should be displayed to console.
        """
        self._root_data_dir: str = root_data_dir
        # Ensure the root_data_dir is valid:
        if not os.path.isdir(self.root_data_dir):
            raise FileNotFoundError('The provided root data directory: \'%s\' is invalid!' % self.root_data_dir)
        self.is_debug: bool = is_debug

    def get_tf_record_dataset(self, dataset_split_type: DatasetSplitType, order_deterministically: bool,
                              prefetch: bool = True, cache: bool = False) -> tf.data.TFRecordDataset:
        """
        get_tf_record_dataset: Convenience method which returns an instance of a TFRecordLoader object for the specified
         dataset.
        :param dataset_split_type: <DatasetSplitType> Either a TRAIN, VAL, or TEST enumerated type (from the
         BeeGAN.Utils.EnumeratedTypes.DatasetSplitType module) representing the current split/partition of the dataset.
        :param order_deterministically: <bool> A boolean flag which indicates whether the outputs from the dataset are
         to be produced in deterministic order during iteration (see:
         https://www.tensorflow.org/api_docs/python/tf/data/Options). For optimal performance, disable this flag and
         read multiple files at once while disregarding the order of the data. If you plan to shuffle the data anyway,
         then it makes sense to consider reading non-deterministically.
        :param prefetch: <bool> A boolean flag indicating if prefetching should be enabled. Prefetching allows later
         elements to be prepared while the current element is being processed. This often improves latency and
         throughput, at the cost of using additional memory to store prefetched elements (see:
         https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch).
        :param cache: <bool> A boolean flag indicating if elements in the dataset should be cached to memory. WARNING:
         Do not attempt to cache datasets that have a size larger than GPU memory, in memory; instead modify this method
         to accept a filename to cache to, and take over clearing the cache on subsequent iterations (see:
          https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache).
        :return:
        """
        dataset: tf.data.TFRecordDataset

        # Get all TFRecord files belonging to the specified dataset split ('train', 'test', 'val', 'all'):
        if dataset_split_type == DatasetSplitType.ALL:
            file_pattern = os.path.join(self.root_data_dir, '*-*.tfrec')
        else:
            file_pattern = os.path.join(self.root_data_dir, '{}-*.tfrec'.format(dataset_split_type.value))
        file_dataset = tf.data.Dataset.list_files(file_pattern=file_pattern)

        # This boolean flag indicates whether the outputs from the dataset need to be produced in deterministic order
        #   (see: https://www.tensorflow.org/api_docs/python/tf/data/Options). For optimal performance, enable this flag
        # and read multiple files at once while disregarding the order of the data. If you plan to shuffle the data
        # anyway, then it makes sense to consider reading non-deterministically.
        tf_data_options = tf.data.Options()
        tf_data_options.experimental_deterministic = order_deterministically
        file_dataset = file_dataset.with_options(options=tf_data_options)

        # Read the raw binary TFRecord files into a dataset:
        dataset: tf.data.TFRecordDataset = tf.data.TFRecordDataset(
            filenames=file_dataset,
            compression_type='ZLIB',
            num_parallel_reads=tf.data.experimental.AUTOTUNE
        )
        # De-serialize the DS of tf.train.Examples into tuple (tf.Tensor(tf.string), tf.Tensor(tf.float32 1D array)):
        dataset = dataset.map(
            lambda x: self.deserialize_tf_record(serialized_tf_record=x),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        if prefetch:
            if cache:
                dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE).cache()
            else:
                dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            if cache:
                dataset = dataset.cache()
            else:
                pass
        return dataset

    def get_batched_tf_record_dataset(
            self, dataset_split_type: DatasetSplitType, order_deterministically: bool, batch_size: int,
            prefetch: bool = True, cache: bool = False) -> tf.data.TFRecordDataset:
        """
        get_batched_tf_record_dataset: Retrieves the tf.data.TFRecordDataset object for the specified dataset split (
         e.g. 'train', 'test', 'val', 'all').
        WARNING: Improper use of this method will exhaust GPU memory resources and crash TensorFlow (see below).
        :param dataset_split_type: <DatasetSplitType> Either a TRAIN, VAL, or TEST enumerated type (from the
         BeeGAN.Utils.EnumeratedTypes.DatasetSplitType module) representing the current split/partition of the dataset.
        :param order_deterministically: <bool> A boolean flag which indicates whether the outputs from the dataset are
         to be produced in deterministic order during iteration (see:
         https://www.tensorflow.org/api_docs/python/tf/data/Options). For optimal performance, disable this flag and
         read multiple files at once while disregarding the order of the data. If you plan to shuffle the data anyway,
         then it makes sense to consider reading non-deterministically.
        :param batch_size: <int> The batch size that the iterator of the dataset should yield in each step.
        :param prefetch: <bool> A boolean flag indicating if prefetching should be enabled. Prefetching allows later
         elements to be prepared while the current element is being processed. This often improves latency and
         throughput, at the cost of using additional memory to store prefetched elements (see:
         https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch).
        :param cache: <bool> A boolean flag indicating if elements in the dataset should be cached to memory. WARNING:
         Do not attempt to cache datasets that have a size larger than GPU memory, in memory; instead modify this method
         to accept a filename to cache to, and take over clearing the cache on subsequent iterations (see:
          https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache).
        :return dataset: <tf.data.TFRecordDataset> A tf.data.TFRecordDataset (see:
         https://www.tensorflow.org/tutorials/load_data/tfrecord) which is the recommended data storage format for
         Tensorflow 2.0 with datasets that will not fit into memory (see:
         https://www.tensorflow.org/guide/effective_tf2#combine_tfdatadatasets_and_tffunction).
        """
        dataset: tf.data.TFRecordDataset

        # Ensure the root_data_dir is valid:
        if not os.path.isdir(self.root_data_dir):
            raise FileNotFoundError('The provided root data directory: \'%s\' is invalid!' % self.root_data_dir)
        # Get all TFRecord files belonging to the specified dataset split ('train', 'test', 'val', 'all'):
        if dataset_split_type == DatasetSplitType.ALL:
            file_pattern = os.path.join(self.root_data_dir, '*-*.tfrec')
        else:
            file_pattern = os.path.join(self.root_data_dir, '{}-*.tfrec'.format(dataset_split_type.value))
        file_dataset = tf.data.Dataset.list_files(file_pattern=file_pattern)

        # This boolean flag indicates whether the outputs from the dataset need to be produced in deterministic order
        #   (see: https://www.tensorflow.org/api_docs/python/tf/data/Options). For optimal performance, disable this
        # flag and read multiple files at once while disregarding the order of the data. If you plan to shuffle the data
        # anyway, then it makes sense to consider reading non-deterministically.
        tf_data_options = tf.data.Options()
        tf_data_options.experimental_deterministic = order_deterministically
        file_dataset = file_dataset.with_options(options=tf_data_options)

        # Read the raw binary TFRecord files into a dataset:
        dataset: tf.data.TFRecordDataset = tf.data.TFRecordDataset(
            filenames=file_dataset,
            compression_type='ZLIB',
            num_parallel_reads=tf.data.experimental.AUTOTUNE
        )
        # De-serialize the DS of tf.train.Examples into tuple (tf.Tensor(tf.string), tf.Tensor(tf.float32 1D array)):
        dataset = dataset.map(
            lambda x: self.deserialize_tf_record(serialized_tf_record=x),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        # Pre-split the Dataset into batches for training:
        if cache:
            if prefetch:
                dataset = dataset.batch(batch_size=batch_size, drop_remainder=True).cache().prefetch(
                    buffer_size=tf.data.AUTOTUNE
                )
            else:
                dataset = dataset.batch(batch_size=batch_size, drop_remainder=True).cache()
        else:
            if prefetch:
                dataset = dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(
                    buffer_size=tf.data.AUTOTUNE
                )
            else:
                dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

        # A single item from the dataset is now a batch of tensors (dataset_batch_size x 1):
        # tf_example_batch = next(iter(dataset))

        # There will be (dataset_batch_size x 1) raw/encoded ISO 8601 tensors in the tf_example_batch[0]:
        # iso_8601_tensor_batch = tf_example_batch[0]

        # There will be (dataset_batch_size x 1) raw/encoded 2D spectrogram tensors in the tf_example_batch[1]:
        # spectrogram_tensor_batch = tf_example_batch[1]
        return dataset

    def get_dataset_subset_from_memory(self, dataset_split_type: DatasetSplitType, order_deterministically: bool,
                                       num_samples: int, num_frequency_bins: int) -> tf.Tensor:
        """
        get_dataset_subset_from_memory: Loads the specified number of samples from the TFRecordDataset object (via the
         disk) into memory.
        WARNING: Improper usage of this method may exhaust system RAM and lock up the host machine. You must ensure that
         the size of the dataset you have requested (in 32 bit float samples) will fit into memory.
        :param dataset_split_type: <DatasetSplitType> Either a TRAIN, VAL, or TEST enumerated type (from the
         BeeGAN.Utils.EnumeratedTypes.DatasetSplitType module) representing the current split/partition of the dataset.
        :param order_deterministically: <bool>
        :param num_samples:
        :param num_frequency_bins: <int> The number of frequency bins (e.g. the length of a row in the specified
         TFRecord dataset). Typically this value could be computed with data.shape[1], but with TFRecordDatasets the
         cardinality is not implicitly known in advance, so here it must be supplied manually.
        :return data: <tf.Tensor> A tensor containing the requested dataset subset in memory. The subset returned will
         be size (num_samples x num_freq_bins).
        """
        data: tf.Tensor

        # Construct a placeholder tensor to store data in memory:
        data: tf.Tensor = tf.zeros(
            shape=(num_samples, num_frequency_bins),
            dtype=tf.dtypes.float32,
            name='x_%s_mem' % dataset_split_type.value
        )

        # Construct a batch iterator and retrieve the entire num_samples x num_freq in-memory-dataset at once:
        tf_record_batch_ds = self.get_batched_tf_record_dataset(
            dataset_split_type=dataset_split_type,
            order_deterministically=order_deterministically,
            batch_size=num_samples,
            prefetch=True,
            cache=False
        )
        # Now get the single batch from the dataset:
        x_0, _ = next(iter(tf_record_batch_ds))
        # Assign it to a tensor in memory:
        data = tf.add(data, x_0, name='x_%s_mem' % dataset_split_type.value)

        # Manually flag the tf.dataset for de-allocation from memory:
        del tf_record_batch_ds
        return data

    @property
    def root_data_dir(self) -> str:
        return self._root_data_dir

    @staticmethod
    def deserialize_tf_record(serialized_tf_record: bytes) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        deserialize_tf_record: Takes in a serialized TFRecord file, and de-serializes the TFRecord into a tf.Tensor.
         Then this method will de-serialize the ByteList objects to return a tuple containing two Tensors:
            1. A tf.string Tensor containing the ISO 8601 datetime associated with the sample.
            2. A tf.float32 1D Tensor containing a single frequency sample of the spectrogram for a single audio sample.
        :param serialized_tf_record: <bytes> A raw TFRecord file which was previously created by serializing a
         tf.train.Example.
        :returns iso_8601_tensor, frequencies_tensor: <tuple<tf.Tensor, tf.Tensor>> A tuple containing a tf.string
         Tensor which holds the ISO 8601 datetime representation associated with the audio file, and a tf.float32 tensor
          which contains the 1D spectrogram sample associated with the original source audio file.
        """

        '''
        First we have to tell TensorFlow what the format of the encoded data was. As TensorFlow has no way of knowing 
         from the raw serialized byte representation the format of the original tf.train.Example before serialization 
         to binary:
        '''
        feature_description = {
            # 2D Tensors must be flattened and encoded as a ByteString, 1D tensors are encoded as a ByteString as well:
            'frequencies': tf.io.FixedLenFeature([], tf.string),
            # The (originally 1D source Tensor generated from str obj) is now a serialized ByteString:
            'iso_8601': tf.io.FixedLenFeature([], tf.string)
        }

        ''' 
        Parse the single serialized tf.train.Example Tensor (made up of bytes) into its two serialized component Tensors 
         (one for the ISO_8601 datetime string, and one for the 1D Tensor containing the sample frequency data):
        '''
        read_example: dict = tf.io.parse_example(
            serialized=serialized_tf_record,
            features=feature_description
        )

        '''
        The ISO_8601 string was serialized as a list of bytes in order to store it in the TFRecord format. We now need to
         convert that list of bytes back into a tf.string object:
        '''
        # iso_8601_bytes_list_tensor: tf.Tensor = read_example['iso_8601']
        # iso_8601_tensor: tf.Tensor = tf.io.parse_tensor(
        #     serialized=iso_8601_bytes_list_tensor,
        #     out_type=tf.string,
        #     name='iso_8601'
        # )
        # The https://www.tensorflow.org/tutorials/load_data/unicode#the_tfstring_data_type by default stores the string in
        #   numpy as a binary encoding of the UTF-8 character representation. To decode the byte representation into UTF-8
        #   character codes we can optionally do:
        # tf.strings.unicode_decode(iso_8601_tensor, input_encoding='UTF-8')

        '''
        The Spectrogram was serialized as a list of bytes in order to store the 1D source Tensor in the TFRecord format.
         We now need to convert that list of bytes back into a tf.Tensor object:
        '''
        frequencies_bytes_list_tensor: tf.Tensor = read_example['frequencies']
        frequencies_tensor: tf.Tensor = tf.io.parse_tensor(
            serialized=frequencies_bytes_list_tensor,
            out_type=tf.float32,
            name='frequencies'
        )
        return frequencies_tensor, frequencies_tensor

    @staticmethod
    def benchmark_dataset(ds: tf.data.Dataset, batch_size):
        tfds_benchmark = tfds.core.benchmark(ds=ds, num_iter=1, batch_size=batch_size)
        return tfds_benchmark

    @staticmethod
    def determine_num_samples_in_dataset(dataset: tf.data.Dataset, dataset_split_type: DatasetSplitType) -> int:
        """
        determine_num_samples_in_dataset: Manually computes the number of samples by iterating over all elements (or
         batches of elements) in the provided TFRecordDataset.
        :param dataset: <tf.data.TFRecordDataset> The dataset to determine the cardinality of.
        :param dataset_split_type: <DatasetSplitType> Either a TRAIN, VAL, or TEST enumerated type (from the
         BeeGAN.Utils.EnumeratedTypes.DatasetSplitType module) representing the current split/partition of the dataset.
        :return num_samples: <int> The number of samples in the provided dataset.
        """
        num_samples: int = -1
        assert dataset_split_type != DatasetSplitType.ALL, dataset_split_type

        num_samples = dataset.cardinality().numpy()
        if num_samples == tf.data.UNKNOWN_CARDINALITY:
            num_samples: int = 0
            for i, (x_0, _) in dataset.enumerate(start=0):
                if len(x_0.shape) == 1:
                    # The dataset is not batched and this is a single sample.
                    num_samples += 1
                else:
                    # The dataset is batched and this is a batch of samples.
                    num_samples += x_0.shape[0]
        return num_samples

    # @staticmethod
    # def plot_batch_sizes(ds: tf.data.Dataset):
    #     # tf_example_batch = next(iter(ds))
    #     # spectrogram_tensor_batch = tf_example_batch[1]
    #
    #     for batch in ds:
    #         freqs = batch[0]
    #
    #
    #     # batch_sizes = [batch[0].shape[0] for batch in ds]
    #     # plt.bar(range(len(batch_sizes)), batch_sizes)
    #     # plt.xlabel('Batch number')
    #     # plt.ylabel('Batch size')


def main(args):
    is_debug: bool = args.is_verbose
    root_data_dir: str = args.root_data_dir[0]
    dataset_split_str: str = args.dataset_split_str[0]
    dataset_batch_size: int = args.batch_size
    order_deterministically: bool = args.order_deterministically

    dataset_split_type: DatasetSplitType
    num_samples_per_tf_record: int = 150

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

    tf_record_loader = TFRecordLoader(
        root_data_dir=root_data_dir,
        is_debug=is_debug
    )
    dataset = tf_record_loader.get_batched_tf_record_dataset(
        dataset_split_type=dataset_split_type,
        order_deterministically=order_deterministically,
        batch_size=dataset_batch_size,
        prefetch=False,
        cache=False
    )
    # A single item from the dataset is now a batch of tensors (dataset_batch_size x 1):
    tf_example_batch = next(iter(dataset))

    # There will be (dataset_batch_size x 1) raw/encoded ISO 8601 tensors in the tf_example_batch[0]:
    iso_8601_tensor_batch = tf_example_batch[0]

    # There will be (dataset_batch_size x 4097) raw/encoded 2D spectrogram tensors in the tf_example_batch[1]:
    spectrogram_tensor_batch = tf_example_batch[1]

    # A single sample by default size (4097, 1) is hence:
    sample = spectrogram_tensor_batch[0]
    # And it's corresponding ISO 8601 string of size () is hence:
    sample_iso_8601_str = iso_8601_tensor_batch[0]
    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorFlow TFRecord dataset loader.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--root-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    parser.add_argument('--dataset-split', type=str, nargs=1, action='store', dest='dataset_split_str', required=True,
                        help='The dataset split that should be loaded (e.g. train, test, val, or all).')
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size', required=True)
    parser.add_argument('--order-deterministically', type=bool, action='store', dest='order_deterministically', required=True)
    command_line_args = parser.parse_args()
    main(args=command_line_args)
