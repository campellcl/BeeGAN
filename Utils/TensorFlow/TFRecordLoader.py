import argparse
import os
from typing import Tuple
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
import tensorflow as tf


class TFRecordLoader:
    """
    Loads and de-serializes TFRecord files and provides access to the deserialized TFRecord Dataset objects.
    """

    def __init__(self, root_data_dir: str, dataset_split_type: DatasetSplitType, order_deterministically: bool,
                 is_debug: bool):
        """
        __init__: Initializer for objects of type TFRecordLoader.
        :param root_data_dir:
        :param dataset_split_type:
        :param order_deterministically: <bool> A boolean flag which indicates whether the outputs from the dataset are
         to be produced in deterministic order during iteration (see:
         https://www.tensorflow.org/api_docs/python/tf/data/Options). For optimal performance, enable this flag and
         read multiple files at once while disregarding the order of the data. If you plan to shuffle the data anyway,
         then it makes sense to consider reading non-deterministically.
        :param is_debug:
        """
        self._root_data_dir: str = root_data_dir
        self._dataset_split_type: DatasetSplitType = dataset_split_type
        self._order_deterministically: bool = order_deterministically
        self.is_debug: bool = is_debug

    def get_tf_record_dataset(self, batch_size: int) -> tf.data.TFRecordDataset:
        """
        get_tf_record_dataset: Retrieves the tf.data.TFRecordDataset object for the specified dataset split (e.g.
         'train', 'test', 'val', 'all').
        :param batch_size: <int> The batch size that the iterator of the dataset should yield in each step.
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
        if self.dataset_split_type == DatasetSplitType.ALL:
            file_pattern = os.path.join(self.root_data_dir, '*-*.tfrec')
        else:
            file_pattern = os.path.join(self.root_data_dir, '{}-*.tfrec'.format(self.dataset_split_type.value))
        file_dataset = tf.data.Dataset.list_files(file_pattern=file_pattern)

        # This boolean flag indicates whether the outputs from the dataset need to be produced in deterministic order
        #   (see: https://www.tensorflow.org/api_docs/python/tf/data/Options). For optimal performance, enable this flag
        # and read multiple files at once while disregarding the order of the data. If you plan to shuffle the data
        # anyway, then it makes sense to consider reading non-deterministically.
        tf_data_options = tf.data.Options()
        tf_data_options.experimental_deterministic = self._order_deterministically
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
        # dataset = dataset.batch(batch_size=batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
        dataset = dataset.batch(batch_size=batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        # A single item from the dataset is now a batch of tensors (dataset_batch_size x 1):
        # tf_example_batch = next(iter(dataset))

        # There will be (dataset_batch_size x 1) raw/encoded ISO 8601 tensors in the tf_example_batch[0]:
        # iso_8601_tensor_batch = tf_example_batch[0]

        # There will be (dataset_batch_size x 1) raw/encoded 2D spectrogram tensors in the tf_example_batch[1]:
        # spectrogram_tensor_batch = tf_example_batch[1]
        return dataset

    @property
    def order_deterministically(self) -> bool:
        return self._order_deterministically

    @property
    def dataset_split_type(self) -> DatasetSplitType:
        return self._dataset_split_type

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
            'frequencies': tf.io.FixedLenFeature([], tf.string),
            # The (originally 1D source Tensor) is now a serialized ByteString
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
        iso_8601_bytes_list_tensor: tf.Tensor = read_example['iso_8601']
        iso_8601_tensor: tf.Tensor = tf.io.parse_tensor(
            serialized=iso_8601_bytes_list_tensor,
            out_type=tf.string,
            name='iso_8601'
        )
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
        dataset_split_type=dataset_split_type,
        order_deterministically=order_deterministically,
        is_debug=is_debug
    )
    dataset = tf_record_loader.get_tf_record_dataset(
        batch_size=dataset_batch_size
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
