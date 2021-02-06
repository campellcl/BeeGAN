import argparse
import os
from typing import List, Iterable
from pathlib import Path
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
import tensorflow as tf


class TFRecordLoader:

    def __init__(self, root_data_dir: str, dataset_split_type: DatasetSplitType, is_debug: bool):
        self.root_data_dir: str = root_data_dir
        self.dataset_split_type: DatasetSplitType = dataset_split_type
        self.is_debug: bool = is_debug

    def get_all_dataset_split_tf_records_in_root_data_dir(self):
        tf_record_file_paths: List[str] = []

        # Get the current working dir:
        cwd = os.getcwd()
        if not os.path.isdir(self.root_data_dir):
            raise FileNotFoundError('The provided root data directory \'%s\' is invalid!' % self.root_data_dir)
        else:
            os.chdir(self.root_data_dir)
        if self.is_debug:
            print('Retrieving a list of all TFRecord files in target root_data_dir: \'%s\'' % self.root_data_dir)

        if self.dataset_split_type != DatasetSplitType.ALL:
            for file in Path('.').rglob('{}-*.tfrec'.format(self.dataset_split_type.value)):
                tf_record_file_paths.append(os.path.abspath(file))
        else:
            # If the dataset type is ALL, then run for each of the subtypes:
            for file in Path('.').rglob('{}-*.tfrec'.format(DatasetSplitType.TRAIN.value)):
                tf_record_file_paths.append(os.path.abspath(file))
            for file in Path('.').rglob('{}-*.tfrec'.format(DatasetSplitType.VAL.value)):
                tf_record_file_paths.append(os.path.abspath(file))
            for file in Path('.').rglob('{}-*.tfrec'.format(DatasetSplitType.TEST.value)):
                tf_record_file_paths.append(os.path.abspath(file))
        # Change back to the original working dir:
        os.chdir(cwd)

        return tf_record_file_paths


def _deserialize_tf_record_batch(tf_record_batch: Iterable[tf.train.Example], num_samples_in_batch: int):
    """

    :param tf_record_batch:
    :param num_samples_in_batch:
    :return:
    """

    '''
    First we have to tell TensorFlow what the format of the encoded data was. As TensorFlow has no way of knowing from
     the raw serialized byte representation the format of the original tf.train.Example before serialization to binary:
    '''
    feature_description = {
        'spectrogram': tf.io.FixedLenFeature([1], tf.string),    # The (originally 2D source Tensor) is now a serialized ByteString
        'iso_8601': tf.io.FixedLenFeature([1], tf.string)
    }

    ''' 
    Parse the batch of serialized tf.train.Example Tensor (made up of bytes) into their two serialized component Tensors 
     (one for the ISO_8601 datetime string, and one for the 2D Tensor containing the spectrogram data):
    '''
    read_example_batch = tf.io.parse_example(
        serialized=tf_record_batch,
        features=feature_description
    )

    '''
    The ISO_8601 string was serialized as a list of bytes in order to store it in the TFRecord format. We now need to
     convert that list of bytes back into a tf.string object for each tf.train.Example Tensor in the batch:
    '''
    # TODO: Map this decoding function to the batch of Tensors:
    # for tensor in read_example_batch:
    #     iso_8601_bytes_list
    # for serialized_bytes_list_iso_8601_tensor, serialized_bytes_list_spectrogram_tensor in read_example_batch:
    #     iso_8601_bytes_list_tensor: tf.Tensor = read_example_batch['iso_8601']
    #     iso_8601_tensor: tf.Tensor = tf.io.parse_tensor(
    #         serialized=iso_8601_bytes_list_tensor,
    #         out_type=tf.string,
    #         name='iso_8601'
    #     )

    return read_example_batch['iso_8601'], read_example_batch['spectrogram']


def _deserialize_tf_record(serialized_tf_record: bytes, num_samples_per_tf_record: int):
    """
    _deserialize_tf_record:
    :param serialized_tf_record:
    :param num_samples_per_tf_record:
    :return:
    """

    '''
    First we have to tell TensorFlow what the format of the encoded data was. As TensorFlow has no way of knowing from
     the raw serialized byte representation the format of the original tf.train.Example before serialization to binary:
    '''
    feature_description = {
        'spectrogram': tf.io.FixedLenFeature([], tf.string),
        # The (originally 2D source Tensor) is now a serialized ByteString
        'iso_8601': tf.io.FixedLenFeature([], tf.string)
    }

    ''' 
    Parse the single serialized tf.train.Example Tensor (made up of bytes) into its two serialized component Tensors 
     (one for the ISO_8601 datetime string, and one for the 2D Tensor containing the spectrogram data):
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
    The Spectrogram was serialized as a list of bytes in order to store the 2D source Tensor in the TFRecord format. We
     now need to convert that list of bytes back into a tf.Tensor object:
    '''
    spectrogram_bytes_list_tensor: tf.Tensor = read_example['spectrogram']
    spectrogram_tensor: tf.Tensor = tf.io.parse_tensor(
        serialized=spectrogram_bytes_list_tensor,
        out_type=tf.float32,
        name='spectrogram'
    )
    return iso_8601_tensor, spectrogram_tensor


def main(args):
    is_debug: bool = args.is_verbose
    root_data_dir: str = args.root_data_dir[0]
    dataset_split_str: str = args.dataset_split_str[0]

    dataset_split_type: DatasetSplitType
    dataset_batch_size: int = 150
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
    # tf_record_loader = TFRecordLoader(
    #     root_data_dir=root_data_dir,
    #     dataset_split_type=dataset_split_type,
    #     is_debug=is_debug
    # )
    # tf_record_file_paths: List[str] = tf_record_loader.get_all_dataset_split_tf_records_in_root_data_dir()
    # dataset = tf.data.TFRecordDataset(filenames=tf_record_file_paths, compression_type='ZLIB', num_parallel_reads=1)

    # raw_example = next(iter(dataset))
    # iso_8601_tensor, spectrogram_tensor = _deserialize_tf_record(
    #     serialized_tf_record=raw_example,
    #     num_samples_per_tf_record=num_samples_per_tf_record
    # )

    # use tensorflow to get all relevant tf record files:
    file_pattern = os.path.join(root_data_dir, '{}-*.tfrec'.format(dataset_split_type.value))
    file_dataset = tf.data.Dataset.list_files(file_pattern=file_pattern)

    # Ignore the data order to read the files faster:
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    file_dataset = file_dataset.with_options(ignore_order)

    # Read the TFRecord files in an interleaved order:
    dataset: tf.data.TFRecordDataset = tf.data.TFRecordDataset(
        filenames=file_dataset,
        compression_type='ZLIB',
        num_parallel_reads=None
    )

    # Prepare the Dataset batches:
    dataset = dataset.batch(batch_size=dataset_batch_size)

    # For debugging (see https://www.tensorflow.org/guide/effective_tf2#use_tfconfigexperimental_run_functions_eagerly_when_debugging)
    tf.config.run_functions_eagerly(True)

    # Parse a batch into a dataset of [spectra, label] pairs
    dataset = dataset.map(lambda x: _deserialize_tf_record_batch(
        tf_record_batch=x,
        num_samples_in_batch=num_samples_per_tf_record
    ))
    # A single item from the dataset is now a batch of tensors (dataset_batch_size x 1):
    raw_tf_example_batch = next(iter(dataset))
    # There will be (dataset_batch_size x 1) raw/encoded ISO 8601 tensors in the raw_tf_example_batch[0]:
    raw_iso_8601_tensor_batch = raw_tf_example_batch[0]
    # There will be (dataset_batch_size x 1) raw/encoded 2D spectrogram tensors in the raw_tf_example_batch[1]:
    raw_spectrogram_tensor_batch = raw_tf_example_batch[1]

    # raw_example = next(iter(dataset))
    print('break')
    # dataset = dataset.batch(batch_size=dataset_batch_size)
    # dataset = dataset.map(lambda x: _deserialize_tf_record_batch(x, dataset_batch_size))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorFlow TFRecord dataset loader.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--root-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    parser.add_argument('--dataset-split', type=str, nargs=1, action='store', dest='dataset_split_str', required=True,
                        help='The dataset split that should be loaded (e.g. train, test, val, or all).')
    command_line_args = parser.parse_args()
    main(args=command_line_args)
