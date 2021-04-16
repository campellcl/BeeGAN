import os
import sys
import argparse
from typing import List, Tuple, Union, Dict, Set, Optional
from pathlib import Path
import numpy as np
import math
import tensorflow as tf
import librosa
from scipy import signal
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import datetime
from datetime import timedelta
import pandas as pd
from dateutil.parser import isoparse
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
from Utils.BlacklistedSamples import blacklisted_samples
from Utils.DatetimeUtils import DatetimeUtils


_DEFAULT_AUDIO_DURATION = 60     # seconds
_DEFAULT_SAMPLE_RATE = 8000     # 8 hz
_DEFAULT_SEED = 42
_DEFAULT_TEST_SIZE = .20        # 20%
_DEFAULT_VAL_SIZE = .20         # 20%
MAXIMUM_SHARD_SIZE = 200        # in bytes (200 MB)


def _bytes_feature(value):
    # The following functions can be used to convert a value to a type compatible
    # with tf.train.Example.
    """Returns a bytes_list from a string / byte."""
    # see: https://www.tensorflow.org/tutorials/load_data/tfrecord#tftrainexample
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _tf_float_feature(floats: List[Union[float, np.double]]):
    # The following functions can be used to convert a value to a type compatible
    # with tf.train.Example.
    """Returns a float_list from a float / double."""
    # see: https://www.tensorflow.org/tutorials/load_data/tfrecord#tftrainexample
    return tf.train.Feature(float_list=tf.train.FloatList(value=floats))


def _int64_feature(value):
    # The following functions can be used to convert a value to a type compatible
    # with tf.train.Example.
    """Returns an int64_list from a bool / enum / int / uint."""
    # see: https://www.tensorflow.org/tutorials/load_data/tfrecord#tftrainexample
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _parallelize(func, data):
    """
    _parallelize: Basic data parallelism via subprocess orchestrated by a multiprocessing Pool. This method applies the
     provided function (funct) to every item of the provided iterable (data).
    see: https://docs.python.org/3.8/library/multiprocessing.html#multiprocessing.pool.Pool
    see: https://docs.python.org/3.8/library/multiprocessing.html#multiprocessing.pool.Pool.imap_unordered
    Source: https://gist.github.com/dschwertfeger/3288e8e1a2d189e5565cc43bb04169a1#file-convert-py-L180
    :param func: <func> The function to apply to the specified data.
    :param data: <iterable> The iterable data which the specified function should be applied/mapped to.
    :return None:
    """
    num_available_cpu_cores = cpu_count() - 1
    # Provide the number of worker processes to use to Pool during instantiation (save a core for the parent process):
    with Pool(num_available_cpu_cores) as pool:
        # To display progress with tqdm we need an enclosing list statement (per https://stackoverflow.com/a/45276885/1663506):
        list(tqdm(pool.imap_unordered(func, data), total=len(data)))


class ConvertWAVToTFRecord:
    """
    ConvertWAVToTFRecord: This script creates a list of all *.wav audio files in the specified input directory and then
     writes TFRecord shards to the specified output directory, in parallel, leveraging the number of CPU cores on the
     executing machine, and estimates the ideal maximum size of the TFRecord shards in accordance with the
     recommendations provided by the TensorFlow documentation: https://www.tensorflow.org/tutorials/load_data/tfrecord
    """

    def __init__(self, root_data_dir: str, output_data_dir: str, audio_duration: int, sample_rate: int,
                 test_size: float, val_size: float, is_debug: bool, seed: int):
        """
        __init__: Initializers for objects of type ConvertWAVToTFRecord.
        :param root_data_dir:
        :param output_data_dir:
        :param audio_duration: <int> The desired duration of the source audio samples (in seconds). Audio files that are
         longer than the specified duration will be truncated. Audio files that are shorter than the specified duration
         will be zero-padded.
        :param sample_rate:
        :param test_size: <float> The percentage of the total dataset that should be allocated to the test set.
        :param val_size: <float> The percentage of the total dataset that should be allocated to the validation set.
        :param is_debug:
        see: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        """
        self._root_data_dir: str = root_data_dir
        self._output_data_dir: str = output_data_dir
        self._audio_duration: int = audio_duration
        self._sample_rate: int = sample_rate
        self._test_size: float = test_size
        self._val_size: float = val_size
        self._is_debug: bool = is_debug
        self._seed: int = seed
        np.random.seed(self._seed)
        self._audio_file_paths: List[str] = self.get_audio_file_paths_in_root_data_dir(
            blacklisted_filenames=blacklisted_samples
        )
        self._num_audio_files: int = len(self._audio_file_paths)
        self._num_samples: Optional[int] = None
        # Create a metadata dataframe by augmenting the list of all audio file paths with time information (from the
        # file names):
        self._beemon_df: pd.DataFrame = self._create_beemon_metadata_df()
        '''
        Pre-Process a single sample/audio file to determine the shape of the produced spectrogram given the current 
         runtime settings. This information is required to pre-compute the shard size of the resulting TFRecords:
        '''
        sample_spectrogram = self._beemon_df.iloc[0]
        audio_file_path: str = sample_spectrogram['file_path']
        spectrogram_iso_8601: str = sample_spectrogram['iso_8601']
        # Read in the audio at the specified file path while re-sampling it at the specified rate:
        audio, sample_rate = librosa.load(audio_file_path, sr=self.sample_rate)
        freqs, time_segs, spectrogram = self.apply_fourier_transform(audio_sample=audio)
        self._spectrogram_shape = spectrogram.shape
        self._num_samples_per_spectrogram: int = spectrogram.shape[1]
        # A single column of the spectrogram (a 4097 length 1D vector by default) is a single sample for our use case:
        sample = spectrogram[:, 0]


        # Calculate the ISO timestamp offset for the sample:
        sample_iso_8601: datetime.datetime = self._get_sample_iso_timestamp_offset(
            spectrogram_iso_8601=spectrogram_iso_8601,
            sample_index=0,
            time_segments=time_segs
        )
        # Now we can compute how many samples will fit within an individual TFRecord:
        self._max_num_samples_per_tf_record_file: int = self._compute_maximum_num_samples_per_tf_record(
            sample=sample,
            sample_iso_8601=sample_iso_8601
        )
        # These object properties are initially None (on instantiation), but are set automatically during the sharding
        # process. Perform NoneType checks before utilizing them prior to sharding.
        self._num_spectra: Optional[int] = None
        self._num_train_spectra: Optional[int] = None
        self._num_train_samples: Optional[int] = None

        self._num_val_spectra: Optional[int] = None
        self._num_val_samples: Optional[int] = None

        self._num_test_samples: Optional[int] = None
        self._num_test_spectra: Optional[int] = None

        self._num_train_shards: Optional[int] = None
        self._num_val_shards: Optional[int] = None
        self._num_test_shards: Optional[int] = None

    def _get_sample_iso_timestamp_offset(
            self, spectrogram_iso_8601: str, sample_index: int, time_segments: np.ndarray) -> datetime.datetime:
        """
        _get_sample_iso_timestamp_offset: Calculates the ISO 8601 timestamp of a particular sample from the spectrogram
         by offsetting the ISO 8601 timestamp for the entire audio source by the length of an individual time segment in
         the spectrogram, and the index of the provided time segment bin that the sample was taken from.
        :param spectrogram_iso_8601: <str> The ISO 8601 timestamp corresponding to the entire spectrogram of the source
         audio file.
        :param sample_index: <int> The index (by default a range of [0-65] inclusive) of the time segment of the
         spectrogram that the sample (whose ISO timestamp is to be determined) was taken from. This corresponds to the
         column index in the source 2D spectrogram.
        :param time_segments: <float> The list of time segments (in seconds) produced by scipy.signal.spectrogram
         corresponding to when the FFT was computed for the corresponding list of frequencies.
        :return:
        """
        sample_iso_8601: object
        spectrogram_iso_timestamp: datetime.datetime = isoparse(spectrogram_iso_8601)
        delta = timedelta(seconds=time_segments[sample_index])
        sample_iso_8601: datetime.datetime = spectrogram_iso_timestamp + delta
        return sample_iso_8601

    @staticmethod
    def convert_sample_to_tf_example(sample: np.ndarray, iso_8601_str: str) -> tf.train.Example:
        """
        convert_sample_to_tf_example: Takes the native data format of a single sample in the dataset and performs the
         conversion of the audio spectrogram and associated ISO 8601 string to a tf.train.Example item which can then
         be further serialized and written (in batch) to a TFRecord file downstream.
        :param sample: <np.ndarray> A 1D numpy array representing a single column of a spectrogram of a single
         audio file sample. By default a single sample will be size (4097,).
        :param iso_8601_str: <str> The ISO 8601 datetime string corresponding to the sample. This value is produced by
         parsing the name of the audio file, which was used to generate the spectrogram (supplied in conjunction to this
         method) and offsetting by the time bin of the individual sample.
        :source: https://www.kaggle.com/ryanholbrook/tfrecords-basics#tf.Example
        :return tf_example: <tf.train.Example> The original sample encoded as a tf.train.Example object compatible with
         TFRecord files.
        """
        tf_example: tf.train.Example
        # TFRecord files only support 1D data, so we must first convert the spectrogram 2D np.ndarray to a Tensor:
        sample_tensor: tf.Tensor = tf.convert_to_tensor(sample)
        # Then we must serialize the Tensor:
        serialized_sample_tensor: tf.Tensor = tf.io.serialize_tensor(sample_tensor)
        # And retrieve the Byte String via the numpy method:
        sample_bytes_list: tf.train.BytesList = serialized_sample_tensor.numpy()

        # Convert the ISO 8601 string to a Tensor:
        iso_tensor: tf.Tensor = tf.convert_to_tensor(iso_8601_str)
        # Serialize the Tensor:
        iso_8601_serialized_tensor: tf.Tensor = tf.io.serialize_tensor(iso_tensor)
        # Retrieve the Byte String via the numpy method:
        iso_8601_bytes_list: tf.train.BytesList = iso_8601_serialized_tensor.numpy()

        # Now wrap the BytesLists in Feature objects:
        sample_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[
                sample_bytes_list
            ])
        )
        iso_8601_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[
                iso_8601_bytes_list
            ])
        )
        # Create the Features dictionary:
        features = tf.train.Features(feature={
            'frequencies': sample_feature,
            'iso_8601': iso_8601_feature
        })
        # Finally, wrap the features dictionary with a tensorflow example:
        tf_example = tf.train.Example(features=features)
        return tf_example

    @staticmethod
    def decode_single_tf_example(serialized_example: bytes) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        decode_single_tf_example: Takes a serialized tf.train.Example object (in bytes) and converts it back to Tensors
         containing: a spectrogram 2D array, and an ISO 8601 (datetime string) which corresponds to the audio file that
         produced the spectrogram.
        :param serialized_example: <bytes> The serialized tf.train.Example object containing further individually
         serialized features representing the spectrogram and ISO 8601 datetime strings.
        :returns spectrogram_tensor, iso_8601_tensor:
        :return spectrogram: <tf.Tensor> A tensor containing the 2D array of tf.float32 objects representing the audio
         spectrogram of the source dataset.
        :return iso_8601_tensor: <tf.Tensor> A 1-D tensor containing the tf.string object of the ISO 8601 datetime
         string corresponding to the datetime of the audio file that generated the associated spectrogram.
        """
        # We can now read in the serialized representation with:
        feature_description = {
            'spectrogram': tf.io.FixedLenFeature([], tf.string),  # The 2D source Tensor is a serialized ByteString
            'iso_8601': tf.io.FixedLenFeature([], tf.string)
        }
        read_example = tf.io.parse_single_example(
            serialized=serialized_example,
            features=feature_description
        )
        iso_8601_bytes_list_tensor: tf.Tensor = read_example['iso_8601']
        iso_8601_tensor: tf.Tensor = tf.io.parse_tensor(
            serialized=iso_8601_bytes_list_tensor,
            out_type=tf.string
        )
        # iso_8601: str = iso_8601_tensor.numpy().decode('utf-8')
        spectrogram_bytes_list_tensor: tf.Tensor = read_example['spectrogram']
        spectrogram_tensor: tf.Tensor = tf.io.parse_tensor(
            serialized=spectrogram_bytes_list_tensor, out_type=tf.float32
        )
        return spectrogram_tensor, iso_8601_tensor

    def shard_datasets(self) -> \
            Tuple[List[Tuple[str, List[int]]], List[Tuple[str, List[int]]], List[Tuple[str, List[int]]]]:
        """
        shard_datasets: Breaks the provided metadata pandas DataFrame into separate shards (for each train, val, and
         test dataset subset), constrained by the pre-computed maximum number of samples per TFRecord file.
        :param beemon_df: <pd.DataFrame> The pandas DataFrame containing the metadata for all audio files in the
         dataset. Each record contains the path to an audio file, the ISO 8601 datetime corresponding to the audio file
         name, and additional datetime convenience fields (such as year, day, and week).
        :param max_num_samples_per_tf_record_file: <int> The maximum number of samples (1D np array and ISO 8601 pairs)
         which will fit into a single TFRecord file while adhering to the 200 MB recommended file size limitation.
        :return train_shards, num_train_spectra, val_shards, num_val_spectra, test_shards, num_test_spectra:
        :return train_shards: <List[Tuple[str, List[int]]]> A list of shard filenames, and the associated indices in the
         provided beemon_df that should be used to produce the shard.
        :return num_train_spectra: <int> The number of training spectrograms that have been sharded. This corresponds to
         the summation of the length of every list associated with every shard entry in the train_shards list of tuples.
        :return val_shards: <List[Tuple[str, List[int]]]> A list of shard filenames, and the associated indices in the
         provided beemon_df that should be used to produce the shard.
        :return num_val_spectra: <int> The number of validation samples that have been sharded. This corresponds to the
         summation of the length of every list associated with every shard entry in the val_shards list of tuples.
        :return test_shards: <List[Tuple[str, List[int]]]> A list of shard filenames, and the associated indices in the
         provided beemon_df that should be used to produce the shard.
        :return num_test_spectra: <int> The number of testing samples that have been sharded. This corresponds to the
         summation of the length of every list associated with every shard entry in the test_shards list of tuples.
        """
        train_shards: List[Tuple[str, List[int]]] = []
        num_train_spectra: int
        val_shards: List[Tuple[str, List[int]]] = []
        num_val_spectra: int
        test_shards: List[Tuple[str, List[int]]] = []
        num_test_spectra: int

        # Do the train, test, val, split partition of the metadata dataframe (prior to sharding each dataset):
        train_indices, val_indices, test_indices = self._train_test_val_split(
            beemon_df=self.beemon_df
        )
        # Ensure there are no duplicate indices (sanity check):
        assert len(train_indices) == len(set(train_indices))
        assert len(val_indices) == len(set(val_indices))
        assert len(test_indices) == len(set(test_indices))

        # Hold onto the number of spectrograms for later:
        num_train_spectra = len(train_indices)
        self._num_train_spectra = num_train_spectra
        num_val_spectra = len(val_indices)
        self._num_val_spectra = num_val_spectra
        num_test_spectra = len(test_indices)
        self._num_test_spectra = num_test_spectra
        self._num_spectra = num_train_spectra + num_val_spectra + num_test_spectra

        # Shard the datasets:
        train_meta_df: pd.DataFrame = self.beemon_df.iloc[train_indices]
        train_shards: List[Tuple[str, List[int]]] = self.shard_dataset(
            meta_df=train_meta_df,
            max_num_samples_per_shard=self.max_num_samples_per_tf_record_file,
            dataset_split=DatasetSplitType.TRAIN
        )
        del train_meta_df
        self.num_train_shards = len(train_shards)

        val_meta_df: pd.DataFrame = self.beemon_df.iloc[val_indices]
        val_shards: List[Tuple[str, List[int]]] = self.shard_dataset(
            meta_df=val_meta_df,
            max_num_samples_per_shard=self.max_num_samples_per_tf_record_file,
            dataset_split=DatasetSplitType.VAL
        )
        del val_meta_df
        self.num_val_shards = len(val_shards)

        test_meta_df: pd.DataFrame = self.beemon_df.iloc[test_indices]
        test_shards: List[Tuple[str, List[int]]] = self.shard_dataset(
            meta_df=test_meta_df,
            max_num_samples_per_shard=self.max_num_samples_per_tf_record_file,
            dataset_split=DatasetSplitType.TEST
        )
        del test_meta_df
        self.num_test_shards = len(test_shards)
        return train_shards, val_shards, test_shards

    def shard_dataset(self, meta_df: pd.DataFrame, max_num_samples_per_shard: int, dataset_split: DatasetSplitType) \
            -> List[Tuple[str, List[int]]]:
        """
        shard_dataset: Breaks the provided pandas DataFrame into shards (constrained by the maximum number of samples
         per shard).
        :param meta_df: <pd.DataFrame> The pandas DataFrame containing the metadata for all audio files in the dataset.
         Each record in the dataset will contain the path to an audio file, the ISO 8601 datetime corresponding to the
         audio file name, and additional datetime convenience fields (such as year, day, week).
        :param max_num_samples_per_shard: <int> The maximum number of samples (spectrogram and ISO 8601 pairs) which
         will fit into a single TFRecord file.
        :param dataset_split: <DatasetSplitType> An enumerated type indicating if this is the 'train', 'val' or 'test'
         split/partition of the dataset. This information is used to construct the filename associated with the shard
         which will later be used when outputting the TFRecord files to disk.
        :return shards: <List[Tuple[str, List[str]]]> A list of shard filenames, and the associated indices in the
         provided beemon_df that should be used to produce the shard.
        """
        shards: List[Tuple[str, List[int]]] = []
        global_split_shard_index: int = 0   # Global shard index for the dataset split (train, test, or val) dataset.
        # We want to iterate over each unique day in the dataset:
        # meta_df_grouped_by_day: pd.DataFrameGroupBy = meta_df.groupby(by='day_of_year', as_index=False)
        for day_of_year in meta_df['date'].dt.day_of_year.unique():
            day_df_subset = meta_df[meta_df['date'].dt.day_of_year == day_of_year]
            ''' shard the entire day: '''
            # Determine how many samples reside in the day's subset:
            num_samples_in_subset: int = day_df_subset.shape[0] * self._num_samples_per_spectrogram
            # Determine how many shards will be needed to store the entire day subset:
            num_partial_shards_required_for_day: int = math.ceil(num_samples_in_subset / max_num_samples_per_shard)
            day_requires_multiple_shards: bool
            if num_partial_shards_required_for_day > 1:
                day_requires_multiple_shards = True
                # Preliminary estimates suggest there should never be a day that requires more than two shards to store:
                assert num_partial_shards_required_for_day <= 2, num_partial_shards_required_for_day
            else:
                day_requires_multiple_shards = False

            if not day_requires_multiple_shards:
                # This day will fit into a single shard.
                num_samples_in_shard: int = min(max_num_samples_per_shard, num_samples_in_subset)
                shard_indices: np.ndarray = day_df_subset.index.values
                # Construct a file path for the shard of the form: 'train-001-00-180.tfrec'
                # This corresponds to: dataset_split-dataset_shard_index-partial_shard_index-num_samples_in_shard.tfrec'
                shard_file_path: str = os.path.join(self._output_data_dir, '{}-{:03d}-{:02d}-{:03d}.tfrec'.format(
                    dataset_split.value, global_split_shard_index, 0, num_samples_in_shard))
                # Append the file name, and shard indices (for this particular day) to the list of shards:
                shards.append((shard_file_path, shard_indices))
                # Update the global shard count for this dataset partition (train, val, test):
                global_split_shard_index += 1
            else:
                # This day will not fit into a single shard.
                # Keep track of the size of the previous partial shard:
                num_samples_in_previous_partial_shard: int = 0
                # Iterate over the number of partial shards required to store the day's subset:
                for i in range(num_partial_shards_required_for_day):
                    if i == 0:
                        num_samples_for_partial_shard: int = math.ceil(max_num_samples_per_shard / 2)
                        num_samples_in_previous_partial_shard = num_samples_for_partial_shard
                        # Subset the day's indices by the size of the shard:
                        shard_indices: List[int] = day_df_subset.index.values[0:num_samples_for_partial_shard]
                    else:
                        num_samples_for_partial_shard: int = num_samples_in_previous_partial_shard - math.ceil(max_num_samples_per_shard / 2)
                        # Subset the day's indices by the size of the shard:
                        shard_indices: List[int] = day_df_subset.index.values[num_samples_for_partial_shard - 1:]
                    # Construct a file path for the shard of the form: 'train-001-00-180.tfrec'
                    # This corresponds to: dataset_split-dataset_shard_index-partial_shard_index-num_samples_in_shard.tfrec'
                    shard_file_path: str = os.path.join(self._output_data_dir, '{}-{:03d}-{:02d}-{:03d}.tfrec'.format(
                        dataset_split.value, global_split_shard_index, i, num_samples_for_partial_shard))
                    # Append the file name, and sample indices (for this particular day) to the list of shards:
                    shards.append((shard_file_path, shard_indices))
                    # Update the global shard count for this dataset partition (train, val, test):
                    global_split_shard_index += 1
        return shards

    def _compute_maximum_num_samples_per_tf_record(self, sample: np.ndarray, sample_iso_8601: datetime.datetime) -> int:
        """
        _compute_maximum_num_samples_per_tf_record: Determines how many spectrograms (of the provided dimensions and
         datatype) will fit into a single TFRecord shard to stay within the 100 MB to 200 MB limit that is recommended
         by the TensorFlow documentation (see: https://www.tensorflow.org/tutorials/load_data/tfrecord).
        :param sample: <np.ndarray> A single row of the source spectrogram (presumably produced by
         scipy.signal.spectrogram) whose dimensionality and datatype should be used to calculate the maximum number of
         samples with the same size and data type that will be able to be stored in TFRecord format.
        :return max_num_examples_per_tf_record: <int> The maximum number of spectra (encoded as tf.train.Examples) which
         will fit in a single TFRecord shard to stay withing the recommended file size limit of a single TFRecord shard.
        """
        max_num_samples_per_tf_record: int

        # An individual sample is a single row of the spectrogram of size 4097.
        # Convert the datetime timestamp to ISO 8601 representation:
        sample_iso_8601_str: str = DatetimeUtils.convert_datetime_object_to_iso_8601_zulu_notation(
            date_time=sample_iso_8601
        )
        # Convert the sample into a tf.train.Example:
        example: tf.train.Example = self.convert_sample_to_tf_example(sample=sample, iso_8601_str=sample_iso_8601_str)
        # Serialize the tf.train.Example to get the number of bytes required to store a single sample in the dataset:
        serialized_example: bytes = example.SerializeToString()

        # We could now write out the example as a TFRecord binary protobuffer:
        # output_tf_record_file_path: str = os.path.join(self.output_data_dir, 'example.tfrecord')
        # tf_record_writer = tf.io.TFRecordWriter(output_tf_record_file_path)
        # tf_record_writer.write(serialized_example)
        # tf_record_writer.close()

        # Now we can find out the size in bytes of an individual tf_example object:
        num_bytes_per_megabyte: float = 1E+6
        max_tf_record_shard_size_in_bytes: float = (MAXIMUM_SHARD_SIZE * num_bytes_per_megabyte)
        max_num_examples_per_tf_record = math.ceil(max_tf_record_shard_size_in_bytes / len(serialized_example))
        return max_num_examples_per_tf_record

    def _create_beemon_metadata_df(self) -> pd.DataFrame:
        """
        _create_beemon_metadata_df: Creates a pandas dataframe with all audio file paths, and the file name parsed as a
         valid datetime object; then sorts the dataframe by date in ascending order.
        :return df: <pd.DataFrame> A dataframe comprised of each audio file path, the iso_8601 string representation of
         the specified date and time in the file path name, and the datetime object itself (sorted in ascending order
         by date).
        """
        # ISO Parse and then associate each audio file with a datetime stamp:
        df = pd.DataFrame(data=self._audio_file_paths, columns=['file_path'])
        df["rpi"] = ""
        df["iso_8601"] = ""
        df["date"] = ""

        # Parse the file name into a datetime obj:
        for i, row in df.iterrows():
            base_name = os.path.basename(row['file_path'])
            meta_data = base_name.split('@')
            rpi_id = meta_data[0]
            date_str: str = meta_data[1]
            split_date: List[str] = date_str.split('-')
            year: int = int(split_date[0])
            month: int = int(split_date[1])
            day: int = int(split_date[2])
            time_str = meta_data[2].split('.wav')[0]
            split_time: List[str] = time_str.split('-')
            hour: int = int(split_time[0])
            minute: int = int(split_time[1])
            second: int = int(split_time[2])
            date: datetime.datetime = datetime.datetime(
                year=year, month=month, day=day, hour=hour, minute=minute, second=second
            )
            date_iso_8601: str = date.isoformat()
            df.at[i, 'rpi'] = rpi_id
            df.at[i, 'iso_8601'] = date_iso_8601
        # Sort all audio file paths by datetime in ascending order:
        df["date"] = pd.to_datetime(df.iso_8601)
        df = df.sort_values(by="date")
        # Split the date into multiple columns for ease of access:
        df['year'] = df['date'].dt.year
        df['week'] = df['date'].dt.isocalendar().week
        df['day_of_year'] = df['date'].dt.day_of_year
        df['day_of_week'] = df['date'].dt.dayofweek
        '''
        Add a grouping index to the dataframe for partitioning on the unique ordinal week value AND a unique month 
         value. Note that this is necessary because of a duplicated ordinal week value in some of the ISO calendar years
         (see: https://stackoverflow.com/q/62680813/3429090):
        '''
        df['yr_week_grp_idx'] = df['date'].apply(
            lambda x: '%s-%s' % (x.year, '{:02d}'.format(x.week)))
        return df

    @staticmethod
    def chunk_array(arr: np.ndarray, chunk_size: int) -> np.ndarray:
        """
        chunk_array: Breaks the provided 1D np.ndarray into chunks of the specified size. If the source array is not
         evenly divisible, the last chunk will be less than the specified chunk size.
        :param arr: <np.ndarray> The source array to break into chunks of size 'chunk_size'.
        :param chunk_size: <int> The desired size of the chunks (excluding the last chunk if even division is not
         possible).
        :return chunks: <np.ndarray> The 2D array produced by taking the 1D source array and breaking it into chunks.
        """
        chunks: np.ndarray
        length = arr.shape[0]
        # compute the size of each of the first n-1 chunks:
        # chunk_size: int = int(np.ceil(length / n))
        # get the indices at which the chunk splits will occur:
        chunk_indices = np.arange(chunk_size, length, chunk_size)
        chunks = np.split(arr, chunk_indices)
        return chunks

    @staticmethod
    def _train_test_val_split(beemon_df: pd.DataFrame):
        """
        train_test_val_split: Splits the metadata dataframe (containing all audio file paths and associated dates) into
         into train, val, and test datasets. The dataframe will be partitioned weekly, and from within each week, data
         from a random subset of days will be copied to a train, validation, or testing dataframe. Four random days in
         every week will be allocated to training data, 2 days for validation data, and 1 day for testing data. Each
         respective train/test/val dataframe produced will be sorted (by datetime) in ascending order.
        :param beemon_df: <pd.DataFrame> The metadata dataframe containing all sample data (audio file paths and parsed
         datetime data).
        :returns train_df, val_df, test_df:
        :return train_df: <pd.DataFrame> A subset of the provided beemon_df produced by iterating sequentially over each
         ISO 8601 calendar week in the source beemon_df, randomly permuting the days in each week, and selecting a
         subset of those days.
        :return val_df: <pd.DataFrame> A subset of the provided beemon_df produced by iterating sequentially over each
         ISO 8601 calendar week in the source beemon_df, randomly permuting the days in each week, and selecting a
         subset of those days.
        :return test_df: <pd.DataFrame> A subset of the provided beemon_df produced by iterating sequentially over each
         ISO 8601 calendar week in the source beemon_df, randomly permuting the days in each week, and selecting a
         subset of those days.
        """
        # Empty lists to hold the indices in the parent dataframe which will belong to train, val, and test:
        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        # We consider a week to be 5 days:
        week_duration_in_days: int = 5

        def perform_weekly_train_val_test_split(weekly_day_year_group_indices: np.ndarray):
            # We want to break the chunk into train, val, and test indices:

            weekly_train_day_year_group_indices: List[str]
            weekly_val_day_year_group_indices: List[str]
            weekly_test_day_year_group_indices: List[str]

            # Select random subset of day-of-the-week indices [0 - 4] to be train, val, and test data:
            day_of_week_indices: np.ndarray = np.arange(0, 5)
            day_of_week_indices: np.ndarray = np.random.permutation(day_of_week_indices)
            week_train_indices = day_of_week_indices[0: 3]  # 3 days for the training dataset
            week_val_indices = day_of_week_indices[3: 4]  # 1 day for the validation dataset
            week_test_indices = day_of_week_indices[-1]  # 1 day for the testing dataset
            # Get the year_week_group_index associated with the indices computed above:
            weekly_train_day_year_group_indices = weekly_day_year_group_indices[week_train_indices]
            weekly_val_day_year_group_indices = weekly_day_year_group_indices[week_val_indices]
            weekly_test_day_year_group_indices = weekly_day_year_group_indices[week_test_indices]
            return weekly_train_day_year_group_indices, weekly_val_day_year_group_indices, weekly_test_day_year_group_indices

        # Assign an index that is unique for the ordinal day_of_year and the ordinal year value:
        beemon_df['yr_day_grp_idx'] = beemon_df['date'].apply(
            lambda x: '%s-%s' % (x.year, '{:03d}'.format(x.day_of_year))
        )

        # Group the dataframe by day and year:
        # df_grouped_by_year_and_day = beemon_df.groupby('yr_day_grp_idx', as_index=False)

        # Get the list of unique year and day combinations:
        unique_year_day_groups = beemon_df.yr_day_grp_idx.unique()

        # Break this list into chunks of x days at a time:
        chunked_year_day_group_indices: np.ndarray = ConvertWAVToTFRecord.chunk_array(
            arr=unique_year_day_groups,
            chunk_size=week_duration_in_days
        )

        for week_index, week_day_year_group_indices in enumerate(chunked_year_day_group_indices):
            # For each "week" of the specified size, determine the days which are to belong to the train, val, and test:

            if week_index == len(chunked_year_day_group_indices) - 1:
                # The last chunk may have a size that is smaller than the other chunks due to uneven division:
                if len(week_day_year_group_indices) < week_duration_in_days:
                    # Skip the last chunk if it is not a complete week-sized window.
                    break

            # Get the days in the dataframe which belong to specified "week" chunk:
            weekly_train_year_group_indices, weekly_val_year_group_indices, weekly_test_year_group_indices = \
                perform_weekly_train_val_test_split(
                    weekly_day_year_group_indices=week_day_year_group_indices
                )
            # Get the indices from the dataframe which correspond to the specified day_year_group_indices:
            week_train_meta_data_subset: pd.DataFrame = \
                beemon_df.query('yr_day_grp_idx in @weekly_train_year_group_indices')
            train_indices.extend(week_train_meta_data_subset.index)
            week_val_meta_data_subset: pd.DataFrame = \
                beemon_df.query('yr_day_grp_idx in @weekly_val_year_group_indices')
            val_indices.extend(week_val_meta_data_subset.index)
            week_test_meta_data_subset: pd.DataFrame = \
                beemon_df.query('yr_day_grp_idx in @weekly_test_year_group_indices')
            test_indices.extend(week_test_meta_data_subset.index)

        return train_indices, val_indices, test_indices

    def apply_fourier_transform(self, audio_sample: np.ndarray):
        """
        apply_fourier_transform: Takes in a raw audio file (down or up-sampled via the librosa package) and applies
         consecutive Fourier transforms to produce a spectrogram of the non-stationary signal's frequency content over
         time. see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
        :param audio_sample: <np.ndarray> A 2D numpy array representing the raw audio signal after having been down (or
         up) sampled via the librosa package during loading from disk.
        :returns freqs, time_segs, spectrogram:
        :return freqs: <ndarray> A 1D numpy array of sample frequencies.
        :return time_segs: <ndarray> A 1D numpy array of segment times corresponding to the array of sample frequencies.
        :return spectrogram: <ndarray> A 2D numpy array that is the spectrogram of the provided audio sample comprised
         of the freqs array at the produced time_segs array. By default, the last axis of the spectrogram corresponds to
         the segment times.
        """
        ''' Apply the Fourier transform: '''
        # Determine the closest power of two to the provided audio sample rate:
        closest_power_of_two_to_provided_sample_rate: int = math.ceil(np.log2(self.sample_rate))
        # The nperseg argument of the Fourier transform is constrained to be a power of two, choose the closest
        # to the audio sample rate for increased accuracy:
        num_per_segment: int = 2 ** closest_power_of_two_to_provided_sample_rate
        # Use the default number of points to overlap (Tukey window) as:
        # num_points_to_overlap: int = num_per_segment // 8
        freqs, time_segs, spectrogram = signal.spectrogram(
            audio_sample, nperseg=num_per_segment, window='tukey', fs=self.sample_rate, mode='magnitude'
        )
        return freqs, time_segs, spectrogram

    def _get_shard_output_file_path(self, dataset_split_type: DatasetSplitType, shard_id: int, shard_size: int):
        """
        _get_shard_output_file_path: Constructs a file path to write the output TFRecord shard to. The file path will
         be of the form: 'output_data_dir\\dataset_split-shard_id-num_samples_in_shard.tfrec'. For example, the output
         file path: 'D:\\data\\Bees\\beemon\\processed\\train-000-89.tfrec' indicates that this a TFRecord file for the
         training dataset, this is the 0-th shard (1st shard), which contains 89 samples.
        :param dataset_split_type: <DatasetSplitType> Either a TRAIN, VAL, or TEST enumerated type representing the
         current split of the dataset.
        :param shard_id: <int> The unique identifier for this shard (out of the total number of shards in the dataset
         partition). For instance if there are 100 training data shards, the shard_id will range from 0 - 99 inclusive.
        :param shard_size: <int> The number of audio samples that the current shard contains. This is dependent upon the
         number of data samples in the given dataset split/partition, and the total desired number of shards for the
         specified split/partition.
        :return:
        """
        output_shard_file_path: str
        output_shard_file_path = os.path.join(self.output_data_dir, '{}-{:03d}-{}.tfrec'.format(
            dataset_split_type.value, shard_id, shard_size))
        return output_shard_file_path

    def write_shards_to_output_directory(self, shard_metadata: List[Tuple[str, List[int]]], shuffle_shard_data: bool):
        """
        write_shards_to_output_directory: Takes in the shard metadata, performs the audio preprocessing steps for each
         sample associated with the provided shard, and then outputs
        :param shard_metadata: <List[Tuple[str, List[int]]]> A list of file paths and indices. The file paths correspond
         to the desired output shard file name. The associated indices correspond to the location in the global metadata
         dataframe of the audio sample file paths associated with the shard file path.
        :param shuffle_shard_data: <bool> A boolean flag indicating if the spectra within each shard (with each shard
         representing a single day's worth of audio data) should be shuffled before being sharded and written to disk.
        :return None: Upon completion, this method will have preprocessed the spectra associated with the list of shards
         and sequentially written the sharded data to the output directory specified during initialization.
        """
        # Iterate over the shard file paths, where each shard file path represents either a full day of data (or a half
        # of a day of data for large days that span multiple shards):
        for i, (shard_file_path, shard_indices) in enumerate(shard_metadata):
            if self.is_debug:
                print('Preprocessing data for shard [{:02d}/{:02d}] \'{}\':'.format(i+1, len(shard_metadata), shard_file_path))
            # Create a list to hold the concatenated spectra that will make up a single shard:
            shard_data: List[tf.train.Example] = []
            # If the shuffle_shard_data boolean flag is set to true, we want to randomize the data in each shard. This
            #  corresponds to shuffling the order of the 60 second audio samples (taken every hour) within each day.
            if shuffle_shard_data:
                np.random.shuffle(shard_indices)
            for j, shard_index in enumerate(shard_indices):
                if self.is_debug:
                    print('\tPreprocessing spectra [{:03d}/{:03d}] for shard {:01d}'.format(j+1, len(shard_indices), i))
                spectrogram_sample = self._beemon_df.iloc[shard_index]
                audio_file_path: str = spectrogram_sample['file_path']
                spectrogram_iso_8601: str = spectrogram_sample['iso_8601']
                # Read in the audio at the specified file path while re-sampling it at the specified rate:
                audio, sample_rate = librosa.load(audio_file_path, sr=self.sample_rate)
                # Apply the Fourier transform to get the spectrogram
                freqs, time_segs, spectrogram = self.apply_fourier_transform(audio_sample=audio)
                # By default there will be 66 audio samples per-spectrogram:
                for k in range(spectrogram.shape[1]):
                    sample = spectrogram[:, k]
                    # Calculate the ISO 8601 offset for the single sample:
                    sample_iso_8601: datetime.datetime = self._get_sample_iso_timestamp_offset(
                        spectrogram_iso_8601=spectrogram_iso_8601,
                        sample_index=k,
                        time_segments=time_segs
                    )
                    sample_iso_8601_str: str = DatetimeUtils.convert_datetime_object_to_iso_8601_zulu_notation(
                        date_time=sample_iso_8601
                    )
                    # Convert the sample to a tf.train.Example:
                    tf_example = self.convert_sample_to_tf_example(sample=sample, iso_8601_str=sample_iso_8601_str)
                    # Append the tf.train.Example to the shard's data for the day:
                    shard_data.append(tf_example)
            self._write_tfrecord_file(shard_file_path=shard_file_path, shard_data=shard_data)
        return

    def _write_tfrecord_file(self, shard_file_path: str, shard_data: List[tf.train.Example]):
        """
        _write_tfrecord_file:
        :param shard_data: <Tuple[str, List[str]]> A tuple containing the file path that the TFRecord file should be
         written to, and a list of audio file paths associated with the TFRecord file to be written.
        :return:
        """
        with tf.io.TFRecordWriter(shard_file_path, options='ZLIB') as writer:
            for tf_example in shard_data:
                writer.write(tf_example.SerializeToString())
        return

    def __repr__(self):
        return (
            '{}.{}(root_data_dir={}, output_data_dir={}, audio_duration={}, sample_rate={}, seed={}, test_size={}, '
            'val_size={}, num_samples={}, num_train_samples={}, num_test_samples={}, num_val_samples={})'.format(
                self.__class__.__module__,
                self.__class__.__name__,
                self.root_data_dir,
                self.output_data_dir,
                self.audio_duration,
                self.sample_rate,
                self.seed,
                self.test_size,
                self.val_size,
                self.num_samples,
                self.num_train_samples,
                self.num_test_samples,
                self.num_val_samples
            ))

    def get_audio_file_paths_in_root_data_dir(self, blacklisted_filenames: Set) -> List[str]:
        """
        get_audio_file_paths_in_root_data_dir: Performs a recursive search of all subdirectories in the root data
         directory which was specified during initialization. Any audio files which have been manually blacklisted in
         BeeGAN/Utils/BlacklistedSamples.py are excluded from the global list of audio files.
        :param blacklisted_filenames: <set> A set of audio file names which have been manually identified as anomalous
         and are therefore excluded from the set of all samples.
        :return all_audio_file_paths: <List[str]> A list of the absolute file paths of all audio files in the specified
         root data directory; excluding those audio files which have been manually blacklisted.
        """
        all_audio_file_paths: List[str] = []
        # Get the current working dir:
        cwd = os.getcwd()
        if not os.path.isdir(self.root_data_dir):
            raise FileNotFoundError('The provided root data directory \'%s\' is invalid!' % self.root_data_dir)
        else:
            os.chdir(self.root_data_dir)
        if self.is_debug:
            print('Retrieving a list of all sample audio files in target root_data_dir: \'%s\'' % self.root_data_dir)
        # Assemble list of all audio file paths in the targeted directory:
        for file in Path('.').rglob('*.wav'):
            if file.name not in blacklisted_filenames:
                all_audio_file_paths.append(os.path.abspath(file))
        # Change back to the original working dir:
        os.chdir(cwd)
        return all_audio_file_paths

    @property
    def root_data_dir(self) -> str:
        return self._root_data_dir

    @property
    def output_data_dir(self) -> str:
        return self._output_data_dir

    @property
    def audio_duration(self) -> int:
        return self._audio_duration

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def test_size(self) -> float:
        return self._test_size

    @property
    def val_size(self) -> float:
        return self._val_size

    @property
    def is_debug(self) -> bool:
        return self._is_debug

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def beemon_df(self) -> pd.DataFrame:
        return self._beemon_df

    @property
    def max_num_samples_per_tf_record_file(self) -> int:
        return self._max_num_samples_per_tf_record_file

    @property
    def num_train_samples(self) -> Optional[int]:
        return self._num_train_samples

    @num_train_samples.setter
    def num_train_samples(self, num_train_samples: int):
        self._num_train_samples = num_train_samples

    @property
    def num_val_samples(self) -> Optional[int]:
        return self._num_val_samples

    @num_val_samples.setter
    def num_val_samples(self, num_val_samples: int):
        self._num_val_samples = num_val_samples

    @property
    def num_test_samples(self) -> Optional[int]:
        return self._num_test_samples

    @num_test_samples.setter
    def num_test_samples(self, num_test_samples: int):
        self._num_test_samples = num_test_samples

    @property
    def num_train_shards(self) -> Optional[int]:
        return self._num_train_shards

    @num_train_shards.setter
    def num_train_shards(self, num_train_shards: int):
        self._num_train_shards = num_train_shards

    @property
    def num_val_shards(self) -> Optional[int]:
        return self._num_val_shards

    @num_val_shards.setter
    def num_val_shards(self, num_val_shards: int):
        self._num_val_shards = num_val_shards

    @property
    def num_test_shards(self) -> Optional[int]:
        return self._num_test_shards

    @num_test_shards.setter
    def num_test_shards(self, num_test_shards: int):
        self._num_test_shards = num_test_shards


def main(args):
    is_debug: bool = args.is_verbose
    root_data_dir: str = args.root_data_dir[0]
    output_data_dir: str = args.output_data_dir[0]
    audio_duration: int = args.audio_duration
    sample_rate: int = args.sample_rate
    seed: int = args.seed
    test_size: float = args.test_size
    val_size: float = args.val_size
    # Ensure that the provided arguments are valid:
    if not os.path.isdir(root_data_dir):
        raise FileNotFoundError('The provided root data directory \'%s\' is invalid!' % root_data_dir)
    else:
        os.chdir(root_data_dir)
    if not os.path.isdir(output_data_dir):
        raise FileNotFoundError('The provided data export directory \'%s\' is invalid!' % output_data_dir)
    convert_wav_to_tf_record = ConvertWAVToTFRecord(
        root_data_dir=root_data_dir,
        output_data_dir=output_data_dir,
        audio_duration=audio_duration,
        sample_rate=sample_rate,
        test_size=test_size,
        val_size=val_size,
        is_debug=is_debug,
        seed=seed
    )
    print(convert_wav_to_tf_record)
    # Get the (future) shard filenames and the associated indices in the Beemon DataFrame that will be used to produce
    #  the specified shards, for each dataset split:
    train_spectra_shards, val_spectra_shards, test_spectra_shards = convert_wav_to_tf_record.shard_datasets()
    # Write TFRecord shards to the disk:
    convert_wav_to_tf_record.write_shards_to_output_directory(shard_metadata=train_spectra_shards, shuffle_shard_data=True)
    convert_wav_to_tf_record.write_shards_to_output_directory(shard_metadata=val_spectra_shards, shuffle_shard_data=True)
    convert_wav_to_tf_record.write_shards_to_output_directory(shard_metadata=test_spectra_shards, shuffle_shard_data=True)
    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TensorFlow WAV to TFRecord Converter argument parser.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--root-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    parser.add_argument('--output-data-dir', type=str, nargs=1, action='store', dest='output_data_dir', required=True)
    parser.add_argument('--audio-duration', type=int, default=_DEFAULT_AUDIO_DURATION, dest='audio_duration',
                        help='The duration for the resulting fixed-length audio-data in seconds. Longer files are '
                             'truncated. Shorter files are zero-padded. (default: %(default)s)')
    parser.add_argument('--sample-rate', type=int, default=_DEFAULT_SAMPLE_RATE, required=False, dest='sample_rate',
                        help='The sample-rate of the audio wav-files. (default: %(default)s)')
    parser.add_argument('--seed', action='store', dest='seed', default=_DEFAULT_SEED, required=False,
                        help='The random number generator seed for reproducibility. (default: %(default)s)')
    parser.add_argument('--val-size', type=float, dest='val_size', default=_DEFAULT_VAL_SIZE,
                        help='Fraction of examples in the validation set. (default: %(default)s)')
    parser.add_argument('--test-size', type=float, dest='test_size', default=_DEFAULT_TEST_SIZE)
    command_line_args = parser.parse_args()
    main(args=command_line_args)