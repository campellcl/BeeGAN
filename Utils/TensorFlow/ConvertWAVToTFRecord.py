import os
import argparse
from typing import List, Tuple, Union, Dict
import glob
import numpy as np
import math
import tensorflow as tf
import librosa
from scipy import signal
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import datetime
from dateutil import rrule
import pandas as pd
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType


_DEFAULT_AUDIO_DURATION = 60     # seconds
_DEFAULT_SAMPLE_RATE = 8000     # 8 khz
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
        self._audio_file_paths: List[str] = self.get_all_audio_file_paths_in_root_data_dir()
        self._num_samples: int = len(self._audio_file_paths)
        # Create a metadata dataframe by augmenting the list of all audio file paths with time information (from the
        # file names):
        beemon_df: pd.DataFrame = self._create_beemon_metadata_df()
        # Do the train, test, val, split partition of the metadata dataframe (prior to sharding each dataset):
        train_df, val_df, test_df = self._train_test_val_split(
            beemon_df=beemon_df
        )
        # Retain summary statistics:
        self._num_train_samples: int = train_df.shape[0]
        self._num_val_samples: int = val_df.shape[0]
        self._num_test_samples: int = test_df.shape[0]
        '''
        Pre-Process a single sample/audio file to determine the shape of the produced spectrogram given the current 
         runtime settings. This information is required to pre-compute the shard size of the resulting TFRecords:
        '''
        sample = train_df.iloc[0]
        audio_file_path: str = sample['file_path']
        sample_iso_8601: str = sample['iso_8601']
        # Read in the audio at the specified file path while re-sampling it at the specified rate:
        audio, sample_rate = librosa.load(audio_file_path, sr=self.sample_rate)
        freqs, time_segs, spectrogram = self.preprocess_audio_sample(audio_sample=audio)
        # Now we can compute how many samples will fit within an individual TFRecord:
        num_samples_per_tf_record_file: int = self._compute_maximum_num_samples_per_tf_record(
            spectrogram=spectrogram,
            iso_8601=sample_iso_8601
        )
        # Shard the datasets:
        train_shards: Tuple[str, List[int]] = self.shard_dataset(
            meta_df=train_df,
            max_num_samples_per_shard=num_samples_per_tf_record_file,
            dataset_split=DatasetSplitType.TRAIN
        )
        # TODO: Iterate over each day in each dataframe (sequentially) and perform pre-processing on each day's worth
        #  of audio files:
        # _ = self.preprocess_audio_data(metadata_df=train_df)

        # TODO: For each day, concatenate the produced spectrograms together, and determine how many TFRecord files will
        #  be required to store each day's worth of data.
        # TODO: Shuffle the concatenated spectrogram for each day, and then write each day to TFRecord files. Keep the
        #  train, val, and test TFRecord files separate (either by directory or filename).
        # TODO: Write a TFRecord DataSet reader which reads these TFRecord files and reconstructs a TFDataset object
        #  for each split of the dataset (train, val, test)

        # df_week_start_index_inclusive: int = 0
        # df_week_end_index_inclusive: int = -1
        # for dt in rrule.rrule(rrule.WEEKLY, dtstart=df.iloc[0]['date'], until=df.iloc[-1]['date']):
        #     df_start_index_inclusive: int = df.loc[df['iso_8601'] == dt.isoformat()].index[0]


        '''
        Here we shuffle the ordering of the audio sample files so they don't end up encoded sequentially in the same
         TFRecord shards. We do this shuffling here (instead of prior to training) because (to my knowledge) there is 
         no efficient random access to TFRecord objects (see: https://stackoverflow.com/q/35657015/3429090). The 
         sequential hard drive reads during the retrieval of the TFRecord files are preferred for performance reasons.
         For this reason TFRecord files can apparently only be read from disk sequentially (see: 
         https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details)
        '''
        # np.random.shuffle(self._audio_file_paths)
        # self._num_samples: int = len(self._audio_file_paths)
        # self._num_test_samples: int = math.ceil(self._num_samples * test_size)
        # self._num_val_samples: int = math.ceil(self._num_samples * val_size)
        # self._num_train_samples: int = self._num_samples - self._num_test_samples - self._num_val_samples
        # Split the list of audio file paths into train, test, and val sets:
        # self.train_file_paths: List[str] = self._audio_file_paths[0: self._num_train_samples]
        # self.val_file_paths: List[str] = self._audio_file_paths[self._num_train_samples: (self._num_train_samples + self._num_val_samples)]
        # self.test_file_paths: List[str] = self._audio_file_paths[(self.num_train_samples + self._num_val_samples)::]
        # Determine the total number of TFRecord shards that will be necessary for each split of the dataset:
        # self._num_train_shards: int = self._determine_total_number_of_shards(num_samples=self.num_train_samples)
        # self._num_val_shards: int = self._determine_total_number_of_shards(num_samples=self.num_val_samples)
        # self._num_test_shards: int = self._determine_total_number_of_shards(num_samples=self._num_test_samples)

    @staticmethod
    def convert_sample_to_tf_example(spectrogram: np.ndarray, iso_8601: str) -> tf.train.Example:
        """
        convert_sample_to_tf_example: Takes the native data format of a single sample in the dataset and performs the
         conversion of the audio spectrogram and associated ISO 8601 string to a tf.train.Example item which can then
         be further serialized and written (in batch) to a TFRecord file downstream.
        :param spectrogram: <np.ndarray> A 2D numpy array representing the spectrogram of a single audio file sample.
        :param iso_8601: <str> The ISO 8601 datetime string produced by parsing the name of the audio file, which was
         used to generate the spectrogram supplied in conjunction to this method.
        :source: https://www.kaggle.com/ryanholbrook/tfrecords-basics#tf.Example
        :return:
        """
        tf_example: tf.train.Example
        # TFRecord files only support 1D data, so we must first convert the spectrogram 2D np.ndarray to a Tensor:
        spectrogram_tensor: tf.Tensor = tf.convert_to_tensor(spectrogram)
        # Then we must serialize the Tensor:
        spectrogram_serialized_tensor: tf.Tensor = tf.io.serialize_tensor(spectrogram_tensor)
        # And retrieve the Byte String via the numpy method:
        spectrogram_bytes_list: tf.train.BytesList = spectrogram_serialized_tensor.numpy()

        # Convert the ISO 8601 string to a Tensor:
        iso_tensor: tf.Tensor = tf.convert_to_tensor(iso_8601)
        # Serialize the Tensor:
        iso_8601_serialized_tensor: tf.Tensor = tf.io.serialize_tensor(iso_tensor)
        # Retrieve the Byte String via the numpy method:
        iso_8601_bytes_list: tf.train.BytesList = iso_8601_serialized_tensor.numpy()

        # Now wrap the BytesLists in Feature objects:
        spectrogram_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[
                spectrogram_bytes_list
            ])
        )
        iso_8601_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[
                iso_8601_bytes_list
            ])
        )
        # Create the Features dictionary:
        features = tf.train.Features(feature={
            'spectrogram': spectrogram_feature,
            'iso_8601': iso_8601_feature
        })
        # Finally, wrap the features dictionary with a tensorflow example:
        tf_example = tf.train.Example(features=features)
        return tf_example

    def shard_dataset(self, meta_df: pd.DataFrame, max_num_samples_per_shard: int, dataset_split: DatasetSplitType):
        """
        shard_dataset: Breaks the provided pandas DataFrame into shards (constrained by the maximum number of samples
         per shard).
        :param df:
        :param max_num_samples_per_shard:
        :param dataset_split:
        :return shards: List[Tuple[str, List[str]]]
        """
        shards: List[Tuple[str, List[str]]] = []

        def shard_day(day_df: pd.DataFrame, max_num_samples_per_shard: int, dataset_split: DatasetSplitType, shard_index: int):
            day_shards: List[Tuple[str, List[str]]] = []
            # Determine how many shards will be needed to store the entire day subset:
            num_shards_required_for_day: int = math.ceil(day_df.shape[0] / max_num_samples_per_shard)
            shard_day_offset: int = 0
            for i in range(max_num_samples_per_shard):
                day_shard_idx: int = i + shard_day_offset
                sample = day_df.iloc[day_shard_idx]
                audio_file_path = sample['file_path']
                # Read in the audio at the specified file path while re-sampling it at the specified rate:
                audio, sample_rate = librosa.load(audio_file_path, sr=self.sample_rate)
                # Apply the Fourier transform:
                freqs, time_segs, spectrogram = self.preprocess_audio_sample(audio_sample=audio)
                ''' Construct the tf.train.Example key-value dict: '''
                # TFRecords only support 1D data, first convert the 2D array to a Tensor:
                spectrogram_tensor: tf.Tensor = tf.convert_to_tensor(spectrogram)
                # Then serialize the tensor to a binary string:
                spectrogram_byte_string = tf.io.serialize_tensor(spectrogram_tensor)
                # Do the same process with the ISO 8601 date time (convert to a tf.train.BytesList):
                iso_8601_bytes_list: tf.train.BytesList = _bytes_feature(sample['iso_8601'])
                tf_example: tf.train.Example = tf.train.Example(features=tf.train.Features(feature={
                    'audio_byte_str': spectrogram_byte_string,
                    'iso_8601_bytes_list': iso_8601_bytes_list.SerializeToString()
                }))
                # Construct a file path for the shard of the form 'train-001-180.tfrec':
                shard_file_path = os.path.join(self._output_data_dir, '{}-{:03d}-{}.tfrec'.format(
                    dataset_split.value, shard_index, ))

            # # Iterate over the day's data in the metadata dataframe:
            # for i, row in day_df.iterrows():
            #     # Shard the data until we hit the maximum number of samples allocated per shard:
            #     while i <= max_num_samples_per_shard:
            #         # Construct a file path for the shard of the form 'train-001-180.tfrec':
            #         shard_file_path = os.path.join(self._output_data_dir, '{}-{:03d}-{}.tfrec'.format(dataset_split.value, ))

        # Maintain a running counter of the shard index:
        shard_index: int = 0
        # Iterate by year over all existing data in the metadata dataframe:
        for year in meta_df['date'].dt.year.unique():
            year_df_subset = meta_df[meta_df['date'].dt.year == year]

            # Iterate by each week over all existing data in the year:
            for week in year_df_subset['yr_week_grp_idx'].unique():
                week_df_subset = year_df_subset[year_df_subset['yr_week_grp_idx'] == week]

                # Iterate by day over all data within the week:
                for day in week_df_subset['date'].dt.day_of_year.unique():
                    day_df_subset = week_df_subset[week_df_subset['date'].dt.day_of_year == day]
                    # Each day will be sharded:
                    day_shards = shard_day(day_df=day_df_subset, max_num_samples_per_shard=max_num_samples_per_shard, dataset_split=dataset_split, shard_index=shard_index)
                    # Iterate over every audio sample's metadata for the current day:
                    # Accumulate a counter of the size in bytes for the day's worth of audio spectra:
                    # day_spectra_size_in_bytes: int = 0
                    # while day_spectra_size_in_bytes <
                    # audio_file_path = day_df_subset['file_path']
                    # Read in the audio at the specified file path while re-sampling it at the specified rate:
                    # audio, sample_rate = librosa.load(audio_file_path, sr=self.sample_rate)
                    # Apply the Fourier transform:
                    # freqs, time_segs, spectrogram = self.preprocess_audio_sample(audio_sample=audio)


    def _compute_maximum_num_samples_per_tf_record(self, spectrogram: np.ndarray, iso_8601: str):
        """
        _compute_maximum_num_samples_per_tf_record: Determines how many spectrograms (of the provided dimensions and
         datatype) will fit into a single TFRecord shard to stay within the 100 MB to 200 MB limit that is recommended
         by the TensorFlow documentation (see: https://www.tensorflow.org/tutorials/load_data/tfrecord).
        :param spectrogram: <np.ndarray> The input spectrogram (presumably produced by scipy.signal.spectrogram) whose
         dimensionality and datatype should be used to calculate the maximum number of samples with the same size and
         data type that will be able to be stored in TFRecord format.
        :return max_num_samples_per_tf_record: <int> The maximum number of spectra which will fit in a single TFRecord
         shard to stay withing the recommended limit.
        """
        max_num_samples_per_tf_record: int

        # Convert the sample into a tf.train.Example:
        tf_example: tf.train.Example = self.convert_sample_to_tf_example(spectrogram=spectrogram, iso_8601=iso_8601)



        # TFRecord files only support 1D data, so we must first convert the spectrogram 2D np.ndarray to a Tensor:
        spectrogram_tensor: tf.Tensor = tf.convert_to_tensor(spectrogram)
        # Then we must serialize the Tensor:
        spectrogram_serialized_tensor: tf.Tensor = tf.io.serialize_tensor(spectrogram_tensor)
        # And retrieve the Byte String via the numpy method:
        spectrogram_bytes_list: tf.train.BytesList = spectrogram_serialized_tensor.numpy()

        # Convert the ISO 8601 string to a Tensor:
        iso_tensor: tf.Tensor = tf.convert_to_tensor(iso_8601)
        # Serialize the Tensor:
        iso_8601_serialized_tensor: tf.Tensor = tf.io.serialize_tensor(iso_tensor)
        # Retrieve the Byte String via the numpy method:
        iso_8601_bytes_list: tf.train.BytesList = iso_8601_serialized_tensor.numpy()

        # Now wrap the BytesLists in Feature objects:
        spectrogram_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[
                spectrogram_bytes_list
            ])
        )
        iso_8601_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[
                iso_8601_bytes_list
            ])
        )
        # Create the Features dictionary:
        features = tf.train.Features(feature={
            'spectrogram': spectrogram_feature,
            'iso_8601': iso_8601_feature
        })
        # Wrap the features dictionary with a tensorflow Example:
        example = tf.train.Example(features=features)
        # We can now serialize the example:
        serialized_example: bytes = example.SerializeToString()
        # We could now write out the example as a TFRecord binary protobuffer:
        output_tf_record_file_path: str = os.path.join(self.output_data_dir, 'example.tfrecord')
        tf_record_writer = tf.io.TFRecordWriter(output_tf_record_file_path)
        tf_record_writer.write(serialized_example)
        tf_record_writer.close()

        # We can now read in the serialized representation with:
        feature_description = {
            'spectrogram': tf.io.FixedLenFeature([], tf.string),    # The Tensor is a serialized ByteString
            'iso_8601': tf.io.FixedLenFeature([], tf.string)
        }
        read_example = tf.io.parse_single_example(serialized=serialized_example, features=feature_description)
        iso_8601_bytes_list_tensor: tf.Tensor = read_example['iso_8601']
        iso_8601_tensor: tf.Tensor = tf.io.parse_tensor(serialized=iso_8601_bytes_list_tensor, out_type=tf.string)
        iso_8601: str = iso_8601_tensor.numpy().decode('utf-8')
        spectrogram_bytes_list_tensor: tf.Tensor = read_example['spectrogram']
        spectrogram_tensor: tf.Tensor = tf.io.parse_tensor(serialized=spectrogram_bytes_list_tensor, out_type=tf.float32)


        # Now we can find out the size in bytes of an individual tf_example object:
        num_bytes_per_megabyte: float = 1E+6
        max_tf_record_shard_size_in_bytes: float = (MAXIMUM_SHARD_SIZE * num_bytes_per_megabyte)
        max_num_examples_per_tf_record = math.ceil(max_tf_record_shard_size_in_bytes / len(serialized_example))
        print('break')
        # We can read in the serialized representation with:
        example_proto = tf.train.Example.FromString(serialized_example)
        audio_bin_str_bytes_list = example_proto.features.feature['audio_bin_str'].bytes_list
        # Failed attempts at parsing the serialized ByteList protobuffer:
        decoded_audio_tensor = tf.train.Feature.FromString(example_proto.features.feature['audio_bin_str'].bytes_list)
        decoded_audio_bin_bytes_list = tf.io.decode_proto(audio_bin_str_bytes_list, message_type='BytesList', field_names=['audio_bin_str'], output_types=[tf.float32])

        # We can now read back in the source file:

        # Mathematical approach below does not take into account the TFRecord tf.train.Example object serialization:
        # memory_size_in_bytes_of_single_element: float = spectrogram.itemsize
        # num_elements: int = spectrogram.size
        # num_bytes_per_megabyte: float = 1E+6
        # # num_bytes_per_mebibyte = 1024 ** 2
        # max_tf_record_shard_size_in_bytes: float = (MAXIMUM_SHARD_SIZE * num_bytes_per_megabyte)
        # sample_size_in_bytes: float = memory_size_in_bytes_of_single_element * num_elements
        # max_num_samples_per_tf_record = math.ceil(
        #     max_tf_record_shard_size_in_bytes / sample_size_in_bytes
        # )
        return max_num_samples_per_tf_record

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
        df['day_of_year'] = df['date'].dt.day
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
        # Empty placeholder dataframes which will contain the records from the partitioned parent dataframe:
        train_df: pd.DataFrame = pd.DataFrame(
            data=None,
            index=None,
            columns=['file_path', 'rpi', 'iso_8601', 'date', 'year', 'week', 'day_of_year', 'day_of_week']
        )
        val_df: pd.DataFrame = pd.DataFrame(
            data=None,
            index=None,
            columns=['file_path', 'rpi', 'iso_8601', 'date', 'year', 'week', 'day_of_year', 'day_of_week']
        )
        test_df: pd.DataFrame = pd.DataFrame(
            data=None,
            index=None,
            columns=['file_path', 'rpi', 'iso_8601', 'date', 'year', 'week', 'day_of_year', 'day_of_week']
        )

        def perform_weekly_train_val_test_split(
                week_data: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) \
                -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            """
            perform_weekly_train_val_test_split: This helper method takes in existing (and initially empty) train, val,
             and test DataFrames, alongside a week's worth of metadata; and then partitions the week's data into the
             appropriate subset dataframes. Each provided week's worth of data is shuffled randomly (by day) and the
             days are then split among the training, validation, and testing dataframe subsets.
            :param week_data: A unique week in the ISO 8601 calendar year. These are the values produced by iterating
             over the entire source metadata dataframe on week at a time (sequentially) in ascending order.
            :param train_df: <pd.DataFrame> The existing training dataframe (subset of the parent beemon_df dataframe)
             which should be extended with the newly partitioned data from the particular calendar week supplied to this
             method.
            :param val_df: <pd.DataFrame> The existing validation dataframe (subset of the parent beemon_df dataframe)
             which should be extended with the newly partitioned data from the particular calendar week supplied to this
             method.
            :param test_df: <pd.DataFrame> The existing testing dataframe (subset of the parent beemon_df dataframe)
             which should be extended with the newly partitioned data from the particular calendar week supplied to this
             method.
            :returns train_df, val_df, test_df:
            :return train_df: <pd.DataFrame> The provided input dataframe concatenated with the training data from the
             provided week.
            :return val_df: <pd.DataFrame> The provided input dataframe concatenated with the validation data from the
             provided week.
            :return test_df: <pd.DataFrame> The provided input dataframe concatenated with the testing data from the
             provided week.
            """
            # Select a random subset of day-of-the-week indices [0-6] to be training, val, and test data:
            day_of_week_indices = np.arange(0, 7)
            # Shuffle the index array:
            day_of_week_indices = np.random.permutation(day_of_week_indices)
            train_days = day_of_week_indices[0: 4]
            val_days = day_of_week_indices[4: 6]
            test_day = day_of_week_indices[-1]
            week_train_meta_data_series: pd.Series = week_data.query('day_of_week in @train_days')
            week_val_meta_data_series: pd.Series = week_data.query('day_of_week in @val_days')
            week_test_meta_data_series: pd.Series = week_data.query('day_of_week == @test_day')
            # Append each series to their respective train/val/test dataframes:
            train_df = train_df.append(week_train_meta_data_series)
            val_df = val_df.append(week_val_meta_data_series)
            test_df = test_df.append(week_test_meta_data_series)
            # Return the updated provided dataframes:
            return train_df, val_df, test_df

        # Group the dataframe and then iterate over the grouped object:
        df_grouped_by_week_and_year = beemon_df.groupby('yr_week_grp_idx')

        # Iterate through the unique weeks in the dataframe:
        for year_and_week, week_data_subset in df_grouped_by_week_and_year:
            # Split the week's data into training, validation, and testing days; then update the dataframes:
            train_df, val_df, test_df = perform_weekly_train_val_test_split(
                week_data=week_data_subset, train_df=train_df, val_df=val_df, test_df=test_df
            )
        # Attach the yr_week_grp_index for unique week iteration:
        train_df['yr_week_grp_idx'] = train_df['date'].apply(
            lambda x: '%s-%s' % (x.year, '{:02d}'.format(x.week))
        )
        val_df['yr_week_grp_idx'] = val_df['date'].apply(
            lambda x: '%s-%s' % (x.year, '{:02d}'.format(x.week))
        )
        test_df['yr_week_grp_idx'] = test_df['date'].apply(
            lambda x: '%s-%s' % (x.year, '{:02d}'.format(x.week))
        )
        # Return each unique dataframe:
        return train_df, val_df, test_df

    def preprocess_audio_sample(self, audio_sample: np.ndarray):
        # TODO: Docstrings.
        ''' Apply the Fourier transform: '''
        # Determine the closest power of two to the provided audio sample rate:
        closest_power_of_two_to_provided_sample_rate: int = math.ceil(np.log2(self.sample_rate))
        # The nperseg argument of the Fourier transform is constrained to be a power of two, choose the closest
        # to the audio sample rate for increased accuracy:
        num_per_segment: int = 2 ** closest_power_of_two_to_provided_sample_rate
        # Use the default number of points to overlap (Tukey window) as:
        # num_points_to_overlap: int = num_per_segment // 8
        freqs, time_segs, spectrogram = signal.spectrogram(audio_sample, nperseg=num_per_segment)
        return freqs, time_segs, spectrogram

    def preprocess_audio_data(self, metadata_df: pd.DataFrame):
        """
        preprocess_audio_data: Receives a metadata dataframe of either training, testing, or validation data and
         preprocesses the associated audio samples (sequentially) one day at a time.
        :param metadata_df:
        :return:
        """
        # Group the dataframe by the ordinal day of the year:
        df_grouped_by_day = metadata_df.groupby(metadata_df['date'].dt.dayofyear)
        # Iterate over each day of the year, taking all audio files and preprocessing them into spectra:
        for ordinal_day_of_year, day_of_year_subset in df_grouped_by_day:
            # We will concatenate the spectrogram of every sample in the current day and store it here:
            day_of_year_samples_spectra: np.ndarray = np.ndarray(shape=())
            # Iterate over each sample in the current ordinal day of the year:
            for i, row in day_of_year_subset.iterrows():
                audio_file_path: str = row['file_path']
                # Read in the audio at the specified file path while re-sampling it at the specified rate:
                audio, sample_rate = librosa.load(audio_file_path, sr=self.sample_rate)
                ''' Apply the Fourier transform: '''
                freqs, time_segs, spectrogram = self.preprocess_audio_sample(audio_sample=audio)
                # Compute the length of each time segment:
                time_segment_duration: float = time_segs[1] - time_segs[0]
                # TODO: Offset the time mentioned in datetime object by the first segment time?
                # TODO: Somehow encode the time information alongside the

            print('break')



    def _determine_shard_size(self):
        """
        _determine_shard_size: Determines how many WAV files (with the given sample-rate and audio duration) will fit
         into a single TFRecord shard to stay within the 100 MB - 200 MB limit recommended by the TensorFlow
         documentation (see: https://www.tensorflow.org/tutorials/load_data/tfrecord).
        Source: https://gist.github.com/dschwertfeger/3288e8e1a2d189e5565cc43bb04169a1
        :return tf_record_shard_size: <int> The number of samples to contain in a single TFRecord shard.
        """
        num_bytes_per_mebibyte = 1024 ** 2
        maximum_bytes_per_shard = (MAXIMUM_SHARD_SIZE * num_bytes_per_mebibyte)  # 200 MB maximum
        # TODO: Ask Dr. Parry why the sample rate is multiplied by 2 for 16-bit audio, what is it in our case?
        audio_bytes_per_second = self.sample_rate * 2  # 16-bit audio
        audio_bytes_total = audio_bytes_per_second * self.audio_duration
        tf_record_shard_size = maximum_bytes_per_shard // audio_bytes_total
        # TODO: Do we want to apply compression to the TFRecord files, will this compression be lossless enough?
        return tf_record_shard_size
        #
        # num_bytes_per_mebibyte = 1024**2
        # maximum_bytes_per_shard = (MAXIMUM_SHARD_SIZE * num_bytes_per_mebibyte)  # 200 MiB maximum
        # https://www.cs.princeton.edu/courses/archive/spring07/cos116/labs/COS_116_Lab_4_Solns.pdf (page 9/11):
        # https://homes.cs.washington.edu/~thickstn/spectrograms.html
        # We will use scipy.signal.spectrogram, so we will compute the storage requirements in MiB of a single shard...
        # num_samples_in_audio_signal: int = self.sample_rate * self.audio_duration  # The number of float samples in the audio signal (default: 480,000)
        # https://www.princeton.edu/~cuff/ele201/files/spectrogram.pdf
        # https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/positive-definite-matrices-and-applications/complex-matrices-fast-fourier-transform-fft/MIT18_06SCF11_Ses3.2sum.pdf
        # closest_power_of_two_to_provided_sample_rate: int = math.ceil(np.log2(self.sample_rate))
        # TODO: Dr. Parry, are the microphones for the beemon data monophonic?
        # TODO: Dr. Parry, how many bytes per sample per channel?
        # TODO: Ask Dr. Parry why the sample rate is multiplied by 2 for 16-bit audio, what is it in our case?
        # audio_bytes_per_second = self.sample_rate * 2   # 16-bit audio
        # TODO: How many bytes if we decide to encode the spectrogram as a TFRecord and save that?
        ''' Determine how many bytes it will take to encode the spectrogram: '''
        # num_samples_in_spectrogram =
        # audio_bytes_total = audio_bytes_per_second * self.audio_duration
        # tf_record_shard_size = maximum_bytes_per_shard // audio_bytes_total
        # TODO: Do we want to apply compression to the TFRecord files, will this compression be lossless enough?
        # return tf_record_shard_size

    def _determine_total_number_of_shards(self, num_samples: int):
        """
        _determine_total_number_of_shards: Compute the total number of TFRecord shards (given the automatically
         determined ideal shard size and total number of audio samples).
        Source: https://gist.github.com/dschwertfeger/3288e8e1a2d189e5565cc43bb04169a1
        :param num_samples: <int> The number of samples (train, val, or test) to determine the total number of TFRecord
         shards that will be required to store the data.
        :return num_shards: <int> The number of shards
        """
        num_shards: int = math.ceil(num_samples / self._determine_shard_size())
        return num_shards

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

    def split_data_into_shards(self, file_paths: List[str], dataset_split_type: DatasetSplitType) \
            -> List[Tuple[str, List[str]]]:
        """
        _split_data_into_shards: Given a list of file paths for a train, test, or validation split of the dataset and
         the name of the corresponding split (as an enumerated type); this method subsets the provided file paths into
         discrete shards. The result of this method is a list of tuples containing both: the file path for the shard,
         and the list of audio file paths associated with the shards file path.
        :param file_paths: <List[str]> A list of file paths associated with the provided dataset partition/split type.
        :param dataset_split_type: <DatasetSplitType> An enumerated type representing the partition of the dataset:
         train, val, or test.
        :return shards: A list of tuples which contain:
            a) The desired output file path for the shard (once created)
            b) A list of audio sample file paths associated with the shard
        """
        shards: List[Tuple[str, List[str]]] = []
        shard_size: int
        num_shards: int

        if dataset_split_type == DatasetSplitType.TRAIN:
            shard_size = math.ceil(self.num_train_samples / self.num_train_shards)
            num_shards = self.num_train_shards
        elif dataset_split_type == DatasetSplitType.TEST:
            shard_size = math.ceil(self.num_test_samples / self.num_test_shards)
            num_shards = self.num_test_shards
        else:
            shard_size = math.ceil(self.num_val_samples / self.num_val_shards)
            num_shards = self.num_val_shards

        # Split data into shards:
        for shard_id in range(0, num_shards):
            shard_output_file_path = self._get_shard_output_file_path(
                dataset_split_type=dataset_split_type,
                shard_id=shard_id,
                shard_size=shard_size
            )
            shard_starting_index = shard_id * (shard_size - 1)
            shard_ending_index = shard_starting_index + (shard_size - 1)
            shard_file_path_indices = np.arange(shard_starting_index, shard_ending_index)
            file_paths_nd_array = np.array(file_paths)
            # Get a subset of the total train/val/test file paths for the current shard:
            shard_file_paths_nd_array = file_paths_nd_array[shard_file_path_indices]
            # Append the shard file path, and the list of associated samples to the shards array:
            shards.append((shard_output_file_path, list(shard_file_paths_nd_array)))
        return shards

    # def _load_and_sample_audio_file(self, audio_file_path: str) -> np.ndarray:
    #     transformed_audio_file: np.ndarray
    #     y, sr = librose

    def _write_tfrecord_file(self, shard_data: Tuple[str, List[str]]):
        """
        _write_tfrecord_file:
        :param shard_data: <Tuple[str, List[str]]> A tuple containing the file path that the TFRecord file should be
         written to, and a list of audio file paths associated with the TFRecord file to be written.
        :return:
        """
        shard_path, shard_audio_file_paths = shard_data
        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as writer:
            for audio_file_path in shard_audio_file_paths:
                # The spectrogram will be the same dimensionality at
                # Read in the audio at the specified file path while re-sampling it at the specified rate:
                audio, sample_rate = librosa.load(audio_file_path, sr=self.sample_rate)
                ''' Apply the Fourier transform: '''
                # Determine the closest power of two to the provided audio sample rate:
                closest_power_of_two_to_provided_sample_rate: int = math.ceil(np.log2(self.sample_rate))
                # The nperseg argument of the Fourier transform is constrained to be a power of two, choose the closest
                # to the audio sample rate for increased accuracy:
                num_per_segment: int = 2 ** closest_power_of_two_to_provided_sample_rate
                # tf_example = tf.train.Example(features=tf.train.Features(
                #     feature={
                #         'ISO_8601': parsed_time_stamp
                #     }
                # ))

                ''' Parse the filename into a datetime object '''
                # TODO: Parse the filename into a datetime object


                # We use the default overlap by 1/8th:
                # num_points_to_overlap: int = num_per_segment // 8
                # freqs will be the same as long as the sample rate and num per seg is the same.
                # same thing with the time segments. You could have freq, magnitude, and num seconds since january first 1970 as features (0-23 for the hour of the day, and then the minute and second)
                freqs, time_segs, spectrogram = signal.spectrogram(audio, nperseg=num_per_segment)
                # Compute the length of each time segment:
                time_segment_duration: float = time_segs[1] - time_segs[0]
                # TODO: Offset the time mentioned in datetime object by the first segment time:

                # TODODOD:
                # 1. Can we fit an entire day within one TFRecord, play with the parameters a little bit if its close (slightly fewer spectra, use floats instead of doubles)
                # 2. If we can't fit within one TFRecord, then split it in half. Put half of the records in one file with the same name (date) of the audio file (date part 1 and part 2) Save the best guess as the time that this happened, the 't' variable in the output spectrogram.
                #   add that in the time that's in the file name and get a different time date stamp for every spectrum. The day of the year, the hour of the day (could be fractional) in two numbers you could capture seasonal and where is the sun roughly.
                # 3.

                # One sample will NOT correspond to one audio file. Instead we will have multiple spectra per sample
                ''' Create the TFRecord file: '''
                # tf_example = tf.train.Example(features=tf.train.Features(
                #     feature={
                #         'audio_spectrogram': ???
                #     }
                # ))


    def perform_conversion(self):
        # Convert all *.wav files to TFRecords:
        train_shard_splits = self.split_data_into_shards(
            dataset_split_type=DatasetSplitType.TRAIN, file_paths=self.train_file_paths
        )
        for shard in train_shard_splits:
            self._write_tfrecord_file(shard_data=shard)

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

    def get_all_audio_file_paths_in_root_data_dir(self) -> List[str]:
        # Get the current working dir:
        cwd = os.getcwd()
        if not os.path.isdir(self.root_data_dir):
            raise FileNotFoundError('The provided root data directory \'%s\' is invalid!' % self.root_data_dir)
        else:
            os.chdir(self.root_data_dir)
        if self.is_debug:
            print('Retrieving a list of all sample audio files in target root_data_dir: \'%s\'' % self.root_data_dir)
        # Assemble list of all audio file paths in the targeted directory:
        all_audio_file_paths: List[str] = []
        for file in glob.glob('*.wav'):
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
    def num_train_samples(self) -> int:
        return self._num_train_samples

    @property
    def num_test_samples(self) -> int:
        return self._num_test_samples

    @property
    def num_val_samples(self) -> int:
        return self._num_val_samples

    @property
    def num_train_shards(self) -> int:
        return self._num_train_shards

    @property
    def num_test_shards(self) -> int:
        return self._num_test_shards

    @property
    def num_val_shards(self) -> int:
        return self._num_val_shards


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
    convert_wav_to_tf_record.perform_conversion()
    exit(0)


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