import os
import argparse
from typing import List, Tuple, Union
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
        # Create a metadata dataframe with time series information:
        beemon_meta_df: pd.DataFrame = self._create_beemon_metadata_df(
            all_audio_file_paths=self._audio_file_paths
        )
        # Do the train, test, val, split partition:
        meta_train_df, meta_val_df, meta_test_df = self.train_test_val_split(
            beemon_meta_df=beemon_meta_df
        )
        # Create an index






        # df_week_start_index_inclusive: int = 0
        # df_week_end_index_inclusive: int = -1
        # for dt in rrule.rrule(rrule.WEEKLY, dtstart=df.iloc[0]['date'], until=df.iloc[-1]['date']):
        #     df_start_index_inclusive: int = df.loc[df['iso_8601'] == dt.isoformat()].index[0]
        # TODO: Sort by datetime and chunk into 1 week segments.
        # TODO: Randomly select 4 days for train and 1 for test and 2 for val out of each week
        # TODO: For each week, randomize the ordering of the file paths in the respective train test and val splits
        # TODO: Perform pre-processing on each sample and write a separate TFRecord file for each split of the dataset.




        '''
        Here we shuffle the ordering of the audio sample files so they don't end up encoded sequentially in the same
         TFRecord shards. We do this shuffling here (instead of prior to training) because (to my knowledge) there is 
         no efficient random access to TFRecord objects (see: https://stackoverflow.com/q/35657015/3429090). The 
         sequential hard drive reads during the retrieval of the TFRecord files are preferred for performance reasons.
         For this reason TFRecord files can apparently only be read sequentially (see: 
         https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details)
        '''
        np.random.shuffle(self._audio_file_paths)
        self._num_samples: int = len(self._audio_file_paths)
        self._num_test_samples: int = math.ceil(self._num_samples * test_size)
        self._num_val_samples: int = math.ceil(self._num_samples * val_size)
        self._num_train_samples: int = self._num_samples - self._num_test_samples - self._num_val_samples


        # Split the list of audio file paths into train, test, and val sets:
        self.train_file_paths: List[str] = self._audio_file_paths[0: self._num_train_samples]
        self.val_file_paths: List[str] = self._audio_file_paths[self._num_train_samples: (self._num_train_samples + self._num_val_samples)]
        self.test_file_paths: List[str] = self._audio_file_paths[(self.num_train_samples + self._num_val_samples)::]
        # Determine the total number of TFRecord shards that will be necessary for each split of the dataset:
        self._num_train_shards: int = self._determine_total_number_of_shards(num_samples=self.num_train_samples)
        self._num_val_shards: int = self._determine_total_number_of_shards(num_samples=self.num_val_samples)
        self._num_test_shards: int = self._determine_total_number_of_shards(num_samples=self._num_test_samples)

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
        # Split the date into multiple columns for ease of access with groupby:
        df['year'] = df['date'].dt.year
        df['week'] = df['date'].dt.week
        df['day_of_year'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        return df

    def train_test_val_split(self, beemon_meta_df: pd.DataFrame):
        """
        train_test_val_split: Splits the metadata dataframe (containing all audio file paths and associated dates) into
         into train, val, and test datasets. The dataframe will be partitioned weekly, and from within each week, data
         from a random subset of days will be copied to a train, validation, or testing dataframe. Four random days in
         every week will be allocated to training data, 2 days for validation data, and 1 day for testing data. Each
         respective train/test/val dataframe will be sorted by date in ascending order.
        :param beemon_meta_df:
        :return:
        """
        train_meta_df: pd.DataFrame
        val_meta_df: pd.DataFrame
        test_meta_df: pd.DataFrame

        beemon_meta_df['yr_week_grp_idx'] = beemon_meta_df['date'].apply(
            lambda x: '%s-%s' % (x.year, '{:02d}'.format(x.week)))

        # Iterate by year over all the existing data:
        # for year in df['date'].dt.year.unique():
        #     year_df_subset = df[df['date'].dt.year == year]
        #     # Iterate by week over all the existing data in the year:
        #     for week in year_df_subset['date'].dt.week.unique():
        #         week_df_subset = year_df_subset[year_df_subset['date'].dt.week == week]
        #         days_in_week_df_subset = None
        #         pass

        def perform_weekly_train_val_test_split(week_metadata: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
            # Select a random subset of day-of-the-week indices [0-6] to be training, val, and test data:
            day_of_week_indices = np.arange(0, 7)
            # Shuffle the index array:
            day_of_week_indices = np.random.permutation(day_of_week_indices)
            train_days = day_of_week_indices[0: 4]
            val_days = day_of_week_indices[4: 6]
            test_day = day_of_week_indices[-1]
            train_meta_series: pd.Series = week_metadata[week_metadata['day_of_week'] in train_days]

        # Partition each dataframe weekly:
        beemon_meta_df.groupby('yr_week_grp_idx').apply(perform_weekly_train_val_test_split)
        raise NotImplementedError

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