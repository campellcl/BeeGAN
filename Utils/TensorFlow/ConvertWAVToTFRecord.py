import os
import argparse
from typing import List, Tuple
import glob
import numpy as np
import math
import tensorflow as tf
import librosa
from scipy.signal import spectrogram
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType


_DEFAULT_AUDIO_DURATION = 60     # seconds
_DEFAULT_SAMPLE_RATE = 8000     # 8 khz
_DEFAULT_SEED = 42
_DEFAULT_TEST_SIZE = .20        # 20%
_DEFAULT_VAL_SIZE = .20         # 20%
MAXIMUM_SHARD_SIZE = 200        # in bytes (200 MB)


def _tf_float_feature(floats: List[float]):
    # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=floats))


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

    def _determine_shard_size(self):
        """
        _determine_shard_size: Determines how many WAV files (with the given sample-rate and audio duration) will fit
         into a single TFRecord shard to stay within the 100 MB - 200 MB limit recommended by the TensorFlow
         documentation (see: https://www.tensorflow.org/tutorials/load_data/tfrecord).
        Source: https://gist.github.com/dschwertfeger/3288e8e1a2d189e5565cc43bb04169a1
        :return tf_record_shard_size: <int> The number of samples to contain in a single TFRecord shard.
        """
        num_bytes_per_mebibyte = 1024**2
        maximum_bytes_per_shard = (MAXIMUM_SHARD_SIZE * num_bytes_per_mebibyte)  # 200 MB maximum
        # TODO: Ask Dr. Parry why the sample rate is multiplied by 2 for 16-bit audio, what is it in our case?
        audio_bytes_per_second = self.sample_rate * 2   # 16-bit audio
        audio_bytes_total = audio_bytes_per_second * self.audio_duration
        tf_record_shard_size = maximum_bytes_per_shard // audio_bytes_total
        # TODO: Do we want to apply compression to the TFRecord files, will this compression be lossless enough?
        return tf_record_shard_size

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
        with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
            for audio_file_path in shard_audio_file_paths:
                audio, sample_rate = librosa.load(audio_file_path, sr=self.sample_rate)
                ''' Constrain the audio to be one second in length: '''
                # The desired number of floats in the audio file is dictated by the (sample_rate * audio_duration):
                desired_signal_length = self.sample_rate * self.audio_duration
                if len(audio) > desired_signal_length:
                    audio = np.array(audio[0:desired_signal_length])
                elif len(audio) < desired_signal_length:
                    # TODO: Ask Dr. Parry if we should zero-pad the audio file?
                    num_zeros_to_pad_with = len(audio) - desired_signal_length
                    audio = np.pad(audio, (num_zeros_to_pad_with, 0))
                assert len(audio) == desired_signal_length
                ''' Apply the Fourier transform: '''
                # Determine the closest power of two to the provided sample rate:
                closest_power_of_two_to_provided_sample_rate: int = math.ceil(np.log2(self.sample_rate))
                # The nperseg argument of the Fourier transform is constrained to be a power of two, choose the closest
                # to the sample rate for increased accuracy:
                num_per_segment: int = 2**closest_power_of_two_to_provided_sample_rate
                # Choose to overlap the audio segments by 50% (hence the division by two):
                num_points_to_overlap: int = num_per_segment // 2
                freqs, time_segs, spectra = spectrogram(audio, nperseg=num_per_segment, noverlap=num_points_to_overlap)
                ''' Create the TFRecord file: '''
                tf_example = tf.train.Example(features=tf.train.Features(
                    feature={
                        # TODO: Ask Dr. Parry if we should serialize the entire spectra? Or just the audio file?
                        #  Probably depends on the loading procedures for TFRecord files.
                        'audio_spectrogram': ???
                    }
                ))
                print('woa')


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