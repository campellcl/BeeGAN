import argparse
import os
from typing import List
import glob
import librosa
import numpy as np


class AudioPreprocessor:

    def __init__(self, root_data_dir: str, output_data_dir: str, sample_rate: int, is_debug: bool):
        self.root_data_dir: str = root_data_dir
        self.output_data_dir: str = output_data_dir
        self.sample_rate: int = sample_rate
        self.is_debug: bool = is_debug

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

    def sample_audio_files_and_store_in_memory(self, all_audio_file_paths: List[str]) -> np.ndarray:
        transformed_audio: np.ndarray
        # Get the size of the audio data:
        num_audio_files = len(all_audio_file_paths)
        audio_file_sample_size: int
        y, sr = librosa.load(all_audio_file_paths[0], sr=self.sample_rate)
        audio_file_sample_size = y.shape[0]
        # Create array to store the results of the down (or up) sampling operations:
        transformed_audio: np.ndarray = np.ndarray(shape=(num_audio_files, audio_file_sample_size))
        # Load and down (or up) sample all audio files:
        for i, file_path in enumerate(all_audio_file_paths):
            if self.is_debug:
                print('Transforming file [%d/%d]' % (i, (num_audio_files - 1)))
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            transformed_audio[i] = y
        return transformed_audio

    def save_transformed_audio_to_compressed_numpy_array(self, transformed_audio: np.ndarray) -> None:
        # Pickle the transformed audio files and write them to the output directory:
        cwd = os.getcwd()
        output_file_path = os.path.join(self.output_data_dir, 'transformed_data.npz')
        if self.is_debug:
            print('Saving transformed (down-sampled or up-sampled) audio files to location: \'%s\'')
        os.chdir(self.output_data_dir)
        np.savez(output_file_path, transformed_audio)
        os.chdir(cwd)
        return


def main(args):
    root_data_dir: str = args.root_data_dir[0]
    output_data_dir: str = args.output_data_dir[0]
    sample_rate: int = args.sample_rate
    is_debug: bool = args.is_verbose
    # Handle argument parsing:
    if not os.path.isdir(root_data_dir):
        raise FileNotFoundError('The provided root data directory \'%s\' is invalid!' % root_data_dir)
    else:
        os.chdir(root_data_dir)
    if not os.path.isdir(output_data_dir):
        raise FileNotFoundError('The provided data export directory \'%s\' is invalid!' % output_data_dir)
    # Instantiate helper class:
    audio_preprocessor = AudioPreprocessor(
        root_data_dir=root_data_dir,
        output_data_dir=output_data_dir,
        sample_rate=sample_rate,
        is_debug=is_debug
    )
    all_audio_file_paths_in_root_dir = audio_preprocessor.get_all_audio_file_paths_in_root_data_dir()
    transformed_audio = audio_preprocessor.sample_audio_files_and_store_in_memory(
        all_audio_file_paths=all_audio_file_paths_in_root_dir
    )
    audio_preprocessor.save_transformed_audio_to_compressed_numpy_array(
        transformed_audio=transformed_audio
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Preprocessor argument parser.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--root-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    parser.add_argument('--output-data-dir', type=str, nargs=1, action='store', dest='output_data_dir', required=True)
    parser.add_argument('--sample-rate', type=int, action='store', dest='sample_rate', required=True)
    args = parser.parse_args()
    main(args=args)
    # TODO: write a TensorFlow data loader that will read from the directory and perform normalization, etc. on the data.