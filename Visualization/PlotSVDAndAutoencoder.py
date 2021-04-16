import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional
from scipy.fft import rfftfreq
from scipy.signal import get_window
import math


def main(args):
    # Command line arguments:
    root_data_dir: str = args.root_data_dir[0]
    output_data_dir: str = args.output_data_dir[0]

    if not os.path.isdir(root_data_dir):
        raise FileNotFoundError('The provided root data directory \'%s\' is invalid!' % root_data_dir)
    if not os.path.isdir(output_data_dir):
        raise FileNotFoundError('The provided output data directory: \'%s\' is invalid!' % output_data_dir)

    # Search for latent encoding file:
    weights_encoder_file_path: Optional[str] = None
    for path in Path(root_data_dir).rglob('WeightsEncoder-*.npy'):
        weights_encoder_file_path = str(path)
        break
    assert weights_encoder_file_path is not None, 'Failed to find file matching pattern \'%s\' in specified input' \
                                                  ' directory: \'%s\'' % ('WeightsEncoder-*.npy', root_data_dir)
    weights_encoder = np.load(weights_encoder_file_path)
    print('Loaded the Encoder\'s trained weights from file: \'%s\'' % weights_encoder_file_path)

    weights_decoder_file_path: Optional[str] = None
    for path in Path(root_data_dir).rglob('WeightsDecoder-*.npy'):
        weights_decoder_file_path = str(path)
        break
    assert weights_decoder_file_path is not None, 'Failed to find file matching pattern \'%s\' in specified input' \
                                                  ' directory: \'%s\'' % ('WeightsDecoder-*.npy', root_data_dir)
    weights_decoder = np.load(weights_decoder_file_path)
    print('Loaded the Decoder\'s trained weights from file: \'%s\'' % weights_decoder_file_path)

    # Search for the serialized singular vector file:
    singular_vector_file_path: Optional[str] = None
    for path in Path(root_data_dir).rglob('SingularVector-*.npy'):
        singular_vector_file_path = str(path)
        break
    assert singular_vector_file_path is not None, 'Failed to find file matching pattern \'%s\' in specified input' \
                                                  ' directory: \'%s\'' % ('SingularVector-*.npy', root_data_dir)
    singular_vector = np.load(singular_vector_file_path)
    print('Loaded closed-form SVD Singular Vector from file: \'%s\'' % singular_vector_file_path)

    num_samples_utilized = int(weights_encoder_file_path.split('-')[-1].split('.npy')[0])
    dataset_split_type: str = weights_encoder_file_path.split('-')[1]

    original_sample_rate_in_hz = 8000
    # Determine the closest power of two to the provided audio sample rate:
    closest_power_of_two_to_provided_sample_rate: int = math.ceil(np.log2(original_sample_rate_in_hz))
    # The nperseg argument of the Fourier transform is constrained to be a power of two, choose the closest
    # to the audio sample rate for increased accuracy:
    num_per_segment: int = 2 ** closest_power_of_two_to_provided_sample_rate
    window_length = num_per_segment
    # window_length = len(get_window('tukey', num_per_segment, fftbins=True))

    # Takes (n=window_length, d=sample spacing (inverse of sampling rate)):
    freq_bin_centers = rfftfreq(window_length, 1/original_sample_rate_in_hz)

    plt.plot(freq_bin_centers, singular_vector, label='Singular Vector')
    plt.title('Frequency Bins vs. Singular Vector (Closed Form SVD) [%s]' % dataset_split_type)
    plt.ylabel('Singular Vector')
    plt.xlabel('Frequency Bins')
    plt.show()

    plt.clf()
    plt.title('Frequency Bins vs. Trained Encoder\'s Weights (1 unit) [%s]' % dataset_split_type)
    plt.ylabel('Encoder Weights')
    plt.xlabel('Frequency Bins')
    plt.plot(freq_bin_centers, weights_encoder, label='Encoder Weights')
    plt.show()

    plt.clf()
    plt.title('Frequency Bins vs. Trained Decoder\'s Weights (1 unit) [%s]' % dataset_split_type)
    plt.ylabel('Decoder Weights')
    plt.xlabel('Frequency Bins')
    plt.plot(freq_bin_centers, weights_decoder, label='Decoder Weights')

    # plt.clf()
    # fig, ax1 = plt.subplots()
    # ax1.set_xlabel('Frequency Bins')
    # ax1.set_ylabel('Encoder Weights')
    # ax1.plot(freq_bin_centers, weights_encoder, label='Encoder Weights')
    # max_weight = np.max(np.max(weights_encoder), np.max(weights_decoder))
    # min_weight = np.min(np.min(weights_encoder), np.min(weights_decoder))
    # ax1.set_ylim(max_weight)
    # # ax1.set_xticks([np.arange(np.min(weights_encoder, weights_decoder), np.max(weights_encoder, weights_decoder), 1.0)])
    # ax1.tick_params(axis='y', labelcolor='blue')
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Decoder Weights')
    # ax2.set_ylim(ax1.get_ylim())
    # ax2.plot(freq_bin_centers, weights_decoder, label='Decoder Weights')
    # plt.legend()
    # plt.show()

    # plt.clf()
    # plt.title('Frequency Bins vs. Trained Autoencoder Weights (1 unit) [%s]' % dataset_split_type)
    # plt.ylabel()

    plt.clf()
    plt.title('Frequency Bins vs. Encoding/SVD (Closed Form) [%s]' % dataset_split_type)
    plt.ylabel('Encoder/Decoder Weights & Singular Vector')
    plt.xlabel('Frequency Bins')
    plt.plot(freq_bin_centers, singular_vector, label='Singular Vector')
    plt.plot(freq_bin_centers, weights_encoder, label='Encoder Weights', alpha=0.2)
    plt.plot(freq_bin_centers, weights_decoder, label='Decoder Weights', alpha=0.2)
    geometric_mean = np.sqrt(np.multiply(weights_encoder, weights_decoder))
    plt.plot(freq_bin_centers, geometric_mean, label='Geometric Mean', alpha=1.0)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PloatSVDAndAutoencoder.py argument parser.')
    parser.add_argument('--root-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    parser.add_argument('--output-data-dir', type=str, nargs=1, action='store', dest='output_data_dir', required=True)
    command_line_args = parser.parse_args()
    main(args=command_line_args)
