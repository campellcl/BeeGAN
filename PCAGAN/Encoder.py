import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, losses, activations
from Utils.EnumeratedTypes.DatasetSplitType import DatasetSplitType
from Utils.TensorFlow.TFRecordLoader import TFRecordLoader

latent_dim = 1


class SVDAutoencoder(tf.keras.Model):

    def __init__(self, latent_dim: int):
        super(SVDAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
          layers.Flatten(),
          layers.Dense(latent_dim, activation=activations.linear),
        ])
        self.decoder = tf.keras.Sequential([
          layers.Dense(4097, activation=activations.linear),
          layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def main(args):
    # Command line arguments:
    is_debug: bool = args.is_verbose
    root_data_dir: str = args.root_data_dir[0]
    batch_size: int = args.batch_size
    dataset_split_str: str = args.dataset_split_str[0]
    order_deterministically: bool = args.order_deterministically

    dataset_split_type: DatasetSplitType

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

    # Obtain the TFRecord dataset corresponding to the requested dataset split ('train', 'val', 'test', 'all'):
    tf_record_loader: TFRecordLoader = TFRecordLoader(
      root_data_dir=root_data_dir,
      dataset_split_type=dataset_split_type,
      is_debug=is_debug,
      order_deterministically=order_deterministically
    )
    tf_record_ds: tf.data.TFRecordDataset = tf_record_loader.get_tf_record_dataset(
      batch_size=batch_size
    )

    autoencoder = SVDAutoencoder(latent_dim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    autoencoder.fit(tf_record_ds, tf_record_ds, epochs=10, shuffle=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder argument parser.')
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--root-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    parser.add_argument('--dataset-split', type=str, nargs=1, action='store', dest='dataset_split_str', required=True,
                        help='The dataset split that should be loaded (e.g. train, test, val, or all).')
    parser.add_argument('--batch-size', type=int, action='store', dest='batch_size', required=True)
    parser.add_argument('--order-deterministically', type=bool, action='store', dest='order_deterministically',
                        required=True)
    command_line_args = parser.parse_args()
    main(args=command_line_args)