import os
import argparse
import copy
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Any

IS_DEBUG: bool


def make_sequential_pca_model(receptive_field_size: int, train_batch_size: int, num_units_h1: int,
                              activation_h1: Optional[Any]) -> tf.keras.Sequential:
    """
    make_sequential_pca_model: Creates a tf.keras.Sequential model with a single fully connected/Dense hidden layer,
     whose number of units is equal to the specified number of principal components via command line during program
     execution.
    :param receptive_field_size: <int> The size of the receptive field of the input layer. For PCA in the regression
     case, this is equal to the total number of features in the data.
    :param train_batch_size: <int> The size of a training batch for a single step per training epoch.
    :param num_units_h1: <int> The number of neurons/units in the hidden layer of the neural network. For linear-PCA,
     this corresponds to the number of principal components that should be used.
    :param activation_h1: <tf.nn.*/None> The type of activation function that should be applied to the hidden layer,
     (e.g. the activation function that should be applied to the layer containing the principal components) for linear-
     based PCA this value can be None.
    :return model: <tf.keras.Sequential> The constructed Keras linear PCA model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=receptive_field_size, batch_size=train_batch_size, name='input_layer'),
        tf.keras.layers.Dense(units=num_units_h1, use_bias=True, kernel_initializer='random_uniform',
                              bias_initializer='ones', kernel_regularizer=None, kernel_constraint=None,
                              bias_constraint=None, activation=activation_h1, name='h1'),
        tf.keras.layers.Dense(units=receptive_field_size, batch_size=train_batch_size, name='output_layer')
    ])
    return model


def load_npz_training_data_from_drive(data_file_path: str) -> np.ndarray:
    """
    load_npz_training_data_from_drive: Reads in a single compressed numpy file (.npz) at the provided file path and
     returns the results. Raises a FileNotFoundError if the provided data_file_path is found not to exist.
    :param data_file_path: <str> The file path to the compressed numpy (.npz) file that contains all sample data.
    :return X: <np.ndarray> All sample data stored in the specified .npz file.
    """
    if not os.path.isfile(data_file_path):
        raise FileNotFoundError('The provided root data file path \'%s\' is invalid!' % data_file_path)
    with np.load(data_file_path) as npz_file:
        file_name: str = npz_file.files[0]
        X = npz_file[file_name]
        if IS_DEBUG:
            print('sample data shape: %s' % (X.shape,))
    return X


def perform_neural_network_based_pca(train_data: np.ndarray, num_principal_components: int) -> tf.keras.Model:
    """
    perform_neural_network_based_pca: Performs PCA via neural network on the provided training data. The width (e.g. the
     number of neurons/units) in the sole hidden layer will be determined by the provided num_principal_components. The
     neural network will be trained according to the settings in this method, and the trained tf.keras.Model will be
     returned.
    :param train_data: <np.ndarray> The training data for the neural network.
    :param num_principal_components: <int> The number of principal components to leverage in the sole hidden layer in
     the network.
    :return pca_model: <tf.keras.Model> A fully trained Keras model that has been fit to the provided training data.
    """
    # TODO: Do the train-test-val partition here (after obtaining more data):
    # Note: For the PCA regression case, the training data is the same as the targets (y_train):
    train_data, train_targets = train_data, copy.deepcopy(train_data)
    # Create the model:
    pca_model = make_sequential_pca_model(
        receptive_field_size=train_data.shape[1],
        train_batch_size=train_data.shape[0],
        num_units_h1=num_principal_components,
        activation_h1=None
    )
    if IS_DEBUG:
        pca_model.summary()
    # Compile the model:
    pca_model.compile(
        optimizer='rmsprop',
        loss='MSE',
        metrics=['MSE'],
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None
    )
    if IS_DEBUG:
        print('Fitting the above model to the training data with %s principal components...' % num_principal_components)
    # Fit the model to the training data:
    pca_model.fit(
        x=train_data,
        y=train_targets,
        batch_size=train_data.shape[0],
        epochs=10,
        verbose=2,
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=1,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )
    return pca_model


def perform_sklearn_based_pca(x: np.array, num_components: int) -> Tuple[PCA, np.array, np.array]:
    """
    perform_sklearn_based_pca: Performs PCA via the SKLearn library and returns the singular values resulting from the
     PCA and the explained variance ratio.
    See: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    :param x: <np.array> The data which PCA should be performed on.
    :param num_components: <int> The desired number of principle components to produce in the output.
    :returns pca, singular_values, explained_variance_ratio:
    :return pca: <sklearn.decomposition.PCA> The fitted/trained PCA model with the specified number of components.
    :return singular_values: <np.ndarray> The singular values after having performed PCA with the specified number of
     components on the provided data.
    :return explained_variance_ratio: <np.ndarray> Percentage of variance explained by each of the selected components.
    """
    singular_values: np.array
    explained_variance_ratio: np.array
    pca = PCA(n_components=num_components)
    if IS_DEBUG:
        print('Fitting the sklearn PCA model to the training data with %s principal components...' % num_components)
    pca.fit(X=x, y=None)
    singular_values = pca.singular_values_
    explained_variance_ratio = pca.explained_variance_ratio_
    if IS_DEBUG:
        print('singular_values: %s' % singular_values)
        print('explained_variance_ratio: %s' % explained_variance_ratio)
    return pca, singular_values, explained_variance_ratio


def main(args):
    # Parse command line arguments:
    data_file: str = args.data_file
    is_debug: bool = args.is_verbose
    num_principal_components: int = args.num_components

    # Init global vars:
    global IS_DEBUG
    IS_DEBUG = is_debug

    # Load all sample data from hard drive in compressed (.npz) format:
    X = load_npz_training_data_from_drive(data_file_path=data_file)

    # Perform PCA via Multi-Layered Perceptron (MLP):
    fitted_pca_model = perform_neural_network_based_pca(
        train_data=X,
        num_principal_components=num_principal_components
    )

    # Perform PCA via SKLearn with the same settings:
    fitted_sklearn_pca_model, singular_values, explained_variance_ratio = perform_sklearn_based_pca(
        x=X,
        num_components=num_principal_components
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCGAN argument parser.')
    parser.add_argument(
        '--data-file', action='store', type=str, dest='data_file', required=True,
        help='The location of the transformed (down-sampled or up-sampled) audio data stored as a .npz file.'
    )
    parser.add_argument('-v', '--verbose', action='store_true', dest='is_verbose', required=False)
    parser.add_argument('--num-components', type=int, action='store', dest='num_components', required=True,
                        help='The number of principle components corresponding to the number of neurons in the first '
                             'hidden layer (the width of the first hidden layer)')
    # parser.add_argument('--beemon-data-dir', type=str, nargs=1, action='store', dest='root_data_dir', required=True)
    args = parser.parse_args()
    main(args=args)
