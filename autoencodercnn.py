from sklearn.model_selection import train_test_split
from anomalydetector import AnomalyDetector
from extractor import FeatureExtractor

# reduce TF verbosity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.models import Model
from keras.layers import Input, Reshape, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from preprocess import *
from keras.models import load_model
from keras.utils import Sequence
import math
from config import Config

class Batch_Generator(Sequence):

    def __init__(self, filenames, batch_size, input_shape, val_min, val_max, overlap, inf, sup):
        self._min = val_min
        self._max = val_max
        self._overlap = overlap
        self.filenames = filenames
        self.batch_size = batch_size
        self._input_shape = input_shape
        self._size_x = input_shape[0]
        self._size_y = input_shape[1]
        self._inf = inf
        self._sup = sup

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        try:
            batch_x = self.filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
#            out = np.array([crop_sample(normalize(read_file(file_name), self._min, self._max), self._size_x, self._size_y).reshape(self._input_shape) for file_name in batch_x])
#            print(len(batch_x), self._overlap, self._size_x, self._size_y, self._input_shape)
            out = [decompose(
#                crop_sample(
                    normalize(
                        read_file(file_name)[:,self._inf:self._sup]
                        , self._min, self._max),
#                    self._size_x, self._size_y),
                self._input_shape, self._overlap)
#            .reshape(self._input_shape)
            for file_name in batch_x]
#            print(out)
            out = np.concatenate(out)
#            print(out.shape)
            return out, out # parce que la sortie et l'entrée de l'autoencoder doivent être identiques
        except ValueError as e:
            print(e, batch_x)
            raise

#    def _process(self, data):
#        max = 0
#        min = -150
#        data = (data - min) / (max - min)
#        return self._crop_sample(data).reshape(self._input_shape)

#    def _crop_sample(self, data):
#        shape_x = data.shape[0]
#        shape_y = data.shape[1]
#        return data[int((shape_x - self._size_x) / 2) : int((shape_x + self._size_x) / 2),
#                    int((shape_y - self._size_y) / 2) : int((shape_y + self._size_y) / 2)]


class CNN(FeatureExtractor):
    """
        Autoencoder convoluted neural network
    """

    def decompose(self, data, overlap=None):
        if overlap == None:
            overlap = self._overlap
#        print("ae.dec",data.shape)
        return decompose(data, self._shape, overlap)

    def __init__(self, i, s, shape, nb_epochs):
        self._config = Config()
        self._nb_epochs = nb_epochs
        self._batch_size = self._config.get_config_eval('batch_size')
#        self._nb_epochs = self._config.get_config_eval('nb_epochs')
        self._original_shape = (self._config.get_config_eval('waterfall_dimensions')[0], s-i)
        self._shape = shape
        self._overlap = self._config.get_config_eval('window_overlap_training')
        self._min = self._config.get_config_eval('min_value')
        self._max = self._config.get_config_eval('max_value')

#        self._shape = shape
#        self._overlap = overlap
#        self._input_shape = None
        if K.image_data_format() == 'channels_first':
            self._input_shape = (1, self._shape[0], self._shape[1])
        else:
            self._input_shape = (self._shape[0], self._shape[1], 1)
        self._input_tensor = Input(shape = self._input_shape)
        self._size_x = self._shape[0]
        self._size_y = self._shape[1]
#        step_x = round(self._size_x * (1 - self._overlap))
#        x = math.floor((self._original_shape[0] - self._size_x) / step_x) + 1
#        self._delta_timestamp = np.array(range(2*x)) * step_x * waterfall_duration / self._original_shape[0] + self._size_x / 2 * waterfall_duration / self._original_shape[0]
#        self._delta_timestamp = self._delta_timestamp[self._delta_timestamp < waterfall_duration]
#        self._delta_timestamp = self._delta_timestamp.reshape(self._delta_timestamp.shape[0], 1)


    def _new_model(self):
        """
            À utiliser si l'autoencoder n'est pas chargé mais appris
        """

        # Nouveau réseau de neurones
        # L'extraction de features se fait avec Conv2D -> augmentation des dimensions
        # MaxPooling permet de réduire les dimensions
        # Toujours utiliser une activation "relu"
        m = Conv2D(32, (3, 3), activation='relu', padding='same')(self._input_tensor)
        m = MaxPooling2D(pool_size=(2,2))(m)
        m = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self._input_shape)(m)
        m = MaxPooling2D(pool_size=(2,2))(m)
        m = Conv2D(32, (3, 3), activation='relu', padding='same')(m)
        m = MaxPooling2D(pool_size=(2,2))(m)
        m = Conv2D(4, (3, 3), activation='relu', padding='same')(m)
        m = Flatten()(m)
        m = Dense(496, activation='relu')(m)
        m = Dense(300, activation='relu')(m)
        self._coder = Model(self._input_tensor, m)
        self._coder.compile(loss='mean_squared_error',
                                  optimizer='adam')

        # Permet d'éviter l'overfitting
        m = Dropout(0.5)(m)
        m = Dense(496, activation='relu')(m)

        m = Reshape((2,62,4))(m)

        # Maintenant on reconstitue l'image initiale
#        m = UpSampling2D((2,2))(m)
        m = Conv2D(32, (3, 3), activation='relu', padding='same')(m)
        m = UpSampling2D((2,2))(m)
        m = Conv2D(32, (3, 3), activation='relu', padding='same')(m)
        m = UpSampling2D((2,2))(m)
        m = Conv2D(32, (3, 3), activation='relu', padding='same')(m)
        m = UpSampling2D((2,2))(m)

        decoded = Conv2D(1, (3, 3), activation='linear', padding='same')(m)

        # Compilation du modèle + paramètres d'évaluation et d'apprentissage
        self._autoencoder = Model(self._input_tensor, decoded)

        self._autoencoder.compile(loss='mean_squared_error',
                                  optimizer='adam')

        self._autoencoder.summary()

    def learn_extractor(self, filenames, inf, sup):
        self._new_model()
        [training_filenames, validation_filenames] = train_test_split(filenames)
        training_batch_generator = Batch_Generator(training_filenames, self._batch_size, self._input_shape, self._min, self._max, self._overlap, inf, sup)
        validation_batch_generator = Batch_Generator(validation_filenames, self._batch_size, self._input_shape, self._min, self._max, self._overlap, inf, sup)
#        train_X,valid_X,train_ground,valid_ground = train_test_split(data, data, test_size=0.2)
        self._autoencoder.fit_generator(generator=training_batch_generator,
                                        epochs=self._nb_epochs,
                                        verbose=1,
                                        validation_data=validation_batch_generator,
                                        use_multiprocessing=True,
                                        workers=8,
                                        max_queue_size=32)

    def save(self, prefix):
        filename_coder = os.path.join(self._config.get_config("section"), prefix+"coder"+self._config.get_config("autoenc_filename"))
        self._coder.save(filename_coder)
        filename_autoencoder = os.path.join(self._config.get_config("section"), prefix+"autoenc"+self._config.get_config("autoenc_filename"))
        self._autoencoder.save(filename_autoencoder)

    def load(self, prefix):
        print("Loading autoencoder from",prefix+"…"+self._config.get_config("autoenc_filename"))
        filename_coder = os.path.join(self._config.get_config("section"), prefix+"coder"+self._config.get_config("autoenc_filename"))
        self._coder = load_model(filename_coder)
        filename_autoencoder = os.path.join(self._config.get_config("section"), prefix+"autoenc"+self._config.get_config("autoenc_filename"))
        self._autoencoder = load_model(filename_autoencoder)
#        print("Autoencoder loaded!")

    def squared_diff(self, data):
        data = self.decompose(normalize(data, self._min, self._max))
#        return np.sqrt(np.mean(np.subtract(self._autoencoder.predict(data).reshape(-1, self._input_shape[0], self._input_shape[1]), np.squeeze(data))**2, axis=(1,2)))
        return np.subtract(self._autoencoder.predict(data).reshape(-1, self._input_shape[0], self._input_shape[1]), np.squeeze(data))**2


    def learn_threshold(self, filenames, inf, sup):
        print("Threshold estimation…")
        predictions = test_prediction(data, self)
        print("max:",np.max(predictions))
        p = np.percentile(predictions, 1)
        print("1% quantile:",p)
        print("min",np.min(predictions))
        print("mean",np.mean(predictions))
        self._threshold = p

    def reconstruct(self, data):
        data = self.decompose(data)
        return denormalize(self._autoencoder.predict(normalize(data, self._min, self._max)).reshape(-1, self._input_shape[0], self._input_shape[1]), self._min, self._max)

    def extract_features(self, data):
        data = self._crop_samples(data)
        data = self._add_samples(data)
        out = np.array(self._coder.predict(data))
        out = out.reshape(out.shape[0], -1)
        return out

    def _add_samples(self, data):
        return data.reshape(-1, self._input_shape[0], self._input_shape[1], self._input_shape[2])

    def _crop_samples(self, data):
        shape_x = data.shape[1]
        shape_y = data.shape[2]
        return data[:,
                    int((shape_x - self._size_x) / 2) : int((shape_x + self._size_x) / 2),
                    int((shape_y - self._size_y) / 2) : int((shape_y + self._size_y) / 2)]

