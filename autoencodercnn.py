from sklearn.model_selection import train_test_split
from anomalydetector import AnomalyDetector
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from preprocess import *
from keras.models import load_model
from keras.utils import Sequence
import math

class Batch_Generator(Sequence):

    def __init__(self, filenames, batch_size, input_shape):
        self.filenames = filenames
        self.batch_size = batch_size
        self._input_shape = input_shape
        self._size_x = input_shape[0]
        self._size_y = input_shape[1]

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        out = np.array([self._process(read_file(file_name)) for file_name in batch_x])
        return out, out

    def _process(self, data):
        max = 0
        min = -150
        data = (data - min) / (max - min)
        return self._crop_sample(data).reshape(self._input_shape)

    def _crop_sample(self, data):
        shape_x = data.shape[0]
        shape_y = data.shape[1]
        return data[int((shape_x - self._size_x) / 2) : int((shape_x + self._size_x) / 2),
                    int((shape_y - self._size_y) / 2) : int((shape_y + self._size_y) / 2)]


class CNN:
    """
        Autoencoder convoluted neural network
    """

    def decompose(self, data):
        return decompose(data, self._shape, self._overlap)

    def __init__(self, shape, overlap):
        self._shape = shape
        self._overlap = overlap
        self._input_shape = None
        if K.image_data_format() == 'channels_first':
            self._input_shape = (1, shape[0], shape[1])
        else:
            self._input_shape = (shape[0], shape[1], 1)
        input_tensor = Input(shape = self._input_shape)
        self._size_x = shape[0]
        self._size_y = shape[1]

        # Nouveau réseau de neurones
        # L'extraction de features se fait avec Conv2D -> augmentation des dimensions
        # MaxPooling permet de réduire les dimensions
        # Toujours utiliser une activation "relu"
        m = Conv2D(32, (5, 3), activation='relu', padding='same')(input_tensor)
        m = MaxPooling2D(pool_size=(2,2))(m)
        m = Conv2D(16, (5, 3), activation='relu', padding='same', input_shape=self._input_shape)(m)
#        m = MaxPooling2D(pool_size=(2,2))(m)
#        m = Conv2D(8, (5, 3), activation='relu', padding='same')(m)
#        m = MaxPooling2D(pool_size=(2,2))(m)
#        m = Conv2D(4, (5, 3), activation='relu', padding='same')(m)
#        m = MaxPooling2D(pool_size=(2,3))(m)
#        m = Conv2D(1, (5, 3), activation='relu', padding='same')(m)

        # la couche de features
#        m = MaxPooling2D(pool_size=(2,5))(m)
        self._coder = Model(input_tensor, m)

        # Permet d'éviter l'overfitting
#        m = Dropout(0.5)(m)

        # Maintenant on reconstitue l'image initiale
#        m = Conv2D(1, (5, 3), activation='relu', padding='same')(m)
#        m = UpSampling2D((2,5))(m)
#        m = Conv2D(4, (5, 3), activation='relu', padding='same')(m)
#        m = UpSampling2D((2,3))(m)
#        m = Conv2D(8, (5, 3), activation='relu', padding='same')(m)
#        m = UpSampling2D((2,2))(m)
        m = Conv2D(32, (5, 3), activation='relu', padding='same')(m)
        m = UpSampling2D((2,2))(m)

        decoded = Conv2D(1, (5, 3), activation='sigmoid', padding='same')(m)

        # Compilation du modèle + paramètres d'évaluation et d'apprentissage
        self._autoencoder = Model(input_tensor, decoded)

        self._autoencoder.compile(loss='mean_squared_error',
                    optimizer='adadelta')

        self._autoencoder.summary()

    def learn_autoencoder(self, filenames, batch_size):
        [training_filenames, validation_filenames] = train_test_split(filenames)
        training_batch_generator = Batch_Generator(training_filenames, batch_size, self._input_shape)
        validation_batch_generator = Batch_Generator(validation_filenames, batch_size, self._input_shape)
#        train_X,valid_X,train_ground,valid_ground = train_test_split(data, data, test_size=0.2)
        self._autoencoder.fit_generator(generator=training_batch_generator,
                                        epochs=5,
                                        verbose=1,
                                        validation_data=validation_batch_generator,
                                        use_multiprocessing=True,
                                        workers=8,
                                        max_queue_size=32)
        #self._autoencoder.fit(train_X, train_ground, batch_size=128,epochs=1000,verbose=1,validation_data=(valid_X, valid_ground))

    def save(self, filename):
        self._coder.save("coder"+filename)
        self._autoencoder.save("autoenc"+filename)

    def load(self, filename):
        self._coder = load_model("coder"+filename)
        self._autoencoder = load_model("autoenc"+filename)

    def reconstruct(self, data):
        data = self._crop_samples(data)
        data = self._add_samples(data)
        return self._reverse_process(self._autoencoder.predict(self._process(data)).reshape(-1, self._input_shape[0], self._input_shape[1]))

    def extract_features(self, data):
        print(data.shape)
        data = self._crop_samples(data)
        print(data.shape)
        data = self._add_samples(data)
        out = self._coder.predict(data)
        print(out.shape)
        return out.reshape(data.shape[0],-1)

    def _add_samples(self, data):
        print(data.shape)
        return data.reshape(-1, self._input_shape[0], self._input_shape[1], self._input_shape[2])

    def _crop_samples(self, data):
        shape_x = data.shape[1]
        shape_y = data.shape[2]
        return data[:,
                    int((shape_x - self._size_x) / 2) : int((shape_x + self._size_x) / 2),
                    int((shape_y - self._size_y) / 2) : int((shape_y + self._size_y) / 2)]

    def _reverse_process(self, data):
        max = 0
        min = -150
        return data * (max - min) + min

    def _process(self, data):
        max = 0
        min = -150
        return (data - min) / (max - min)


