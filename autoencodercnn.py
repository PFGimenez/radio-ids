from sklearn.model_selection import train_test_split
from anomalydetector import AnomalyDetector
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from preprocess import *
from keras.models import load_model
from keras.utils import Sequence

class Batch_Generator(Sequence):

    def __init__(self, filenames, batch_size, input_shape):
        self.filenames = filenames
        self.batch_size = batch_size
        self._input_shape = input_shape

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
        return data[0:48,0:1488].reshape(self._input_shape)

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

        # Nouveau réseau de neurones
        # L'extraction de features se fait avec Conv2D -> augmentation des dimensions
        # MaxPooling permet de réduire les dimensions
        # Toujours utiliser une activation "relu"
        m = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
        m = Conv2D(16, 3, activation='relu', padding='same', input_shape=self._input_shape)(m)
        m = MaxPooling2D(pool_size=(2,2))(m)
        m = Conv2D(8, 3, activation='relu', padding='same')(m)
        m = MaxPooling2D(pool_size=(2,2))(m)
        m = Conv2D(4, 3, activation='relu', padding='same')(m)
        m = MaxPooling2D(pool_size=(2,2))(m)
        m = Conv2D(2, 3, activation='relu', padding='same')(m)

        # la couche de features
        m = MaxPooling2D(pool_size=(2,2))(m)
        self._coder = Model(input_tensor, m)

        # Permet d'éviter l'overfitting
        m = Dropout(0.5)(m)

        # Maintenant on reconstitue l'image initiale
        m = Conv2D(2, 3, activation='relu', padding='same')(m)
        m = UpSampling2D((2,2))(m)
        m = Conv2D(4, 3, activation='relu', padding='same')(m)
        m = UpSampling2D((2,2))(m)
        m = Conv2D(8, 3, activation='relu', padding='same')(m)
        m = UpSampling2D((2,2))(m)
        m = Conv2D(16, 3, activation='relu', padding='same')(m)
        m = UpSampling2D((2,2))(m)

        decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(m)

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
                                        epochs=10,
                                        verbose=1,
                                        validation_data=validation_batch_generator,
                                        use_multiprocessing=True,
                                        workers=8,
                                        max_queue_size=32)
        #self._autoencoder.fit(train_X, train_ground, batch_size=128,epochs=1000,verbose=1,validation_data=(valid_X, valid_ground))

    def save(self, filename):
        self._coder.save(filename)

    def load(self, filename):
        self._coder = load_model(filename)

    def extract_features(self, data_train, data_test):
        return (self._coder.predict(data_train).reshape(data_train.shape[0], -1),
                self._coder.predict(data_test).reshape(data_test.shape[0], -1))

