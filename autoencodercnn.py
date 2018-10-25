from sklearn.model_selection import train_test_split
from anomalydetector import AnomalyDetector
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K

class CNN(AnomalyDetector):
    """
        Autoencoder convoluted neural network
    """

    def __init__(self, shape):
        shape=(256,256)
        if K.image_data_format() == 'channels_first':
            input_shape = (1, shape[0], shape[1])
        else:
            input_shape = (shape[0], shape[1], 1)
        input_tensor = Input(shape = input_shape)

        m = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
        # Nouveau réseau de neurones
        print("shape ",str(input_shape))
        # L'extraction de features se fait avec Conv2D -> augmentation des dimensions
        # MaxPooling permet de réduire les dimensions
        # Toujours utiliser une activation "relu"
        m = Conv2D(16, 3, activation='relu', padding='same', input_shape=input_shape)(m)
        m = MaxPooling2D(pool_size=(2,2))(m)
        m = Conv2D(8, 3, activation='relu', padding='same')(m)
        m = MaxPooling2D(pool_size=(2,2))(m)
        m = Conv2D(8, 3, activation='relu', padding='same')(m)

        # la couche de features
        m = MaxPooling2D(pool_size=(2,2))(m)
        self._code = m

        # Permet d'éviter l'overfitting
        m = Dropout(0.5)(m)

        # Maintenant on reconstitue l'image initiale
        m = Conv2D(8, 3, activation='relu', padding='same')(m)
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

    def learn(self, data, exo=None):
        # TODO : construire les images

        train_X,valid_X,train_ground,valid_ground = train_test_split(data, data, test_size=0.2)
        self._autoencoder = self._autoencoder.fit(train_X, train_ground, batch_size=128,epochs=50,verbose=1,validation_data=(valid_X, valid_ground))


    def predict(self, data, obs):
        pass
