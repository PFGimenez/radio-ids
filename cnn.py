import tensorflow as tf
from anomalydetector import AnomalyDetector
from tf.keras.models import Sequential
from tf.keras.layers import Dense, Dropout, Activation, Flatten
from tf.keras.layers import Conv2D, MaxPooling2D

class CNN(AnomalyDetector):
    """
        Convoluted neural network
    """

    def __init__(self):
        self._model = tf.keras.models.Sequential([
# Nouveau réseau de neurones
        model = Sequential()

# L'extraction de features se fait avec Conv2D -> augmentation des dimensions
# MaxPooling permet de réduire les dimensions
# Toujours utiliser une activation "relu"
        model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(1,28,28))) # TODO : dimension
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(8, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
# et potentiellement plus de couches…

# Flatten : applatit en une seule dimension
        model.add(Flatten())

# Couche dense
        model.add(Dense(32, activation='relu'))

# Permet d'éviter l'overfitting
        model.add(Dropout(0.5))

# Pas d'activation car c'est l'output
        model.add(Dense(10, activation='softmax'))

        # Compilation du modèle + paramètres d'évaluation et d'apprentissage
#model.compile(loss='mean_squared_error', optimizer='adam')
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])


    def learn(self, data, exo=None):


    def predict(self, data, obs):
