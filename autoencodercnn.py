from sklearn.model_selection import train_test_split
from models import AnomalyDetector, FeatureExtractor
import traceback
# reduce TF verbosity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Input, Reshape, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Conv1D, MaxPooling1D
from keras import regularizers
from keras import backend as K
from preprocess import *
from keras.models import load_model
from keras.utils import Sequence
from sklearn.externals import joblib
import math
from config import Config

class Macro_Batch_Generator(Sequence):

    def __init__(self, filenames, batch_size, merge_shape, input_shape, overlap, quant):
        self._overlap = overlap
        self.filenames = filenames
        self.batch_size = batch_size
        self._merge_shape = merge_shape
        self._input_shape = input_shape
        self._quant = quant

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        try:
            batch_x = self.filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
            out = read_files(batch_x, quant=self._quant)
            out = np.vstack(out)
            out = macro_decompose(out, self._merge_shape, self._input_shape, self._overlap)
#            if self._quant:
#                quantify(out)
            # print(out.shape)
            # out = out.reshape(out.shape[0],out.shape[1]*out.shape[2],1) # flatten
            # print(out.shape)
            return out, out # parce que la sortie et l'entrée de l'autoencoder doivent être identiques
        except ValueError as e:
            traceback.print_exc()
            print(e)
            raise



class Batch_Generator(Sequence):

    def __init__(self, filenames, batch_size, input_shape, overlap, inf, sup, quant):
        self._overlap = overlap
        self.filenames = filenames
        self.batch_size = batch_size
        self._input_shape = input_shape
        self._size_x = input_shape[0]
        self._size_y = input_shape[1]
        self._inf = inf
        self._sup = sup
        self._quant = quant

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        try:
            batch_x = self.filenames[idx * self.batch_size : (idx + 1) * self.batch_size]

            out = read_files(batch_x, quant=self._quant)
            out = np.vstack(out)[:,self._inf:self._sup]
            out = decompose(out, self._input_shape, self._overlap)


            # out = [decompose(
# #                crop_sample(
            #             read_file(file_name, quant=self._quant)[:,self._inf:self._sup],
# #                    self._size_x, self._size_y),
            #     self._input_shape, self._overlap)
            # for file_name in batch_x]
            # out = np.concatenate(out)

#            if self._quant:
#                quantify(out)
            # print(out.shape)
            # out = out.reshape(out.shape[0],out.shape[1]*out.shape[2],1) # flatten
            # print(out.shape)
            return out, out # parce que la sortie et l'entrée de l'autoencoder doivent être identiques
        except ValueError as e:
            print(e, batch_x)
            raise

class CNN(FeatureExtractor, AnomalyDetector):
    """
        Autoencoder convoluted neural network
    """


    def anomalies_have_high_score(self):
        return True

    def learn_threshold(self, data):
        super().learn_threshold(data)
        self._all_th.append(self._thresholds)
        self._thresholds = []


    def get_score(self, data, epoch=None):
        """
            Renvoie une valeur
        """
        # print("GET_SCORE",self)
        # print("get",data.shape)
        out = data[:,self._i:self._s]
        # print("spectre",out.shape)
        # print(self._i, self._s)
        out = np.expand_dims(out, axis=0)
        # print("expand",out.shape)
        out = np.array(self.squared_diff(out))
        before = np.mean(out)
        out[out < 0.04] = 0
        print(before, np.mean(out))
        out = np.mean(out)
        out = np.sqrt(out)
        # print("fin get",out.shape)
        return out


    def get_score_vector(self, data, epoch=None):
        """
            Renvoie un tableau
        """
        data = self.decompose(data[:,self._i:self._s], self._overlap_test)
        out = np.array(self.squared_diff(data))
        out[out < 0.1] = 0
        out = np.mean(out, axis=(1,2))
        out = np.sqrt(out)
        return out

    def decompose_test(self, data, overlap=None):
        if overlap == None:
            overlap = self._overlap_test
        return decompose(data, self._shape, overlap)

    def decompose(self, data, overlap=None):
        if overlap == None:
            overlap = self._overlap
        return decompose(data, self._shape, overlap)

    def __init__(self, number, macro=False):
        """
            i: beginning of the spectral band
            s: end of the spectral band
            shape: input shape
            nb_epoches: self-explanatory
        """
        if macro:
            suffix = "macro_"
        else:
            suffix = ""
        self._thresholds = []
        self._all_th = []
        self._config = Config()
        self._number = number
        (self._i, self._s) = self._config.get_config_eval(suffix+'waterfall_frequency_bands')[number]
        self._shape = self._config.get_config_eval(suffix+'autoenc_dimensions')[number]
        self._features_number = self._config.get_config_eval(suffix+'features_number')[number]
        self._nb_epochs = self._config.get_config_eval(suffix+'nb_epochs')[number]
        self._batch_size = self._config.get_config_eval('batch_size')
#        self._nb_epochs = self._config.get_config_eval('nb_epochs')
        self._original_shape = (self._config.get_config_eval('waterfall_dimensions')[0], self._s - self._i)
        self._quant = self._config.get_config_eval('quantification')
        self._overlap = self._config.get_config_eval(suffix+'window_overlap_training')
        self._overlap_test = self._config.get_config_eval(suffix+'extractors_window_overlap_testing')

#        self._shape = shape
#        self._overlap = overlap
#        self._input_shape = None
        # if macro:
        #     if K.image_data_format() == 'channels_first':
        #         self._input_shape = (1, self._shape[0])
        #     else:
        #         self._input_shape = (self._shape[0], 1)
        # else:
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

    def _new_lstm_model(self):
        pass

    def _new_macro_model(self):
        m = Flatten()(self._input_tensor)
        m = Dense(self._features_number, activation='sigmoid')(m)
        self._coder = Model(self._input_tensor, m)
        self._coder.compile(loss='mean_squared_error', # useless parameters
                                  optimizer='adam')
        m = Dense(self._input_shape[0] * self._input_shape[1], activation='sigmoid')(m) # or linear
        decoded = Reshape(self._input_shape)(m)

        self._autoencoder = Model(self._input_tensor, decoded)
        self._autoencoder.compile(loss='mean_squared_error', optimizer='adam') # TODO : nadam
        self._autoencoder.summary()




    def _new_model_2(self):
        m = Flatten()(self._input_tensor)
        # if self._features_number != 5000:
            # m = Dense(5000, activation='sigmoid')(m)
        m = Dense(self._features_number, activation='sigmoid')(m)
        self._coder = Model(self._input_tensor, m)
        self._coder.compile(loss='mean_squared_error', # useless parameters
                                  optimizer='adam')

        # if self._features_number != 5000:
            # m = Dense(5000, activation='sigmoid')(m)
        # m = Dropout(0.3)(m)
        m = Dense(self._input_shape[0] * self._input_shape[1], activation='sigmoid')(m) # or linear
        decoded = Reshape(self._input_shape)(m)

        # Compilation du modèle + paramètres d'évaluation et d'apprentissage
        self._autoencoder = Model(self._input_tensor, decoded)

        # self._autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
        self._autoencoder.compile(loss='mean_squared_error', optimizer='adam') # TODO : nadam

        self._autoencoder.summary()

    def _new_model_3(self):
        m = Reshape((self._input_shape[0],self._input_shape[1]))(self._input_tensor)
        m = Conv1D(500, 5, activation='sigmoid', padding='valid')(m)
        m = Conv1D(500, 5, activation='sigmoid', padding='valid')(m)
        m = Dense(self._features_number, activation='relu')(m)
        self._coder = Model(self._input_tensor, m)
        self._coder.compile(loss='mean_squared_error',
                                  optimizer='adam')
        m = Dense(self._input_shape[0] * self._input_shape[1], activation='sigmoid')(m) # or linear
        decoded = Reshape(self._input_shape)(m)

        self._autoencoder = Model(self._input_tensor, decoded)

        self._autoencoder.compile(loss='mean_squared_error', optimizer='adam')

        self._autoencoder.summary()



    def _new_model(self):
        """
            À utiliser si l'autoencoder n'est pas chargé mais appris
        """

        # Nouveau réseau de neurones
        # L'extraction de features se fait avec Conv2D -> augmentation des dimensions
        # MaxPooling permet de réduire les dimensions
        # Toujours utiliser une activation "relu"

        # TODO: moins de couches cachées, couche dense entre 2 couche convolutive
        # TODO: sigmoid ou tanh pour Conv
        # TODO: faire une couche juste sur spectral
        # TODO: 6 filtres (~ autant que de cases 3*5)
        # TODO: taux d'apprentissage
        m = Reshape((self._input_shape[0],self._input_shape[1]))(self._input_tensor)
        m = Conv1D(500, 5, activation='relu', padding='valid')(m)
        # m = Conv1D(500, 5, activation='sigmoid', padding='valid')(m)
        # m = Conv2D(500, (3, 1000), strides=(1,1000), activation='sigmoid', padding='valid')(m)
        # m = Conv2D(5, (3, 5), strides=(1,2), activation='sigmoid', padding='same')(self._input_tensor)
        # m = Conv2D(5, (3, 5), strides=(1,2), activation='sigmoid', padding='same')(m)
        # m = MaxPooling2D(pool_size=(2,2))(m)
        # m = MaxPooling1D(pool_size=2)(m)
        # m = Conv2D(5, (3, 5), strides=(1,1), activation='sigmoid', padding='same')(m)
        # m = Conv2D(5, (3, 5), strides=(1,1), activation='sigmoid', padding='same')(m)
        m = Flatten()(m)
        # m = Dense(self._features_number, activation='relu')(m)
        # m = Conv2D(10, (3, 5), strides=(1,2), activation='relu', padding='same', input_shape=self._input_shape)(m)
        # m = MaxPooling2D(pool_size=(2,2))(m)
        # m = Dense(self._features_number, activation='relu')(m)
        m = Dense(self._features_number, activation='sigmoid')(m)
        self._coder = Model(self._input_tensor, m)
        self._coder.compile(loss='mean_squared_error',
                                  optimizer='adam')

        # Permet d'éviter l'overfitting
        # m = Dropout(0.5)(m) # TODO vérifier ?

        # m = Dense(self._features_number, activation='relu')(m)
        m = Dense(self._input_shape[0] * self._input_shape[1], activation='sigmoid')(m) # or linear
        decoded = Reshape(self._input_shape)(m)

        # TODO: reconstruire avec dense
        # TODO: rajouter couche une par une
        # TODO: pré-apprentissage couche par couche

        # Compilation du modèle + paramètres d'évaluation et d'apprentissage
        self._autoencoder = Model(self._input_tensor, decoded)

        # self._autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
        self._autoencoder.compile(loss='mean_squared_error', optimizer='adam')

        self._autoencoder.summary()

    def learn(self, data):
        """
            Pas utilisé directement
        """
        assert False


    def get_memory_size(self):
        return 0

    def learn_extractor(self, filenames, inf, sup, macro=False, macro_merge_shape=None, macro_input_shape=None):
        if macro:
            self._new_macro_model()
        else:
            self._new_model()

        filenames = sorted(filenames)
        training_filenames = []
        validation_filenames = []
        # i = 0
        # for f in filenames:
        #     if int(i / self._batch_size) % 4 == 0:
        #         validation_filenames.append(f)
        #     else:
        #         training_filenames.append(f)
        #     i += 1


        [training_nb, validation_nb] = train_test_split(range(0,int(len(filenames)/self._batch_size)))

        for n in training_nb:
            training_filenames += filenames[n * self._batch_size : (n + 1) * self._batch_size]
        for n in validation_nb:
            validation_filenames += filenames[n * self._batch_size : (n + 1) * self._batch_size]

        if macro:
            training_batch_generator = Macro_Batch_Generator(training_filenames, self._batch_size, macro_merge_shape, macro_input_shape, self._overlap, self._quant)
        else:
            training_batch_generator = Batch_Generator(training_filenames, self._batch_size, self._input_shape, self._overlap, inf, sup, self._quant)
        if macro:
            validation_batch_generator = Macro_Batch_Generator(validation_filenames, self._batch_size, macro_merge_shape, macro_input_shape, self._overlap, self._quant)
        else:
            validation_batch_generator = Batch_Generator(validation_filenames, self._batch_size, self._input_shape, self._overlap, inf, sup, self._quant)
#        train_X,valid_X,train_ground,valid_ground = train_test_split(data, data, test_size=0.2)

        # early stopping TODO tester
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

        self._autoencoder.fit_generator(generator=training_batch_generator,
                                        epochs=self._nb_epochs,
                                        verbose=1,
                                        validation_data=validation_batch_generator,
# TODO
                                        # callbacks=[es],
                                        use_multiprocessing=True,
                                        workers=8,
                                        max_queue_size=32)

    def save(self, prefix):
        joblib.dump(self._thresholds, os.path.join(self._config.get_config("section"), prefix+"thr"+self._config.get_config("autoenc_filename")+".thr"))
        filename_coder = os.path.join(self._config.get_config("section"), prefix+"coder"+self._config.get_config("autoenc_filename")+".h5")
        self._coder.save(filename_coder)
        filename_autoencoder = os.path.join(self._config.get_config("section"), prefix+"autoenc"+self._config.get_config("autoenc_filename")+".h5")
        self._autoencoder.save(filename_autoencoder)

    def load(self, prefix):
        print("Loading autoencoder from",prefix+"…"+self._config.get_config("autoenc_filename")+".h5")
        self._thresholds = joblib.load(os.path.join(self._config.get_config("section"), prefix+"thr"+self._config.get_config("autoenc_filename")+".thr"))
        print("Thresholds :",self._thresholds)
        filename_coder = os.path.join(self._config.get_config("section"), prefix+"coder"+self._config.get_config("autoenc_filename")+".h5")
        self._coder = load_model(filename_coder)
        filename_autoencoder = os.path.join(self._config.get_config("section"), prefix+"autoenc"+self._config.get_config("autoenc_filename")+".h5")
        self._autoencoder = load_model(filename_autoencoder)
#        print("Autoencoder loaded!")

    def squared_diff(self, data):
#        data = self.decompose(normalize(data, self._min, self._max), self._overlap_test)
#        print("squared",data.shape)
#        print("squeeze",np.squeeze(data).shape)
        return np.subtract(self._autoencoder.predict(data).reshape(-1, self._input_shape[0], self._input_shape[1]), np.squeeze(data))**2


    # def learn_threshold(self, data, inf, sup):
    #     print("Threshold estimation…")
# #        predictions = test_prediction(data, self)
    #     # print("max:",np.max(predictions))
    #     # p = np.percentile(predictions, 1)
    #     # print("1% quantile:",p)
    #     # print("min",np.min(predictions))
    #     # print("mean",np.mean(predictions))
    #     self._thresholds.add([np.max(data), np.percentile(data,99), np.pencentile(data,95)])
    #     print(self._thresholds)

    def reconstruct(self, data):
        data = self.decompose(data,overlap=0)
        # return self._autoencoder.predict(data.reshape(-1, self._input_shape[0], self._input_shape[1], 1)).reshape(-1, self._input_shape[0], self._input_shape[1])
        return self._autoencoder.predict(data).reshape(-1, self._input_shape[0], self._input_shape[1])

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

