import os
import pickle
import cv2
import numpy as np
import pandas as pd
import scikitplot
import seaborn as sns
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, models
from keras.utils import np_utils
from tensorflow.python.compiler.mlcompute import mlcompute
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.utils.vis_utils import plot_model

from config import traind_datagen

mlcompute.set_mlc_device(device_name='gpu')
__CURRENT_PATH = os.path.dirname(__file__)
ARCHITECTURE_PATH = os.path.join(__CURRENT_PATH, "models/architectures/")
CONFUSION_MATRIX_PATH = os.path.join(__CURRENT_PATH, "models/confusion_matrix/")
EPOCH_HISTORY_PATH = os.path.join(__CURRENT_PATH, "models/epoch_history/")
MODELS_PATH = os.path.join(__CURRENT_PATH, "models/models/")
PERFORMANCE_DIST_PATH = os.path.join(__CURRENT_PATH, "models/performance_dist/")
VALUES_PATH = os.path.join(__CURRENT_PATH, "models/values/")
DATA_PATH = os.path.join(__CURRENT_PATH, "fer2013/fer2013/fer2013.csv")


def create_datagen(pattern):
    if pattern is not None:
        train_datagen_labels = traind_datagen.train_datagen_labels
        return ImageDataGenerator(
            rotation_range=pattern[train_datagen_labels[0]],
            width_shift_range=pattern[train_datagen_labels[1]],
            height_shift_range=pattern[train_datagen_labels[2]],
            shear_range=pattern[train_datagen_labels[3]],
            zoom_range=pattern[train_datagen_labels[4]],
            horizontal_flip=pattern[train_datagen_labels[5]]
        )
    return pattern


def create_early_stopping(pattern):
    if pattern is not None:
        return EarlyStopping(
            monitor=pattern["monitor"],
            min_delta=pattern["min_delta"],
            patience=pattern["patience"],
            verbose=pattern["verbose"],
            restore_best_weights=pattern["restore_best_weights"]
        )
    return pattern


def create_scheduler(pattern):
    if pattern is not None:
        return ReduceLROnPlateau(
            monitor=pattern["monitor"],
            min_delta=pattern["min_delta"],
            factor=pattern["factor"],
            patience=pattern["patience"],
            min_lr=pattern["min_lr"],
            verbose=pattern["verbose"]
        )
    return pattern


def cnn_for_raw_img(in_shape):
    model_in = Input(shape=in_shape, name="input_CNN")

    conv2d_1 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_1'
    )(model_in)
    batchnorm_1 = BatchNormalization(name='batchnorm_1')(conv2d_1)
    conv2d_2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_2'
    )(batchnorm_1)
    batchnorm_2 = BatchNormalization(name='batchnorm_2')(conv2d_2)

    maxpool2d_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1')(batchnorm_2)
    dropout_1 = Dropout(0.35, name='dropout_1')(maxpool2d_1)

    conv2d_3 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_3'
    )(dropout_1)
    batchnorm_3 = BatchNormalization(name='batchnorm_3')(conv2d_3)
    conv2d_4 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_4'
    )(batchnorm_3)
    batchnorm_4 = BatchNormalization(name='batchnorm_4')(conv2d_4)

    maxpool2d_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2')(batchnorm_4)
    dropout_2 = Dropout(0.4, name='dropout_2')(maxpool2d_2)

    conv2d_5 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_5'
    )(dropout_2)
    batchnorm_5 = BatchNormalization(name='batchnorm_5')(conv2d_5)
    conv2d_6 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_6'
    )(batchnorm_5)
    batchnorm_6 = BatchNormalization(name='batchnorm_6')(conv2d_6)

    maxpool2d_3 = MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3')(batchnorm_6)
    dropout_3 = Dropout(0.5, name='dropout_3')(maxpool2d_3)

    flatten = Flatten(name='flatten')(dropout_3)

    dense_1 = Dense(
        256,
        activation='elu',
        kernel_initializer='he_normal',
        name='dense1'
    )(flatten)
    batchnorm_7 = BatchNormalization(name='batchnorm_7')(dense_1)

    model_out = Dropout(0.6, name='dropout_4')(batchnorm_7)
    return model_in, model_out


class Emotion_model:

    def __init__(self, model_name, selected_emotions, bach_size, epochs, lr_scheduler, early_stopping, train_datagen):
        self.__batch_size = bach_size
        self.__emotions_names = {0: 'anger',
                                 1: 'disgust',
                                 2: 'fear',
                                 3: 'happiness',
                                 4: 'sadness',
                                 5: 'surprise',
                                 6: 'neutral'
                                 }
        self.__select_emotions = selected_emotions
        self.__map_labels = None
        self.__model = None
        self.__optim = optimizers.Adam(0.001)
        self.__model_name = model_name
        self.__history = None
        self.__epochs = epochs
        self.__history_epoch = []
        self.__train_datagen_pattern = train_datagen
        self.__lr_scheduler_pattern = lr_scheduler
        self.__early_stopping_pattern = early_stopping
        self.__trained = False
        self.__X_test = []
        self.__y_test = []

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def lr_scheduler_pattern(self):
        return self.__lr_scheduler_pattern

    @property
    def early_stopping_pattern(self):
        return self.__early_stopping_pattern

    @property
    def train_datagen_pattern(self):
        return self.__train_datagen_pattern

    @property
    def train_datagen(self):
        return create_datagen(self.__train_datagen_pattern)

    @property
    def history_epoch(self):
        return self.__history_epoch

    @property
    def callbacks(self):
        return [create_scheduler(self.__lr_scheduler_pattern), create_early_stopping(self.__early_stopping_pattern)]

    @property
    def model_emotions(self):
        return [self.__emotions_names[i] for i in self.__map_labels.keys()]

    @classmethod
    def create_model(cls, model_name, selected_emotions, bach_size=20, epochs=65,
                     lr_scheduler=None, early_stopping=None, train_datagen=None):
        return cls(model_name, selected_emotions, bach_size, epochs, lr_scheduler, early_stopping, train_datagen)

    @classmethod
    def load_model(cls, model_name):
        with open(VALUES_PATH + model_name + '.pickle', 'rb') as handle:
            values = pickle.load(handle)
        instance = cls(model_name, values["select_emotions"], values["batch_size"], values["epochs"],
                       values["lr_scheduler_pattern"], values["early_stopping_pattern"],
                       values["train_datagen_pattern"])
        instance.__model = models.load_model(MODELS_PATH + model_name + ".h5")
        instance.__map_labels = values["map_labels"]
        instance.__history = values["history"]
        instance.__history_epoch = values["history_epochs"]
        instance.__trained = values["trained"]
        instance.__X_test = values["X_test"]
        instance.__y_test = values["Y_test"]
        return instance

    def save_values(self):
        to_save = {
            "batch_size": self.__batch_size,
            "select_emotions": self.__select_emotions,
            "map_labels": self.__map_labels,
            "model_name": self.__model_name,
            "history": self.__history,
            "epochs": self.__epochs,
            "history_epochs": self.__history_epoch,
            "train_datagen_pattern": self.__train_datagen_pattern,
            "lr_scheduler_pattern": self.__lr_scheduler_pattern,
            "early_stopping_pattern": self.__early_stopping_pattern,
            "trained": self.__trained,
            "X_test": self.__X_test,
            "Y_test": self.__y_test,
        }
        with open(VALUES_PATH + self.__model_name + '.pickle', 'wb') as handle:
            pickle.dump(to_save, handle)

    def load_data(self):
        df = pd.read_csv(DATA_PATH)
        return df

    def fileter_data_by_emotions(self, df):
        return df[df.emotion.isin(self.__select_emotions)]

    def prepare_data(self, df):
        img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
        img_array = np.stack(img_array, axis=0)

        le = LabelEncoder()
        img_labels = le.fit_transform(df.emotion)
        img_labels = np_utils.to_categorical(img_labels)

        self.__map_labels = dict(zip(le.classes_, le.transform(le.classes_)))

        return img_array, img_labels

    @staticmethod
    def split_data_for_train_valid_test(array, labels):
        X_train, X_valid, y_train, y_valid = train_test_split(
            array,
            labels,
            shuffle=True,
            stratify=labels,
            test_size=0.1,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_train,
            y_train,
            shuffle=True,
            stratify=y_train,
            test_size=0.1,
            random_state=42
        )

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    @staticmethod
    def normalize_data(X_train, X_valid, X_test):
        return X_train / 255., X_valid / 255., X_test / 255.

    def __model_builder(self, in_shape, out_shape):
        cnn_in, cnn_out = cnn_for_raw_img(in_shape)
        model_out = Dense(out_shape, activation="softmax", name="out_layer")(cnn_out)
        self.__model = Model(inputs=cnn_in, outputs=model_out, name="CNN")
        print(self.__model.summary())

    def train(self, X_train, y_train, validation_data):
        if not self.__trained:
            self.__model_builder(X_train.shape[1:], y_train.shape[1])

            self.__model.compile(
                loss="categorical_crossentropy",
                optimizer=self.__optim,
                metrics=['accuracy']
            )

            td = self.train_datagen
            td.fit(X_train)
            if self.train_datagen is None:
                self.__history = self.__model.fit(
                    X_train, y_train,
                    validation_data=validation_data,
                    batch_size=self.__batch_size,
                    epochs=self.__epochs,
                    callbacks=self.callbacks,
                )
            else:
                steps_per_epoch = len(X_train) / self.__batch_size
                self.__history = self.__model.fit(
                    td.flow(X_train, y_train, batch_size=self.__batch_size),
                    validation_data=validation_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=self.__epochs,
                    callbacks=self.callbacks,
                )

            self.__history_epoch = self.__history.epoch
            self.__history = self.__history.history
            self.__trained = True

    def learn(self):
        if not self.__trained:
            df = self.load_data()
            df = self.fileter_data_by_emotions(df)
            img_array, img_labels = self.prepare_data(df)
            X_train, X_valid, self.__X_test, y_train, y_valid, self.__y_test = self.split_data_for_train_valid_test(
                img_array, img_labels)
            del df
            del img_array
            del img_labels
            X_train, X_valid, self.__X_test = self.normalize_data(X_train, X_valid, self.__X_test)
            self.train(X_train, y_train, (X_valid, y_valid))
            self.__model.save(MODELS_PATH + self.__model_name + ".h5")
            self.save_values()

    def save_model_architecture(self):
        plot_model(self.__model, show_shapes=True, show_layer_names=True, expand_nested=True,
                   dpi=50, to_file="models/" + "architectures/" + self.__model_name + ".png")

    def evaluate(self):
        if self.__trained:
            yhat_test = np.argmax(self.__model.predict(self.__X_test), axis=1)
            ytest_ = np.argmax(self.__y_test, axis=1)
            scikitplot.metrics.plot_confusion_matrix(ytest_, yhat_test, figsize=(7, 7))
            pyplot.savefig(CONFUSION_MATRIX_PATH + self.__model_name + ".png")
            pyplot.show()
        else:
            raise ValueError("Model is not trained yet, call train first")

    def show_validation_metric(self):
        if self.__trained:
            sns.set()
            fig = pyplot.figure(0, (12, 4))

            ax = pyplot.subplot(1, 2, 1)
            sns.lineplot(self.__history_epoch, self.__history['accuracy'], label='train')
            sns.lineplot(self.__history_epoch, self.__history['val_accuracy'], label='valid')
            pyplot.title('Accuracy')
            pyplot.tight_layout()

            ax = pyplot.subplot(1, 2, 2)
            sns.lineplot(self.__history_epoch, self.__history['loss'], label='train')
            sns.lineplot(self.__history_epoch, self.__history['val_loss'], label='valid')
            pyplot.title('Loss')
            pyplot.tight_layout()

            pyplot.savefig(EPOCH_HISTORY_PATH + self.__model_name + '.png')
            pyplot.show()

            df_accu = pd.DataFrame({'train': self.__history['accuracy'], 'valid': self.__history['val_accuracy']})
            df_loss = pd.DataFrame({'train': self.__history['loss'], 'valid': self.__history['val_loss']})

            fig = pyplot.figure(0, (14, 4))
            ax = pyplot.subplot(1, 2, 1)
            sns.violinplot(x="variable", y="value", data=pd.melt(df_accu), showfliers=False)
            pyplot.title('Accuracy')
            pyplot.tight_layout()

            ax = pyplot.subplot(1, 2, 2)
            sns.violinplot(x="variable", y="value", data=pd.melt(df_loss), showfliers=False)
            pyplot.title('Loss')
            pyplot.tight_layout()

            pyplot.savefig(PERFORMANCE_DIST_PATH + self.__model_name + '.png')
            pyplot.show()

    def predict(self, img):
        img = np.array(cv2.resize(img, (48, 48)), dtype=float)
        img = img.flatten()
        img = img.reshape([1, 48, 48, 1]) / 255.
        pr = self.__model.predict(img).flatten() * 100
        index, val = 0, 0
        for i in range(len(pr)):
            if pr[i] > val:
                index, val = i, pr[i]
        return {self.model_emotions[index]: int(val)}


def model_list():
    return [x.split('.')[0] for x in os.listdir(MODELS_PATH)]


def my_models():
    m = model_list()
    return [Emotion_model.load_model(x) for x in m]


def delete_model(model_name):
    if os.path.exists(CONFUSION_MATRIX_PATH + model_name + ".png"):
        os.remove(CONFUSION_MATRIX_PATH + model_name + ".png")
    if os.path.exists(EPOCH_HISTORY_PATH + model_name + ".png"):
        os.remove(EPOCH_HISTORY_PATH + model_name + ".png")
    if os.path.exists(PERFORMANCE_DIST_PATH + model_name + ".png"):
        os.remove(PERFORMANCE_DIST_PATH + model_name + ".png")
    if os.remove(VALUES_PATH + model_name + ".pickle"):
        os.remove(VALUES_PATH + model_name + ".pickle")
    if os.path.exists(MODELS_PATH + model_name + ".h5"):
        os.remove(MODELS_PATH + model_name + ".h5")
        return True
    return False
