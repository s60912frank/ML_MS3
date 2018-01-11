import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
import os

from read_config import *
import util as _

class VisualModel:
    def __init__(self):
        cfg_name = 'visual'
        # Get all settings from config file
        self.lr = get_float(cfg_name, 'learning_rate')
        self.batch_size = get_int(cfg_name, 'batch_size')
        self.epochs = get_int(cfg_name, 'epochs')
        self.num_classes = get_int(cfg_name, 'num_classes')
        self.data_augmentation = get_bool(cfg_name, 'data_augmentation')
        self.save_dir = os.path.join(os.getcwd(), get_str(cfg_name, 'model_save_dir'))
        self.model_name = get_str(cfg_name, 'model_name')
        # If saved model exist, we skip training
        if self.__check_model_exist():
            print('Model exist, abort training visual model.')
            return
        print('Start training vistul model.')
        self.load_data()
        self.build_model()
        self.train_ops()
        self.save_model()
        print('Vistul model training finished.')

    def __check_model_exist(self):
        # Check if saved model exist.
        return os.path.exists(os.path.join(self.save_dir, self.model_name))

    def load_data(self):
        x_train, y_train = _.load_train_set()
        x_test, y_test = _.load_test_set()
        # Some simple preprocess.
        self.x_train = x_train.astype('float32') / 255
        self.x_test = x_test.astype('float32') / 255
        # Convert class vectors to binary class matrices.
        self.y_train = to_categorical(y_train, self.num_classes)
        self.y_test = to_categorical(y_test, self.num_classes)
        print('Data load complete.')

    def build_model(self):
        # Build model structure
        model = Sequential()
        # Convolution layers
        model.add(Conv2D(64, (3, 3), padding='same',
                        input_shape=self.x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), padding='same',
                        input_shape=self.x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, (3, 3), padding='same',
                        input_shape=self.x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(2048))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        model.summary()

        self.model = model

    def train_ops(self):
        # Actually train neural network
        # Set optimizers, we use adam
        opt = keras.optimizers.adam(lr=self.lr)
        # Use cross entropy as our loss function
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
        
        if not self.data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(self.x_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=(self.x_test, self.y_test),
                        shuffle=True)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            # Use data augmentation to generate more sample
            datagen.fit(self.x_train)

            # Fit the model on the batches generated by datagen.flow().
            self.model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                                steps_per_epoch=int(np.ceil(self.x_train.shape[0] / float(self.batch_size))),
                                epochs=self.epochs,
                                validation_data=(self.x_test, self.y_test),
                                workers=4)

        
    def save_model(self):
        # Save model and weights
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        model_path = os.path.join(self.save_dir, self.model_name)
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)