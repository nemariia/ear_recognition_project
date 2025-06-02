import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten


class EarRecognitionModel:
    def __init__(self, n_classes, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.n_classes = n_classes

    '''
    A model with the architecture from the article by
    Hidayati, N., Maulidah, M., & Saputra, E. P. (2022)
    '''
    def cnn_h_m_s(self):
        model = Sequential([
            Input(shape=self.input_shape),
            Conv2D(16, kernel_size=(2, 2), strides=1, activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2),

            Conv2D(32, kernel_size=(2, 2), strides=1, activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2),

            Conv2D(64, kernel_size=(2, 2), strides=1, activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2),

            Flatten(),

            Dense(500, activation='relu'),
            Dense(self.n_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    