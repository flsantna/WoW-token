from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, GlobalMaxPool2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

import tensorflow as tf
class LSTM_Model(Model):
    def __init__(self, window, look_back):
        super().__init__()

        self.model = Sequential()
        self.model.add(LSTM(units=256, return_sequences=True, input_shape=(window, look_back)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=256, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=256, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=256, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=128))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128))
        self.model.add(Dense(window))

    def call(self, inputs, training=None, mask=None):
        output = self.model(inputs)
        return output

    def create_model(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss)
        return self.model
