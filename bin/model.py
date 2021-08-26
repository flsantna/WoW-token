from abc import ABC

from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, GRU, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model


class LSTM_Model(Model, ABC):
    def __init__(self, window, look_back, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Sequential()
        self.model.add(LSTM(units=512, input_shape=(look_back, window), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=256, return_sequences=True))
        self.model.add(LSTM(units=256, return_sequences=True))
        self.model.add(LSTM(units=256))
        self.model.add(Flatten())
        self.model.add(Dense(1))

    def call(self, inputs, training=None, mask=None):
        output = self.model(inputs)
        return output

    def create_model(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss)
        return self.model
