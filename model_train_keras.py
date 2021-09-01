import pandas as pd

from config import list_key, total_epochs, test_rate, window, look_back, \
    early_stopping_patience, val_loss_on_train, batch_size, train_in_batch, threshold_eval
from bin.model import LSTM_Model
from bin.early_stop import EarlyStop, BreakException
from data_proc import Processing
from tensorflow import GradientTape
import tensorflow as tf
import time
from pandas import DataFrame, concat, Series
import numpy as np


# Creation of an object that process the dataset.
proc = Processing()

# Model structure defined.
model = LSTM_Model(window=window, look_back=look_back)
loss = 'binary_crossentropy'
metrics = ["accuracy"]
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.7)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


model = model.create_model(optimizer=optimizer, loss=loss, metrics=metrics)

# EarlyStoping

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=early_stopping_patience,
    restore_best_weights=True,
)

# create a log of loss and validation loss
loss_log = DataFrame(index=range(total_epochs))

for key in range(len(list_key)):

    start_time = time.time()
    data_x, data_y = proc.get_data(key=key, look_back=look_back, window=window)
    data_train_x = data_x[:-look_back]
    data_train_y = data_y[:-look_back]
    # Val data
    data_test_x = data_x[-look_back:]
    data_test_y = data_y[-look_back:]

    history = model.fit(data_train_x, data_train_y, batch_size=batch_size, epochs=total_epochs
                        ,validation_data=(data_test_x, data_test_y), shuffle=False, callbacks=[early_stopping])

    loss_list = history.history
    loss_log = concat([loss_log, Series(loss_list['loss']).rename("loss " + list_key[key])], axis=1)
    loss_log = concat([loss_log, Series(loss_list['accuracy']).rename("accuracy " + list_key[key])], axis=1)
    loss_log = concat([loss_log, Series(loss_list['val_loss']).rename("loss " + list_key[key])], axis=1)
    loss_log = concat([loss_log, Series(loss_list['val_accuracy']).rename("accuracy " + list_key[key])], axis=1)

    model.save_weights(filepath='model/w{}/lb{}_trained_with_{}_total_val_loss_mean.result()epochs{}'
                       .format(window, look_back, list_key[key], total_epochs), save_format='tf')
loss_log.to_csv("output/loss_log_keras.csv", index_label="Epochs")
