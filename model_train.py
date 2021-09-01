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


def weighted_binary_crossentropy(y_true, y_pred):
    """
    Weighted binary crossentropy between an output tensor
    and a target tensor. POS_WEIGHT is used as a multiplier
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """

    # Transform back to logits.
    epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), dtype=tf.dtypes.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true,
                                                    logits=y_pred,
                                                    pos_weight=2)
   # return loss
    return tf.reduce_mean(loss, axis=-1)


def map_classification(x_data):
    if x_data[0] > threshold_eval:
        x_data = [1, 0]
        return x_data
    else:
        x_data = [0, 1]
        return x_data


def train_step(batch_data, batch_label):
    with GradientTape() as tape:
        predicted = model.call(batch_data)
        loss_value = loss(y_true=batch_label, y_pred=predicted)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
    loss_mean.update_state(values=loss_value)


def test_step(batch_data, batch_label):
    data_pred_test = batch_data
    predict_test = model.call(inputs=data_pred_test)
    val_loss_value = loss(y_true=batch_label, y_pred=predict_test)
    val_loss_mean.update_state(values=val_loss_value)


def get_batch_data(data, batch_size, index):
    data_batch = data[index*batch_size:index*batch_size + batch_size]
    return data_batch


# Creation of an object that process the dataset.
proc = Processing()

# Model structure defined.
loss = tf.keras.losses.BinaryCrossentropy()
loss_mean = tf.keras.metrics.Mean()
val_loss_mean = tf.keras.metrics.Mean()
model = LSTM_Model(window=window, look_back=look_back)
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=15000,
                                                              decay_rate=0.97)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

# create a log of loss and validation loss
loss_log = DataFrame(index=range(total_epochs))

for key in range(len(list_key)):

    start_time = time.time()
    data_x, data_y = proc.get_data(key=key, look_back=look_back, window=window)
    test_size = int(data_x.shape[0] * test_rate)
    data_train_x = data_x[:-test_size]
    data_train_y = data_y[:-test_size]
    # Val data
    data_test_x = data_x[-test_size:]
    data_test_y = data_y[-test_size:]

    # Create a log for capture loss and validation loss
    loss_list = []
    val_loss_list = []

    early_stopping = EarlyStop()

    # Step
    train_step_size = (data_train_x.shape[0] // batch_size) + 1
    test_step_size = (data_test_x.shape[0] // batch_size) + 1

    for epoch in range(total_epochs):
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        if train_in_batch:
            for step in range(train_step_size):
                data_train_x_batch = get_batch_data(data=data_train_x, batch_size=batch_size, index=step)
                data_train_y_batch = get_batch_data(data=data_train_y, batch_size=batch_size, index=step)
                train_step(batch_data=data_train_x_batch, batch_label=data_train_y_batch)
#                print("Epoch: {}/{},{}/{} steps, Loss: {:.5f}"
#                      .format(epoch, total_epochs, step, train_step_size, loss_mean.result()))
        else:
            train_step(batch_data=data_train_x, batch_label=data_train_y)
        if val_loss_on_train:
            test_step(batch_data=data_test_x, batch_label=data_test_y)
            loss_list.append(loss_mean.result().numpy())
            val_loss_list.append(val_loss_mean.result().numpy())
            print("Epoch: {}/{},{:.2f}s/epoch, Loss: {:.5f} Val Loss {:.5f}, "
                  "Estimated time to end all epochs: {:.0f}h:{:.0f}m"
                  .format(epoch,
                          total_epochs,
                          time_per_epoch,
                          loss_mean.result(),
                          val_loss_mean.result(),
                          time.localtime((time_per_epoch * total_epochs) + start_time).tm_hour,
                          time.localtime((time_per_epoch * total_epochs) + start_time).tm_min))
            try:
                early_stopping.check(loss=round(val_loss_mean.result().numpy(), ndigits=5),
                                     model=model, patience=early_stopping_patience)
            except BreakException:
                print("No improvement by the last {} epochs. Loaded best model!!!".format(early_stopping_patience))
                model = early_stopping.get_best_model()
                break

        else:
            print("Epoch: {}/{}, {:.2f}s/epoch, Loss: {:.5f}, Estimated time to end all epochs: {:.0f}h:{:.0f}m"
                  .format(epoch,
                          total_epochs,
                          time_per_epoch,
                          loss_mean.result(),
                          time.localtime((time_per_epoch * total_epochs) + start_time).tm_hour,
                          time.localtime((time_per_epoch * total_epochs) + start_time).tm_min))
    loss_log = concat([loss_log, Series(loss_list).rename("loss " + list_key[key])], axis=1)
    loss_log = concat([loss_log, Series(val_loss_list).rename("val_loss " + list_key[key])], axis=1)
    loss_mean.reset_states()
    val_loss_mean.reset_states()
    model.save_weights(filepath='model/w{}/lb{}_trained_with_{}_total_val_loss_mean.result()epochs{}'
                       .format(window, look_back, list_key[key], total_epochs), save_format='tf')
loss_log.to_csv("output/loss_log.csv", index_label="Epochs")
