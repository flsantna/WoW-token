import numpy as np
from config import list_key, total_epochs, test_rate, window, look_back, \
    early_stopping_patience, val_loss_on_train
from bin.model import LSTM_Model
from data_proc import Processing
from tensorflow import GradientTape
import tensorflow as tf
import time


def train_step(batch_data, batch_label):
    with GradientTape() as tape:
        predicted = model.call(batch_data)
        loss_value = loss(y_true=batch_label, y_pred=predicted)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
    loss_mean.update_state(values=loss_value)


def test_step(batch_data, batch_label):
    data_pred_test = batch_data[0]
    predict = model.call(inputs=tf.expand_dims(data_pred_test, axis=0))
    val_loss_value = loss(y_true=batch_label[0], y_pred=predict)
    val_loss_mean.update_state(values=val_loss_value)


if __name__ == '__main__':
    # Creation of an object that process the dataset.
    proc = Processing()

    # Model structure defined.
    loss = tf.keras.losses.MeanSquaredError()
    loss_mean = tf.keras.metrics.Mean()
    val_loss_mean = tf.keras.metrics.Mean()
    model = LSTM_Model(window=window, look_back=look_back)
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=5000,
                                                                  decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_scheduler, epsilon=0.002)

    for key in range(len(list_key)):

        start_time = time.time()
        data_x, data_y = proc.get_data(key=key, look_back=look_back, window=window)
        test_size = int(data_x.shape[0] * test_rate)
        data_train_x = data_x[:-test_size]
        data_train_y = data_y[:-test_size]
        # Val data
        data_test_x = data_x[-test_size:]
        data_test_y = data_y[-test_size:]

        val_loss_list = []
        for epoch in range(total_epochs):
            time_per_epoch = (time.time() - start_time) / (epoch + 1)
            train_step(batch_data=data_train_x, batch_label=data_train_y)
            if val_loss_on_train:
                test_step(batch_data=data_test_x, batch_label=data_test_y)
                val_loss_list.append(round(val_loss_mean.result().numpy(), ndigits=5))
                print("Epoch: {}/{}, {:.2f}s/epoch, Loss: {:.5f} Val Loss {:.5f}, "
                      "Estimated time to end all epochs: {:.0f}h:{:.0f}m"
                      .format(epoch,
                              total_epochs,
                              time_per_epoch,
                              loss_mean.result(),
                              val_loss_mean.result(),
                              time.localtime((time_per_epoch * total_epochs) + start_time).tm_hour,
                              time.localtime((time_per_epoch * total_epochs) + start_time).tm_min))
                if len(val_loss_list) == early_stopping_patience:
                    std = np.std(val_loss_list)
                    if std == 0:
                        break
                    val_loss_list = []

            else:
                print("Epoch: {}/{}, {:.2f}s/epoch, Loss: {:.5f}, Estimated time to end all epochs: {:.0f}h:{:.0f}m"
                      .format(epoch,
                              total_epochs,
                              time_per_epoch,
                              loss_mean.result(),
                              time.localtime((time_per_epoch * total_epochs) + start_time).tm_hour,
                              time.localtime((time_per_epoch * total_epochs) + start_time).tm_min))
        loss_mean.reset_states()
        val_loss_mean.reset_states()
        model.save_weights(filepath='model/w{}/lb{}_trained_with_{}_total_epochs{}'
                           .format(window, look_back, list_key[key], total_epochs), save_format='tf')
