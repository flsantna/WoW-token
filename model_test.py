import numpy as np
import pandas as pd

from config import list_key, total_epochs, look_back, window, configs_test, threshold_eval
from bin.model import LSTM_Model
from data_proc import Processing, plot_graph, window_to_series
import tensorflow as tf
from config import test_type


def map_threshold(data, threshold=threshold_eval):
    if data[0] > threshold:
        predict = 1
        return predict
    else:
        predict = 0
        return predict


def map_classification(data, predict):
    if predict >= data:
        predict = 1
        return predict
    else:
        predict = 0
        return predict


def map_comparison(item_x, item_y):
    if item_x == item_y:
        return True
    else:
        return False


if __name__ == '__main__':
    test_rate = 0.15
    data_test_x, data_test_y = [], []
    accuracy_log = pd.DataFrame(columns=['Accuracy in %','RMSE'],index=list_key)
    next_day_class_log = pd.DataFrame(columns=['Prediction', 'Real Value'], index=list_key)

    # Test type 0 is a test that only shows one graph and the testing is done using the last true data as look back.
    if test_type == 0:
        # Creating class that process all the data from the CSV in "dataset/"
        proc = Processing()
        # Model creation, as parameters is needed window and look_back value.
        model = LSTM_Model(window=window, look_back=look_back)
        loss = 'binary_crossentropy'
        metrics = ["accuracy"]
        optimizer = 'adam'
        model = model.create_model(optimizer=optimizer, loss=loss, metrics=metrics)
        for key in range(len(list_key)):
            model_load_path = 'model/w{}/lb{}_trained_with_{}_total_val_loss_mean.result()epochs{}' \
                .format(window, look_back, list_key[key], total_epochs)

            data_x, data_y = proc.get_data(key=key, look_back=look_back, window=window)
            test_size = int(data_x.shape[0] * test_rate)
            data_train_x = data_x[:-look_back]
            data_train_y = data_y[:-look_back]
            # Val data
            data_test_x = data_x[-look_back:]
            data_test_y = data_y[-look_back:]

            # Dates str for plot.
            dates = proc.dates_list

            model.load_weights(filepath=model_load_path)
            prediction = model.call(data_test_x)

            prediction_map = list(map(map_threshold, prediction.numpy().tolist()))

            accuracy = list(map(map_comparison, data_test_y, prediction_map))
            rmse = tf.math.sqrt(
                tf.keras.losses.mean_squared_error(y_true= tf.squeeze(data_test_x, axis=-1)[..., -1:], y_pred=prediction))
            accuracy_log.loc[list_key[key]]['Accuracy in %'] = (sum(accuracy)/look_back)*100
            accuracy_log.loc[list_key[key]]['RMSE'] = np.mean(rmse.numpy())

            print("Accuracy: ", sum(accuracy), "/", len(accuracy))
            print("RMSE: {}".format(np.mean(rmse.numpy())))

            next_day_class_log.loc[list_key[key]]['Prediction'] = prediction_map[0]
            next_day_class_log.loc[list_key[key]]['Real Value'] = int(data_test_y.numpy()[0][0])

        accuracy_log.to_csv("output/classification_accuracy.csv", index_label="Location")
        next_day_class_log.to_csv("output/next_day_classification.csv", index_label="Location")





