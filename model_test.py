import numpy as np
import config
from config import list_key, total_epochs, look_back, window, configs_test
from bin.model import LSTM_Model
from data_proc import Processing, plot_graph, plot_graph_array, window_to_series
import tensorflow as tf
from config import test_type, predict_days

test_rate = 0.15

# Creation of an object that process the dataset.


data_test_x, data_test_y = [], []
if __name__ == '__main__':

    if test_type == 0:
        proc = Processing()
        model = LSTM_Model(window=window, look_back=look_back)
        model_load_path = 'model/w{}/lb{}_trained_with_{}_total_epochs{}'.format(window, look_back, list_key[4],
                                                                                 total_epochs)
        all_predicts = []
        all_true_data = []
        all_train_data = []
        for key in range(len(list_key)):
            data_x, data_y = proc.get_data(key=key, look_back=look_back, window=window)
            test_size = int(data_x.shape[0] * test_rate)

            data_test_x = data_x[-test_size:]
            data_train_y = data_y[:-test_size]
            data_test_y = data_y[-test_size:]

            model.load_weights(filepath=model_load_path)
            predict = model.call(data_test_x)

            data_in_series_y = window_to_series(data_test_y, window=window)
            data_pred_series_y = window_to_series(predict, window=window)
            data_train_series_y = window_to_series(data_train_y, window=window)

            rmse = tf.math.reduce_mean(
                tf.keras.losses.mean_squared_error(y_true=data_in_series_y, y_pred=data_pred_series_y))
            print('{} RMSE: {}%'.format(list_key[key], rmse))

            plot_graph(true_data=np.append(data_train_series_y, data_in_series_y, axis=0), window=window,
                       predicted_data=[np.append(data_train_series_y, data_pred_series_y, axis=0),
                                       '{} : RMSE: {:.5f}'.format(list_key[key], rmse.numpy())],
                       title=list_key[key], plot_predict=True, plot=False)

    elif test_type == 1:
        all_predicts = []
        all_true_data = []
        all_train_data = []
        for key in range(len(list_key)):

            for day in range(len(predict_days)):
                window = configs_test[day][0]
                look_back = configs_test[day][1]
                config.window = window
                config.look_back = look_back
                proc = Processing()
                model = LSTM_Model(window=window, look_back=look_back)
                model_load_path = 'model/w{}/lb{}_trained_with_{}_total_epochs{}' \
                    .format(window, look_back, list_key[4], total_epochs)

                data_x, data_y = proc.get_data(key=key, window=window, look_back=look_back)
                test_size = int(data_x.shape[0] * test_rate)
                data_test_x = data_x[-test_size:]
                data_test_y = data_y[-test_size:]
                data_train_y = data_y[:-test_size]

                model.load_weights(filepath=model_load_path)

                data_pred_test = data_test_x[0]
                predict_list = []
                for index in range(data_test_x.shape[0]):
                    predict_value = model.call(inputs=tf.expand_dims(data_pred_test, axis=0))
                    predict_list.append(tf.squeeze(predict_value, axis=0))
                    predict_value = tf.reshape(predict_value, shape=[window, 1])
                    data_transfer = data_pred_test[..., 1:]
                    data_transfer = np.concatenate([data_transfer, predict_value], axis=1)
                    data_pred_test = data_transfer

                predict = tf.cast(predict_list, dtype=tf.dtypes.float32)

                data_in_series_y = window_to_series(data_test_y, window=window)
                data_pred_series_y = window_to_series(predict, window=window)
                data_train_series_y = window_to_series(data_train_y, window=window)
    
                """ #to inverse the normalization
                data_test_y = proc.inverse_normalization(data_test_y,key=list_key[key])
                predict = proc.inverse_normalization(predict,key=list_key[key])
                data_train_y = proc.inverse_normalization(data_train_y,key=list_key[key])
                """

                pred_y_unit = data_pred_series_y[:window]
                test_y_unit = data_in_series_y[:window]
                test_x_unit = data_train_series_y[-look_back:]

                rmse = tf.math.sqrt(
                    tf.keras.losses.mean_squared_error(y_true=data_in_series_y, y_pred=data_pred_series_y))
                rmse_2 = tf.math.sqrt(tf.keras.losses.mean_squared_error(y_true=test_y_unit, y_pred=pred_y_unit))
                print('{} RMSE: {}'.format(list_key[key], rmse))

                all_predicts.append([pred_y_unit, rmse_2.numpy(), predict_days[day]])
                all_true_data.append(test_y_unit)
                all_train_data.append(test_x_unit)

                plot_graph(true_data=np.append(data_train_series_y, data_in_series_y, axis=0), window=window,
                           predicted_data=[np.append(data_train_series_y, data_pred_series_y, axis=0),
                                           '{} : RMSE: {:.5f}'.format(list_key[key], rmse.numpy())],
                           title=list_key[key], plot_predict=True, plot=False)

                plot_graph(true_data=np.append(test_x_unit, test_y_unit, axis=0), window=window,
                           predicted_data=[np.append(test_x_unit, pred_y_unit, axis=0),
                                           '{} : RMSE: {:.5f}'.format(list_key[key], rmse_2.numpy())],
                           title=list_key[key], plot_predict=True, plot=False)

        for index in range(len(list_key)):
            plot_graph_array(true_data=all_true_data[len(predict_days) * index+len(predict_days) -1
                                                     :len(predict_days) * index+len(predict_days)][0],
                             predicted_data=all_predicts[len(predict_days) * index:
                                                         index * len(predict_days) + len(predict_days)],
                             title=list_key[index], plot=True)