import numpy as np
import pandas as pd
from config import list_key, convert_to_USD, normalization
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import datetime as dt


def plot_graph(true_data, window, dates, test_size=0, predicted_data=None, plot_predict=False, title='',
               plot=False, file_name='days predicted', plot_tick=5):
    if predicted_data is None:
        predicted_data = []
    fig, ax = plt.subplots()

    if plot_predict:
        ax.plot(predicted_data[0], label=predicted_data[1], color='red')
    ax.plot(true_data, label="Real data", color='blue')
    ax.grid()
    # position of the legend
    ax.legend(loc='best', shadow=False, fontsize='small')
    ax.set_title('WoWtokens price from {}. Predicted {} days.'.format(title.capitalize(), window))
    # X legend and Y legend
    ticks_xaxis = np.arange(0, true_data.shape[0], true_data.shape[0]//plot_tick)
    plt.xticks(ticks_xaxis, labels=[dates[i] for i in ticks_xaxis], rotation=20)
    if not test_size == 0:
        plt.axline((true_data.shape[0]-test_size, np.max(predicted_data[0])*0.8),
                   (true_data.shape[0]-test_size, np.min(predicted_data[0])*0.8), color='black', lw=1.5)
    plt.xlabel('Dates')
    plt.ylabel('Normalized dataset')
    plt.subplots_adjust(left=0.125, bottom=0.152, right=0.9, top=0.88, wspace=0.2, hspace=0.2)
    plt.savefig('graphs-2/{} - {} {}'.format(title, file_name, window))
    if plot:
        plt.show()


def plot_graph_array(true_data, dates, predicted_data, test_size=0, title='',
                     plot=False, file_name="all predicts", plot_tick=5):
    fig, ax = plt.subplots()
    # legends and colors
    # ax.plot(data2, color='#d7191c', label=data2_name) # ls=--
    colors = ['g', 'r', 'c', 'm', 'y', ]
    for i in range(len(predicted_data)):
        predict_days = predicted_data[i][2]
        ax.plot(predicted_data[i][0], label=str(predict_days)+" days. RMSE: "+str(predicted_data[i][1]),
                color=colors[i])
    ax.plot(true_data, label="Real data", color='b')
    ax.grid()
    # position of the legend
    ax.legend(loc='best', shadow=False, fontsize='small')
    ax.set_title('WoWtokens price from ' + title.capitalize())
    # X legend and Y legend
    ticks_xaxis = np.arange(0, true_data.shape[0], true_data.shape[0] // plot_tick)
    plt.xticks(ticks_xaxis, labels=[dates[i] for i in ticks_xaxis], rotation=20)
    if not test_size == 0:
        plt.axline((true_data.shape[0]-test_size, np.max(predicted_data[0])*0.8),
                   (true_data.shape[0]-test_size, np.min(predicted_data[0])*0.8), color='black', lw=1.5)
    plt.xlabel('Dates')
    plt.ylabel('Normalized dataset')
    plt.subplots_adjust(left=0.125, bottom=0.152, right=0.9, top=0.88, wspace=0.2, hspace=0.2)
    plt.savefig('graphs-2/{} - {}'.format(title, file_name))
    if plot:
        plt.show()


def date_time(x):
    return x.timestamp()


def date_conver(x):
    return str(time.localtime(x).tm_year) + '/' + str(time.localtime(x).tm_mon) + '/' + str(time.localtime(x).tm_mday)


def date_inverse_conver(x):
    time_convert = dt.datetime.strptime(x, "%b %d, %Y")
    time_convert = time_convert - dt.datetime(1970, 1, 1)
    return time_convert.total_seconds()


def date_index_dict(y=2015, m=4, d=12):
    # First day of series of data. Using timedelta 1 to manipulate the indexing in python.
    d_day = dt.datetime(year=y, month=m, day=d) + dt.timedelta(days=1)
    t_day = dt.datetime.today() + dt.timedelta(days=1)
    range_date = pd.date_range(start=d_day, end=t_day).to_list()

    # Creation of a dict to index the dataset throughout clean dates.
    date_list = list(map(date_time, range_date))
    dates_index = list(range(len(date_list) + 1))
    dates_dict = dict(zip(list(map(date_conver, date_list)), list(range(len(date_list)))))
    return dates_index, dates_dict


def window_to_series(array, window):
    # Convert an array of windows in a series forward one step.
    series = np.array(array[0])
    for i in array[1:]:
        temp_array = i[-window:]
        series[(1-window):] = (series[1-window:]+temp_array[:window-1])/2
        series = np.append(series, values=np.reshape(i[-1], newshape=[1]), axis=0)
    return series


class Processing(object):
    def __init__(self):
        self.full_dataset = pd.read_csv('dataset/dataset_complete.csv', index_col=0)
        self.date_index = self.full_dataset['date'].to_numpy()
        self.dataset = self.full_dataset.drop('date', axis=1)
        self.list_currency_types = {"eu": 'EUR', "china": 'CNY', "korea": "KRW", "taiwan": "TWD"}
        self.currency_token = {"us": 20, "eu": 20, "china": 75, "korea": 22000, "taiwan": 500}
        _, self.dates_dict = date_index_dict()
        self.dates_list = []
        # Variables to store max_min and mean_std values to further inverse_normalization if needed.
        self.max_min = []
        self.mean_std = []
        self.keys = []
        # Normalized final dataset.
        self.normalized_data = self.normalization()

    def currency_proc(self, key):
        currency = pd.read_csv('dataset/USD_' + self.list_currency_types[key] + ' Historical Data.csv')
        currency["Date"] = list(map(date_inverse_conver, currency['Date']))
        currency = currency.sort_values(by=['Date'], ascending=True)
        currency["Date"] = list(map(date_conver, currency['Date']))
        list_index = []
        for value in currency["Date"]:
            list_index.append(self.dates_dict[value])
        currency['Date'] = list_index
        currency_output = pd.DataFrame(currency, columns=["Date", "Price"])
        currency_output_final = range(0, self.full_dataset.index.values[-1])
        currency_dataframe = pd.DataFrame([], index=currency_output_final, columns=["Price"])

        for i in currency_output.index.to_list():
            currency_dataframe["Price"][currency_output["Date"][i]] = float(
                str(currency_output["Price"][i]).replace(',', ''))
        currency_dataframe["Price"] = list(map(np.float, currency_dataframe["Price"].to_list()))
        currency_output = currency_dataframe.fillna(currency_dataframe.interpolate
                                                    (method='linear', limit_direction='forward'))
        return currency_output.to_numpy()[self.full_dataset.index.values[0]: 2250]

    def normalization(self):
        # Cutting to 2170 to match the range of data in currency conversion arrays and dataset.
        data = self.dataset[:2170]
        for text in list_key:
            if not text == 'us' and convert_to_USD:
                convert_currency = np.reshape(self.currency_proc(text),
                                              newshape=[self.currency_proc(text).shape[0], ])
                data[text] = (data[text].to_numpy()*convert_currency) / self.currency_token[text]
            elif text == "us" and convert_to_USD:
                data[text] = data[text].to_numpy() / self.currency_token[text]
        dataset = data
        if normalization == "min_max":
            normalized_dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
            self.max_min = [dataset.max(), dataset.min()]
            self.keys = dataset.columns.to_list()
        elif normalization == "mean":
            normalized_dataset = (dataset - dataset.mean()) / (dataset.std())
            self.mean_std = [dataset.mean(), dataset.std()]
            self.keys = dataset.columns.to_list()
        else:
            normalized_dataset = (dataset - dataset.mean()) / (dataset.std())
            self.max_min = [dataset.max(), dataset.min()]
            self.keys = dataset.columns.to_list()

        # Final list of range of dates in normalized_dataset.
        self.dates_list = list(self.dates_dict.keys())[normalized_dataset.index.values.tolist()[0]:
                                                       (normalized_dataset.index.values.tolist()[-1] + 1)]

        return normalized_dataset.to_numpy()

    def inverse_normalization(self, data, key=False):
        if normalization == "min_max":
            if not key:
                keys = self.keys
            else:
                keys = key
            data = data * (self.max_min[0][keys] - self.max_min[1][keys]) + self.max_min[1][keys]
        elif normalization == "mean":
            if not key:
                keys = self.keys
            else:
                keys = key
            data = data * self.mean_std[1][keys] + self.mean_std[0][keys]
        return data

    def get_data(self, key, look_back, window):
        x_data, y_data = [], []
        data = self.normalized_data
        for i in range((data[..., key].shape[0]-look_back-window)):
            value = np.ndarray(shape=[window, look_back])
            value_output_window = []
            for j in range(window):
                value[j] = np.reshape(data[(i+j):look_back + (i+j), key], newshape=[1, look_back])
                value_output = data[(i+j) + look_back:1 + (i+j + look_back), key]
                value_output_window = value_output
                if value_output_window > value[0][-1]:
                    value_output_window = [1, 0]
                else:
                    value_output_window = [0, 1]

            x_data.append(value)
            y_data.append(value_output_window)

        x_data = tf.cast(x_data, dtype=tf.dtypes.float32)
        y_data = tf.cast(y_data, dtype=tf.dtypes.float32)
        return x_data, y_data

    def real_dataset(self, key):
        return self.normalized_data[..., key]
