import pandas as pd
import numpy as np
import urllib.request as url_req
import json
from config import list_key, columns, dataset_csv_path
from data_proc import date_conver, date_index_dict


if __name__ == '__main__':
    print("Requesting data from https://wowtokenprices.com/history_prices_full.json", '\n')
    dataset_url = url_req.urlopen("https://wowtokenprices.com/history_prices_full.json")
    dataset = json.load(dataset_url)
    print("Done!", '\n')

    dates_index, dates_dict = date_index_dict()
    # The pandas dataframe to receive the transformed data.
    dataframe = pd.DataFrame(data=[], index=dates_index, columns=columns)

    # Loop through dataset key names and formatting each column and dropping duplicated information from same days.
    for key in list_key:
        print("Processing {} wow token prices in gold.".format(key))
        pd_data = pd.DataFrame(dataset[key], columns=["price", "time"])
        pd_data['time'] = list(map(date_conver, pd_data['time']))
        pd_data.rename(columns={"price": key}, inplace=True)
        pd_data = pd_data.drop_duplicates(subset="time", keep='last', ignore_index=True)
        list_index = []
        for value in pd_data["time"]:
            list_index.append(dates_dict[value])
        pd_data['time'] = list_index
        data_array = pd_data.to_numpy()
        for value_key, time_value in data_array:
            dataframe[key][time_value] = value_key
        if key == "taiwan":
            dataframe[key][718:730] = np.NAN
        print("Done with {} wow token prices!".format(key), '\n')

    # Removing all nan values from beginning and at the end of the dataset.
    print("Dropping the first 80 rows of data, as they are missing in all categories instead US...", '\n')
    dataframe = dataframe[80:]
    dataframe = dataframe[:-1]
    # Filling all none existent data with np.NAN to proper interpolate.
    dataframe = dataframe.fillna(np.NAN)

    # Filling up all np.NAN windows inside the dataset.
    for key in list_key:
        dataframe[key] = dataframe[key].fillna(dataframe[key].interpolate(method='linear', limit_direction='backward'))

    # Concatenating the dict data keys with the processed dataset.
    dataframe = pd.concat([dataframe, pd.DataFrame(dates_dict.keys(), columns=['date'])[80:]], axis=1)

    # Exporting the dataset to a csv file.
    dataframe.to_csv(path_or_buf=dataset_csv_path)
    print("Final Dataset exported!", '\n')
    print(dataframe)
    print("Check if there is any missing NaN value.", '\n')
    print(dataframe.isna().sum)
