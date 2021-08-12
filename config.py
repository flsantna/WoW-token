# Dataset creation config
list_key = ["us", "eu", "china", "korea", "taiwan"]
columns = ["us", "eu", "china", "korea", "taiwan"]
dataset_csv_path = "dataset/dataset_complete.csv"
# Conversion variables.
convert_to_USD = True
list_currency_types = {"eu": 'EUR', "china": 'CNY', "korea": "KRW", "taiwan": "TWD"}
# Price per WoWtoken.
currency_token = {"us": 20, "eu": 20, "china": 75, "korea": 22000, "taiwan": 500}

# "mean" or "min_max"
normalization = "mean"

# LSTM config
configs_test = [[1, 7], [1, 14], [1, 30]]
predict_choice = 1
total_epochs = 1000
batch_size = 100
test_rate = 0.15
val_loss_on_train = True
early_stopping_patience = 20

window = configs_test[predict_choice][0]
look_back = configs_test[predict_choice][1]

# Test types, 0 for generating the last data looking back the look_back value in number of data's. 1 for generating
# all values giving to it only the look_back data, and them generating it's own data to generate the curve, which
# can be tweaked in predict_days and inside the code in model_test.py.
test_type = 1
predict_days = [1, 1, 1]
threshold = 0.7
