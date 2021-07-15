## Prediction of WoWtoken price in gold using LSTM networks.

WoWtoken is an item that can be purchased on World of Warcraft store and be sold in game by it's currency, that is gold
and can be used to redeem gameplay time as the game is monthly subscribed for $13,99/month.
So, the goal is create an model that can predict the next 3 days, 7 days, 14 days and 28 days of gold price in USD.

## Data Processing

For data processing, was used mean normalization and to guarantee a good amount of data, was used prices from the
regions Us, Eu, China,Korea and Taiwan, and before normalizing the data it was needed a conversion to all of them have
the same base currency, so was obtained from https://www.investing.com the historical conversion from current currency
to USD, and matched it with the original price in each region of WoWtoken it self, that lacks historical value but as
was researched, haven't changed over the years.
                             Prices of WoW token on Blizzard service.
                             
| Region	  |  Currency  |	WoW token price in currency   |
| --------- | ---------- | ------------------------------ |
| US	      |  USD       |  	        $20.00              |    
| EU	      |  EUR	     |           20.00 €              |
| CHINA	    |  CNY	     |           ￥75.00              |
| KOREA     |  KRW	     |          ￦22,000              |            
| TAIWAN	  |  TWD	     |         NT$500.00              |  
                         



After all processing, the final data was from Gold per WoW token in each region to gold per USD, as was correlated currency
conversion to WoW token price in each region resulting them to an gold per USD base.


## Training and evaluation of the model

The final model was constructed with 4 lstm layers and as output layer, one dense with output of X units, equal to the
selected window in config.py file. As input, the model get the data in the format [Batch, Window, Lookback] as window
being the prediction window, 3,, 7, 14 or 28 units, and lookback, that is the amount of data that the model receives to
do its prediction.

LSTM layer need it's data to be in window format, a window that moves through the dataset and creates overlapping between
next and previous windows. For that to happen, was choose that the window moves 1 data per windows, so every data that is
feeded to the model it is in general, 1 unit forward the previous one. As exemple:

                                      Lookback                     Window

                                    [1, 2, 3, 4, 5]                 [6, 7, 8]
                                    [2, 3, 4, 5, 6]                 [7, 8, 9]
                                                             .
                                                             .
                                                             .
                                    [95, 96, 97, 98, 99]         [100, 101, 102]

This leads to the input shape [Quantity of sequences, 3, 5].

After defined the model, got up to training it. Was configured the Early stopping to avoid overfiting and save time. As
base, was chosen 1500 epochs to train the model, but in average, it finished in 800 epochs of no change in last 10 epochs.

## Result

As result, was generated some range of predictions, they are:

1) All test dataset predicted using true data as training, in a cenário where are the need to get an insight about it, and them the model can be feeded with the true data to predict the next window. Images on graphs/ with suffix "True Data as lookback".
2) All test dataset predicted using, as the model moves foward predicting, it's own predicitions, which leads overall a bigger RMSE. Images on graphs/ with suffix "all data".
3) Predictions of only the chosen windows. Images on graphs/ with suffix "days predicted".

All predicted windows in one image with their own RMSE.

<img src="https://github.com/flsantna/WoW-token/blob/master/graphs/us%20-%20all%20predicts.png" width="40%" height="40%"> <img src="https://github.com/flsantna/WoW-token/blob/master/graphs/china%20-%20all%20predicts.png" width="40%" height="40%">
<img src="https://github.com/flsantna/WoW-token/blob/master/graphs/eu%20-%20all%20predicts.png" width="40%" height="40%"> <img src="https://github.com/flsantna/WoW-token/blob/master/graphs/korea%20-%20all%20predicts.png" width="40%" height="40%">
<img src="https://github.com/flsantna/WoW-token/blob/master/graphs/taiwan%20-%20all%20predicts.png" width="40%" height="40%">

Other graphs can be found at [https://github.com/flsantna/WoW-token/tree/master/graphs].

## Instructions to use

The project is mainly focused on 3 scripts:

1) [https://github.com/flsantna/WoW-token/blob/master/creation_of_csv_wowtoken_price.py] - Creates the base CSV file, colecting data on [https://wowtokenprices.com] and parsing the json to csv while filling all missing values and croping them.
2) [https://github.com/flsantna/WoW-token/blob/master/model_train.py] - Train the model, calling [https://github.com/flsantna/WoW-token/blob/master/data_proc.py] to process the dataset, normalizing its values and if chosen in config.py, convert all bases to gold/usd.
3) [https://github.com/flsantna/WoW-token/blob/master/model_test.py] - Test the model and plot all the graphs.

## Used packages and versions

| Package    | Version |
| ---------- | ------ |
| tensorflow | 2.5.0 |
| numpy | 1.19.5 |
| pandas | 1.2.4 |
| matplotlib | 3.4.2 |
