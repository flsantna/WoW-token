## Prediction of WoWtoken price in gold using LSTM networks.

The project was a different approach to [Correlation](https://github.com/Cividati-inatel-ic/correlation) and [Prediction](https://github.com/cividati-inatel-ic/Prediction) applying lstm deep learning neural networks to predict in 4 differents windows and 2 ways of generating a data and compared to the original test dataset that was not presented to the model during training.

As said by [Cividati](https://github.com/cividati) a WoWtoken is an item that can be purchased on World of Warcraft store and be sold in game by it's currency, that is gold and can be used to redeem gameplay time as the game is monthly subscribed for $13,99/month. So, the goal is to create a model that can predict the next 3 days, 7 days, 14 days and 28 days of gold price in USD.

## Data Processing

For data processing, was used mean normalization and to guarantee a good amount of data, was used prices from the
regions Us, Eu, China,Korea and Taiwan, and before normalizing the data it was needed a conversion to all of them have
the same base currency, so was obtained from [Investing.com](https://www.investing.com) the historical conversion from current currency
to USD, and matched it with the original price in each region of WoWtoken itself, that lacks historical value but as
was researched hasn't changed over the years.

                             Prices of WoW tokens on Blizzard service.
                             
| Region	  |  Currency  |	WoW token price in currency   |
| --------- | ---------- | ------------------------------ |
| US	      |  USD       |  	        $20.00              |    
| EU	      |  EUR	     |           20.00 €              |
| CHINA	    |  CNY	     |           ￥75.00              |
| KOREA     |  KRW	     |          ￦22,000              |            
| TAIWAN	  |  TWD	     |         NT$500.00              |  
                         



After all processing, the final data was from Gold per WoW token in each region to gold per USD, as was correlated currency
conversion to WoW token price in each region resulting in gold per USD base.


## Training and evaluation of the model

The final model was constructed with 4 lstm layers and as output layer, one dense with output of X units, equal to the
selected window in config.py file. As input, the model get the data in the format [Batch, 1, Lookback] as prediction
being a classification into [1, 0] for increase in value or [0, 1] for decreasing in value, comparing the target value
to last one in Lookback units 3, 7, 14 or 28 units, that is the amount of data that the model receives to
do its prediction.

LSTM layer need it's data to be in window format, a window that moves through the dataset and creates overlapping between
next and previous windows. For that to happen, was chosen that the window moves 1 data per windows, so every data that is
feeded to the model it is in general, 1 unit forward the previous one. As example:

                                      Lookback                     Classification

                                    [1, 2, 3, 4, 5]                 [6]
                                    [2, 3, 4, 5, 6]                 [7]
                                                             .
                                                             .
                                                             .
                                    [95, 96, 97, 98, 99]            [100]

This leads to the input shape [Quantity of sequences, 1, 5].

After defining the model, I got up to training it. Was configured Early stopping to avoid overfitting and save time.

## Result

As result, was generated some range of predictions, they are:

1) All test dataset predicted using true data as training, in a scenario where there is the need to get an insight about it, and then the model can be feeded with the true data to predict the next window. Images on graphs/ with suffix "True Data as lookback".
2) All test dataset predicted using, as the model moves forward predicting, it's own predictions, which leads overall a bigger Accuracy. Images on graphs/ with the suffix "all data".
3) Predictions of only the chosen windows. Images on graphs/ with the suffix "days predicted".

All predicted windows in one image with their own ACCURACY.
# IN CONSTRUCTION!!

Other graphs can be found at [Graphs](https://github.com/flsantna/WoW-token/tree/master/graphs).

## Instructions to use

The project is mainly focused on 3 scripts:

1) [Creation CSV](https://github.com/flsantna/WoW-token/blob/master/creation_of_csv_wowtoken_price.py) - Creates the base CSV file, collecting data on [WoWtokenPrices](https://wowtokenprices.com) and parsing the json to csv while filling all missing values and cropping them.
2) [Model Train](https://github.com/flsantna/WoW-token/blob/master/model_train.py) - Train the model, calling [Data Processing](https://github.com/flsantna/WoW-token/blob/master/data_proc.py) to process the dataset, normalizing its values and if chosen in [config.py](https://github.com/flsantna/WoW-token/blob/master/config.py), convert all bases to gold/usd.
3) [Model Test](https://github.com/flsantna/WoW-token/blob/master/model_test.py) - Test the model and plot all the graphs.

## Used packages and versions

| Package    | Version |
| ---------- | ------ |
| tensorflow | 2.5.0 |
| numpy | 1.19.5 |
| pandas | 1.2.4 |
| matplotlib | 3.4.2 |
