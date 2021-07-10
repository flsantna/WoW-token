##Prediction of WoWtoken price in gold using LSTM networks.

WoWtoken is an item that can be purchased on World of Warcraft store and be sold in game by it's currency, that is gold
and can be used to redeem gameplay time as the game is monthly subscribed for $13,99/month.
So, the goal is create an model that can predict the next 3 days, 7 days, 14 days and 28 days of gold price in USD.

##Data Processing

For data processing, was used mean normalization and to guarantee a good amount of data, was used prices from the
regions Us, Eu, China,Korea and Taiwan, and before normalizing the data it was needed a conversion to all of them have
the same base currency, so was obtained from https://www.investing.com the historical conversion from current currency
to USD, and matched it with the original price in each region of WoWtoken it self, that lacks historical value but as
was researched, haven't changed over the years.

                             Prices of WoW token on Blizzard service.
                        Region	  |  Currency  |	WoW token price in Currency  
                                  |            |                                
                        US	      |  USD       |  	        $20.00                  
                        EU	      |  EUR	     |           20.00 €               
                        CHINA	    |  CNY	     |           ￥75.00  
                        KOREA	    |  KRW	     |          ￦22,000               
                        TAIWAN	  |  TWD	     |         NT$500.00                

After all processing, the final data was from Gold per WoW token in each region to gold per USD, as was correlated currency
conversion to WoW token price in each region resulting them to an gold per USD base.


##Training and evaluation of the model

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

##Result


