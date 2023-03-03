# Tesla_Tweets_Sentiment

# Subject Background
In the past, sentiment analysis has been used in the financial domain to predict stock prices. However, previous
approaches have mostly focused on traditional statistical models, such as regression and time series analysis, and have
limited the scope of the analysis to a small number of features. These approaches also require significant manual feature
engineering and often lack the ability to capture the context and nuances of language.
Recently, with the advancements in deep learning techniques, specifically the development of the Long Short-Term
Memory (LSTM) model, which is a type of recurrent neural network, researchers have demonstrated the effectiveness
of applying these techniques to the field of stock market prediction using social media data. The LSTM model has shown
to be particularly effective in capturing the temporal dependencies and contextual information of the data, which are
crucial for stock market prediction. Therefore, applying data science techniques such as LSTM model to analyze Twitter
data is a promising approach for stock market prediction, as it can provide more accurate and relevant insights into the
sentiment of users towards companies.

# Details on dataset
There are two data sources: market price history and tweets from Twitter. We obtained daily price history in CSV format
from Yahoo Finance, covering the period from January 1, 2020, to December 31, 2022. The data includes open, close,
high, and low prices, as well as volume. To collect tweet data, we used the Twitter API with the Tweepy package in
Python. The API offers two versions with different authentication services. To extract useful features from Twitter, we
defined a search query that included the keyword #tesla and excluded all media, links, and retweeted content. We utilized
the API's research accessibility tier, which provided full access to Twitter's archive. The requested features and metrics
from the API include the tweet's unique ID, author ID, author's number of followers, number of likes, status of the post
(if retweeted), and whether the post contains the cointag $TSLA. Additionally, the API provides information on whether
the author has a verified account.

# Summary of cleaning and preprocessing
## 1) Cleaning
The price data sourced from Yahoo Finance has already been cleaned. As for the tweets data, we are utilizing an API
that enables us to ensure the integrity and purity of the data. For instance, we can filter out any undesired contents and2
features by defining a query. Each tweet has a unique ID, which ensures that duplication is not possible. In case a tweet
is deleted or there is missing data, we can simply ignore it and prevent any null values from being generated.
## 2) Preprocessing, data wrangling and feature extraction
To prepare our data for modeling, we must first wrangle and transform it, and extract new features if necessary. One
such feature is tweet sentiment, which we can extract by performing sentiment analysis on the text of each tweet. This
analysis yields a sentiment polarity value for each tweet, categorized as negative, neutral, or positive. However, before
we can perform this analysis, we must first prepare and clean the text by replacing emojis with their definitions, removing
stop words, punctuation, and special characters, and applying stemming and lemmatizing techniques. Once the text is
ready, we use RoBERTa, a robustly optimized BERT pretraining approach. This model was trained on approximately 58
million tweets and fine-tuned for sentiment analysis with the TweetEval benchmark.
Each tweet on Twitter carries a unique social impression that varies depending on its level of engagement. Accounts
with a large following or tweets that have been retweeted multiple times and received numerous likes hold more weight
on the platform. To ensure accuracy in our final aggregation, we must consider each tweet's impression factor. There are
various ways to define this factor, but our approach is as follows:
𝐼𝑚𝑝𝑟𝑒𝑠𝑠𝑖𝑜𝑛 𝐹𝑎𝑐𝑡𝑜𝑟 = log (𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑓𝑜𝑙𝑙𝑜𝑤𝑒𝑟𝑠 + 1) + log (𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑙𝑖𝑘𝑒𝑠 + 1) + (0.5 ∗ 𝑖𝑠_𝑟𝑒𝑡𝑤𝑒𝑒𝑡𝑒𝑑 + 0.5 ∗
ℎ𝑎𝑠_𝑐𝑜𝑖𝑛𝑡𝑎𝑔𝑇𝑆𝐿𝐴 + 𝑖𝑠_(𝑎𝑐𝑐𝑜𝑢𝑛𝑡_𝑣𝑒𝑟𝑖𝑓𝑖𝑒𝑑 ) )
We have the ability to generate a feature known as the "price action feature" by utilizing market price data. This feature
will provide insights into the market trend for a particular day. For example, if the closing price is higher than the opening
price, and the difference accounts for 75% of the difference between the high and low prices, then the market behavior
for that day can be considered as "bullish." Conversely, if the closing price is lower than the opening price and the
difference represents 75% of the high and low differences, then the market is deemed "bearish." If none of these
conditions are met, then it is referred to as "consolidation." This terminology is similar to that of engulfing candles.

# Modeling
After extracting the desired features, we perform daily aggregation on tweets and merge the resulting data with price
data. The shape of the data needs to be reshaped to be suitable for the model's input. Since the features have different
ranges, scaling the data is necessary before feeding it to the model. The data model architecture consists of two LSTM
layers with dropout regularization, followed by two dense layers. The best model is proposed based on hyperparameter
optimization.
