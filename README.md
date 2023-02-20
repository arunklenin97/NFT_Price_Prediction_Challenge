# NFT Price Prediction Challenge
## About
The competition was conducted on [Bitgrit](https://bitgrit.net/competition/17#)
The goal of this competition is to build a model that predicts the price of NFTs based on a select group of their attributes that are publicly accessible and also based on their social media activity.

Note that not all NFTs of a collection are necessarily present in the data.

## To build the machine learning model, the following datasets were provided:

### collections.csv: data about the collections that are present in the training and prediction datasets.
| Column Name             | Description                                 |
| ----------------------- | ------------------------------------------- |
| collection_id             |ID to identify the collection and the NFTs of the collection       |
| total_supply                  | total number of NFTs in the collection                          |
| creation_date| creation date of the collection in the marketplace|
|verification_status| status of verification of the collection in the marketplace|
|n_of_traits|number of particular traits the NFTs in the collection can have|
|contract_type| type of contract in the marketplace|
|seller_fees|fees that the seller receives for transactions|
|platform_fees| fees that the marketplace receives for transactions|
|openrarity_enabled| whether the collection uses OpenRarity to calculate a rarity score and rank for the NFTs in the collection|
|has_website|whether the collection has a website or not|
|has_own_twitter| whether the collection has its own Twitter account or not|
|has_discord| whether the collection has a Discord channel or not|
|has_medium| whether the collection has a Medium account or not|
 
### collections_twitter_stats.csv: simple stats about the Twitter accounts of the collections or the creators of the collections.
| Column Name             | Description                                 |
| ----------------------- | ------------------------------------------- |
|collection_id|ID to identify the collection and the NFTs of the collection|
|n_tweets_in_range| number of tweets in the relevant timeframe|
|avg_likes| average number of likes per tweet|
|avg_replies|average number of replies per tweet|
|avg_retweets| average number of retweets per tweet|
|min_likes| minimum number of likes per tweet|
|min_replies| minimum number of replies per tweet|
|min_retweets| minimum number of retweets per tweet|
|max_likes|maximum number of likes per tweet|
|max_replies| maximum number of replies per tweet|
|max_retweets| maximum number of retweets per tweet|
 
### nfts_train.csv: NFT data from all collections which can be, alongside the above info, used to train the price prediction model.
| Column Name             | Description                                 |
| ----------------------- | ------------------------------------------- |
|global_index|global ID of the NFT to identify it considering all collections|
|nft_id|D of the NFT to identify its particular collection|
|collection_id| ID of the collection the NFT belongs to as per collections.csv file|
|rarity_score|estimated rarity score within the collection|
|openrarity_score| OpenRarity score if it is enabled (see 1)|
|openrarity_rank| OpenRarity rank within the collection if it is enabled (see 1)|
|openrarity_max_rank| OpenRarity maximum rank of the collection if it is enabled (see 1)|
|last_sale_date| approximated date of the last sale of the NFT|
|last_sale_price| price at which the NFT was last sold for (target variable)|
 
### nfts_predict.csv: NFT data from all collections whose price should be predicted using the trained model.
| Column Name             | Description                                 |
| ----------------------- | ------------------------------------------- |
|global_index|global ID of the NFT to identify it considering all collections|
|nft_id|D of the NFT to identify its particular collection|
|collection_id| ID of the collection the NFT belongs to as per collections.csv file|
|rarity_score|estimated rarity score within the collection|
|openrarity_score| OpenRarity score if it is enabled (see 1)|
|openrarity_rank| OpenRarity rank within the collection if it is enabled (see 1)|
|openrarity_max_rank| OpenRarity maximum rank of the collection if it is enabled (see 1)|
|last_sale_date| approximated date of the last sale of the NFT|

Considering that this is a kind of regression problem, the evaluation metric for the score was based on RMSE.
In particular, the formula shall be: exp( -RMSE / 10 ) where 10 is used as a normalization factor. The maximum score thus will be 1.0 and the minimum score will be 0.0.
 

## Solution Approach 

All the given datasets are combined using a left join on "collection_id"

### Handling Missing Values
We can see that for openrarity_rank and openrarity_max_rank, the unique values are less than 10% of the total population and hence can be treated as descrete/categorical. Hence they were converted to ctageroical and introduced a bin for missing.
openrarity_score can be filled with zero as the scores are not available.

### Dropped redundant features with only 1 unique value
'platform_fees', 'has_website', 'has_own_twitter'

### Exploratory Data Analysis
Most of the features are descrete, this leaves the opputunity to treat them as categorical based on performance

* Date Time features: 'last_sale_date', 'creation_date' were converted to datetime
* New Date features: 'sale_year' and 'sale_month' were created. Also, the gap between creation and sale dates were computed. 
* Date features to categorical: I have observed that these sales and creation had happened only on a few dates constituting <1% of total population. Hence converted the date variables to categorical and all dates have multiple records.
* Descrete features: All numerical features with less than 100 unique records(0.02%). However, this method has not improved performance.
* Distribution of Numerical Features: No particular distribution was initially shown but a log transformation has improved a bit. Log Transformation of NFT price has shown a chi-square distribution as expected. Log transformation has been the key for this problem and has improved the model performance significantly.
* Outliers: There were a lot of outliers however I have not used any distance based algoritms. Moreover Tree based algorithms Like Gradient Boosting & XGBoost perform well on most of the cases.
* Categorical Features: Across all the categories for all the variables, the median of NFT price was different indicating that there is a correlation between category and the prices associated with it.

### Feature Engineering: 
* Numerical features: Only log transformation with a correction to include zeros.
* Categorical features: This is where I have gathered all my experiments and applied a bunch of methods. Fianlly used the below methods
- One Hot Encoding: This is applied on 'openrarity_rank'& 'openrarity_max_rank'. Initially rare categories were clubbed, reducing 15556 categories to 2 and 11 to 9 respectively. Had this massive reduction not happened, I would not have used one hot for these features since it would not be parsimonious. 
- Count Based Label Encoding: The method of ranking the distribution of population across the categories. Is one of the best method for features for both high and low number of features. A variant of this would be just a replacement of the category with it's population count.
* New features: I have tried creating few features listed below but none of them have improved performance.
- Total likes: avg_likes*n_tweets_in_range
- Total Retweets: avg_retweets*n_tweets_in_range
- Total Replies: avg_replies*n_tweets_in_range
- Collection Count: collection count across NFT ID
- NFT count: NFT distribution across each collection
- total Collection Count: Population distribution of collection
- total NFT count: Population Distribution of NFTs

### Scaling
Scaled using a standardization
### Modeling Approach
* Basic OLS fit: Just to get an understanding about importance of features through p-value, AIC/BIC. This will help to drop/transform features or try another encoding technique 
* Linear Regression & regularizatiom: As expected, all of them were performing poorly
* Tree based Methods: All methods are tuned to get the best hyper parameters on full set or subset of data based on time of execution
- Decision Trees
- Extra Trees
- Gradient Boosting(has shown the best performance exlcuding stacking ~ 93.26%)
- Light GBM
- Random Forests
- XGBoost
* KNNs: initially this was performig well before log transformation(has given around 86% of acore)

### New Method
#### Panel Regression
The idea is if the entire dataset shows no particular patterns, so The data can be broken down in to sets and fit algoritm that performs well in each set. Ideally this should be done on a feature that has multiple rows for a particular nft_id. However, I have decided to make clusters and then fit algos for each clusters. Clusters were decided using elbow method. 
* Two approaches of Panel Regression:
- Generic Panel Regression: A function is created to fit any algorithm but parameters are same for all the subsets.
- Tuned Panel Regression: Here along with the above step, for each subset the hyper paramters are tuned(takes more time to execute)
### Model Stacking: 
Stacking is one of the best method to improve your performance in datascience competitions. Many of the times, the positions change after private leaderboard is revealed. One who has fit a generic method that works well on the entire spacial distribution of data moves up the board. A single algorithm many not perform well on the entirity of the data however a stacked model with multiple algoritms performs better.

### Output Stacking:
Many submissions that have performed well(>93%) have been stacked on multiple levels by Geometric Mean, Simple Mean etc. 

The final output has shown a score of 93.299%
outputs from panel regressions are stacked by simple average.
