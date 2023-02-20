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
* Descrete features: All numerical features with less than 100 unique records(0.02%). 
