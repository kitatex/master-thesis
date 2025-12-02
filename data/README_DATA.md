# Data  

## Bluesky Data Format

todo


## Querying Pipeline

Process for obtaining data (users, posts, ...) for a discussion topic based on a seed hashtag:

1. For a discussion topic of your choice, select a seed hashtag that represents the discussion topic and around which the data set will be built. You can use the hashtag counts list in ``data/general/`` to pick a seed hashtag of your choice.

2. Use ``copy_keyword_posts.py`` to obtain an initial corpus of data of that hashtag.

3. Deduplicate posts using ``deduplicate_posts.py`` because the original data contains duplicates already.

4. Create a ranked list of the closest hashtags to the seed hashtags using [``related_hashtags.py``](../src/tf_idf/related_hashtags.py). The closest hashtags are ranked using tf-idf scores, so that hashtags which are generally unpopular but co-occur highly with the seed hashtag get a higher score (and vice-versa). 

5. From the ranked list from step 4, manually select the top $k$ hashtags that will later be used to obtain posts for the discussion topic based on related keywords with tf-idf. For the thesis, the choice for $k$ is 50 and I ommit hashtags which are not in English and that are too broad (not necessarily about the discussion topic). The chosen top $k$ hashtags should be 
todo





