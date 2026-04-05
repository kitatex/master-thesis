# Data  

### Source

The data is an open dataset which can be accessed [here](https://zenodo.org/records/14669616).


# General Querying Pipeline


### Post corpus

Process for obtaining data for a *discussion topic* (e.g., climate change, football) based on a *seed hashtag*:

- **Step 1**: For a discussion topic of your choice, select a seed hashtag that clearly represents the discussion topic and around which the data set will be built. You can use the hashtag counts list in ``data/general/`` to pick a seed hashtag of your choice.

- **Step 2**: Use [``copy_keyword_posts.py``](../data_processing/query/copy_keyword_posts.py) to obtain an initial corpus of posts which contain that hashtag.

- **Step 3**: Deduplicate posts using [``deduplicate_posts.py``](../data_processing/format/deduplicate_posts.py) because the original data contains duplicates.

- **Step 4**: Create a ranked list of the closest hashtags to the seed hashtags using [``related_hashtags.py``](../data_processing/tf_idf/related_hashtags.py). The closest hashtags are ranked using tf-idf scores, so that hashtags which are generally unpopular but co-occur highly with the seed hashtag get a higher score (and vice-versa).

- **Step 5**: From the ranked list from Step 4, manually select the top $k$ hashtags that will later be used to obtain posts for the discussion topic based on related keywords with tf-idf. It is important to omit hashtags which do not represent the issue well enough according to predefined criteria.
Do this by copying the output file (e.g., ``closest_hashtags.jsonl``) and deleting the first and last rows (the brackets) and all hashtags which are not relevant. Then. convert to .txt.
For this project, $k=20$ was chosen since Garimella et al. (2018) did so similarly.

- **Step 6**: Use [``copy_keyword_posts.py``](../data_processing/query/copy_keyword_posts.py) to obtain an initial corpus of posts which contain the co-hashtags selected from Step 5.

todo:
- **Step 7**: obtain keywords that represent seed hashtag and co-hashtags (important to also capture posts that don't use hashtags)

- **Step 8**: query posts that contain keywords (define a logic. e.g., have to contain at least keywords?? or statistical technique)


### Graph Building

- **Step x**: obtain structural data based on the post corpus to construct a network (first, i think just work with retweets. Either a network with weighted edges (how many retweets or log(retweets) as weights) OR min 2 retweets to form an edge; compare Garimella)




# Documentation Data Querying Thesis
todo

Documents the choices made during the data querying process for my master thesis.



### Discussion Topics & Seed Hashtags (Step 1)

#### Political Issues

todo
Criteria for manually choosing seed hashtags:
- topic fits the definition of a political issue
- not generic / too broad (e.g., #politics, #science)
- seed hashtags represents the political discussion around that topic
end
- idea: topic has to allow controversy / different stances
- maybe: search for discussion topics in the first place and then for a hashtag representation
- not predominantly photo-based

Selected discussion topics / seed hashtags:

- *war in ukraine*: based on #ukraine with 19369 occurrences in the data (rank 77)

- *war in gaza*: based on #gaza with 13435 occurrences in the data (119)

- *climate change*: based on #climatechange with 5288 occurrences in the data (rank 365)

- *AI ethics*: based on #aiethics with 769 occurrences in the data.


#### Non-political Issues

Criteria for manually choosing seed hashtags:
- not predominantly photo-based
- not political 

Selected discussion topics / seed hashtags:

- "#gamedev": 22466 - Game development

- "#historicalfiction": 13152 - Book discussion

- "#musicsky": 5229 - Music discussion

- "#dadjokes": 772 - Humor


ideas:
cats




### Co-Hashtag Selection Criteria (Step 5)

k = 50 

Criteria for filtering out co-hashtags (idea: use the same ones for step 1?):
- hashtag must be in English
- hashtag must not be applicable to political discussion in general (e.g., #politics or #news)
- hashtag must not be applicable to other political issues (e.g., #capitalism correlates with #climatecrisis, but can also be used when discussing wealth inequality)
- hashtag must represent the issue from the political perspective (e.g., #canada or #nature strongly correlate with #climatecrisis, but can also represent discussion around the issue form a non-political angle, so it is ommited)


### Related Keywords






