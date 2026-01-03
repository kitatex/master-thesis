# Data  

## Bluesky Data Format
todo
maybe just reference to zenodo link


## General Querying Pipeline

todo / mention:
more detailed instructions in the .py files at the top
how did i choose my discussion topics?


### Post corpus

Process for obtaining data for a *discussion topic* (e.g., climate change, football) based on a *seed hashtag*:

- **Step 1**: For a discussion topic of your choice, select a seed hashtag that clearly represents the discussion topic and around which the data set will be built. You can use the hashtag counts list in ``data/general/`` to pick a seed hashtag of your choice.

- **Step 2**: Use [``copy_keyword_posts.py``](../src/query/copy_keyword_posts.py) to obtain an initial corpus of posts which contain that hashtag.

- **Step 3**: Deduplicate posts using ``deduplicate_posts.py`` because the original data contains duplicates.

- **Step 4**: Create a ranked list of the closest hashtags to the seed hashtags using [``related_hashtags.py``](../src/tf_idf/related_hashtags.py). The closest hashtags are ranked using tf-idf scores, so that hashtags which are generally unpopular but co-occur highly with the seed hashtag get a higher score (and vice-versa).

- **Step 5**: From the ranked list from Step 4, manually select the top $k$ hashtags that will later be used to obtain posts for the discussion topic based on related keywords with tf-idf. It is important to omit hashtags which do not represent the issue well enough according to predefined criteria.

- **Step 6**: Use [``copy_keyword_posts.py``](../src/query/copy_keyword_posts.py) to obtain an initial corpus of posts which contain the co-hashtags selected from Step 5.

todo:
- **Step 7**: obtain keywords that represent seed hashtag and co-hashtags (important to also capture posts that don't use hashtags)

- **Step 8**: query posts that contain keywords (define a logic. e.g., have to contain at least keywords?? or statistical technique)

### Graph

- **Step x**: obtain structural data based on the post corpus to construct a network (first, i think just work with retweets. Either a network with weighted edges (how many retweets or log(retweets) as weights) OR min 2 retweets to form an edge; compare Garimella)




## Documentation Data Querying Thesis
todo

Documents the choices made during the data querying process for my master thesis.



### Discussion Topics & Seed Hashtags (Step 1)
todo
criteria for manually choosing seed hashtags:
- topic fits the definition of social issue
- not generic / too broad (e.g., #politics, #science)
- seed hashtags represents the political discussion around that topic
end

First, 5 discussion topics are selected that represent a *political issue*. 

Political issues:
- *black people's rights*: based on #listentoblackvoices with 30482 occurrences in the data (rank 82)

- *war in gaza*: based on #freepalestine with 19026 occurrences in the data (rank 155)

- *war in ukraine*: based on #ukrainianview with 14417 occurrences in the data (rank 213)

- *climate change*: based on #climatecrisis with 6504 occurrences in the data (rank 519)
    - Current Step: 6 (30-12-2025)

- *trans rights*: based on #climatecrisis with 3235 occurrences in the data (rank 1175)

Non-political Issues:
- s



### Co-Hashtag Selection Criteria (Step 5)

k = 50 

Criteria for filtering out co-hashtags (idea: use the same ones for step 1?):
- hashtag must be in English
- hashtag must not be applicable to political discussion in general (e.g., #politics or #news)
- hashtag must not be applicable to other political issues (e.g., #capitalism correlates with #climatecrisis, but can also be used when discussing wealth inequality)
- hashtag must represent the issue from the political perspective (e.g., #canada or #nature strongly correlate with #climatecrisis, but can also represent discussion around the issue form a non-political angle, so it is ommited)


### Related Keywords






