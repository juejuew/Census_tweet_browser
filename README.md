# Census_tweet_browser

Files: 
- allCensus_sample.csv: Random sample (0.5%) of all 'Census' tweets provided by Sprinklr query.
- app_with_advanced_options.py: Tweet browser in Dash. Performs dimension reduction and clustering on document-word matrix to obtain cluster labels for each tweet. 
- tweet_browser_not_dash.py: Contains functions used to the tweet browser, but not written as a Dash app. When adding new features to tweet browser, I find it easier to write, debug, and test the function in this script first, and then copy over to the Dash app.
- trust_condensed_sample_smaller.csv: Example csv of tweets to test tweet browser. Contains a sample of 1018 tweets from the Census query that contain the string 'trust'.
- distance_quote.py: Takes a quote (e.g. from a focus group) and returns tweets 'close' to that quote as determined by LDA, LSA, and a new distance metric that uses co-occurrence of words to create distances between texts.
