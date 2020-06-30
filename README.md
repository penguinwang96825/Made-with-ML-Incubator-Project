# Made with ML Incubator Project
In this project, we focus on the researches applying natural language processing (NLP) technologies in the finance domain. We scrape tweets for forex traders then apply sentiment analysis to generate investment insight.

Project [link](https://madewithml.com/projects/1390/nlp-oriented-finance-analyzer/).

## Introduction
Sentiment analysis can use natural language processing, artificial intelligence, text analysis and computational linguistics to identify the attitude of several topics. In this project, we focus on the researches applying natural language processing (NLP) technologies in the finance domain. First, we will dig into some people who have huge impact on financial market. Second, we will predict foreign exchange rates by making use of the trending topics from Twitter, using a machine learning based model.

## Collect Data

### Collect Tweets Data
Forex trading is fast, very fast, and Twitter fits like a glove to any forex trader’s hand. There’s lots of quick and useful information coming in the form of tweets and sometimes too much information. We got a list of top forex twitter accounts from [here](https://www.forexcrunch.com/60-top-forex-twitter-accounts/), each one coming with different characteristics, to suit traders interested in different aspects of trading (technical, fundamental, educational,, sentiment, a mix of some or all, etc.). We Crawled 63 forex twitter accounts listed on the website and store it into `trader_account` list for future use. Here is our [notebook](https://github.com/penguinwang96825/Made-with-ML-Incubator-Project/blob/master/notebook/twint.ipynb).

[Twint](https://github.com/twintproject/twint) is an advanced Twitter scraping tool written in Python that allows for scraping Tweets from Twitter profiles without using Twitter's API. We utilise twint to get tweets, and store the results into a pandas dataframe. We created a simple function that you can see in the actual project that integrate Pandas with Twint API for this part. Next, there are many features we have from the query we just did. There’s a lot of different things to do with this data, but for this project we’ll only use some of them, namely `date`, `time`, `username`, `tweet`, `hashtags`, `likes_count`, `replies_count`, and `retweets_count`.

### Collect Forex Data
We downloaded the forex data from Mecklai Financial. After pre-processing, we get a new column "label", which means the differentiation between two days. Label {0, 1} is the forex movement label telling whether the forex trade price is up or down after a certain time. For this study, we used only GBP/USD price from the Macrotrends website. Macrotrends provides lots of foreign exchange data, such as EUR/USD, USD/JPY, USD/CNY, AUD/USD, EUR/GBP, USD/CHF, EUR/CHF, GBP/JPY and EUR/JPY. Moreover, they also illustrate interactive historical chart showing the daily forex price.

### Combine Tweets and Forex
The forex movement prediction task can be defined as assigning movement label for the tweets input. The forex prediction is conducted as a binary classification task (up or down). The evaluation metrics are F1 and Matthews Correlation Coefficient (MCC). MCC is often reported in stock movement forecast (Xu and Cohen, 2018; Ding et al., 2016) because it can overcome the data imbalance issue.

## Modelling
![](https://github.com/penguinwang96825/Made-with-ML-Incubator-Project/blob/master/image/model%20structure.png?raw=true)

The overview of our model is displayed above. The model can be generally devided into three steps:
1. Tweets ranking.
2. Tweets pre-processing.
3. Inter-groups aggregation.

In step 1, we conduct zero-shot learning on this paragraph to select the most important tweets on a daily basis. Tweets are then ranked by latent embedding approach, which is a common approach to zero shot learning in the computer vision setting. In the text domain, we have the advantage that we can trivially use a single model to embed both the data and the class names into the same space, eliminating the need for the data-hungry alignment step. We therefore decided to run some experiments with Sentence-BERT, a recent technique which fine-tunes the pooled BERT sequence representations for increased semantic richness, as a method for obtaining sequence and label embeddings. Here is our [notebook](https://github.com/penguinwang96825/Made-with-ML-Incubator-Project/blob/master/notebook/Zero-shot%20Learning.ipynb).

In step 2, we conduct some text pre-processing work. Here is our [notebook](https://github.com/penguinwang96825/Made-with-ML-Incubator-Project/blob/master/notebook/Tweet%20Preprocessing.ipynb).
```python
class TweetsPreprocessor:
    
    def __init__(self, contractions_dict, lower=True):
        self.contractions_dict = contractions_dict
        self.lower = lower
        
    def remove_unicode(self, text):
        """ Removes unicode strings like "\u002c" and "x96" """
        text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
        text = re.sub(r'[^\x00-\x7f]',r'',text)
        return text

    def replace_URL(self, text):
        """ Replaces url address with "url" """
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',text)
        text = re.sub(r'#([^\s]+)', r'\1', text)
        return text

    def replace_at_user(self, text):
        """ Replaces "@user" with "atUser" """
        text = re.sub('@[^\s]+','atUser',text)
        return text

    def remove_hashtag_in_front_of_word(self, text):
        """ Removes hastag in front of a word """
        text = re.sub(r'#([^\s]+)', r'\1', text)
        return text

    # Function for expanding contractions
    def expand_contractions(self, text, contractions_dict=contractions_dict):
        # Regular expression for finding contractions
        contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, text)

    def ultimate_clean(self, text):
        if self.lower:
            text = text.lower()
        text = self.remove_unicode(text)
        text = self.replace_URL(text)
        text = self.replace_at_user(text)
        text = self.remove_hashtag_in_front_of_word(text)
        text = self.expand_contractions(text)
        return text
```

In step 3, we combine top k daily tweets in order to aggregate semantic information at the inter-groups level. Here is our [notebook](https://github.com/penguinwang96825/Made-with-ML-Incubator-Project/blob/master/notebook/BERT%20Aggregate%20Model.ipynb).

### Experiment Setting
We choose the transformers from HuggingFace as implement and choose the bert-base-uncased version. We truncate the BERT input to 64 tokens and fine-tune the BERT parameters during training. We adopt the Adam optimizer with the initial learning rate of 2e-5. We apply the dropout regularization with the dropout probability of 0.25 to reduce over-fitting. The batch size is 32. The training epoch is 4. The weight of L2 regularization is 0.1. When splitting the dataset, we guarantee that the samples in train set are previous to samples in valid set and test set to avoid the possible information leakage. The forex prediction is conducted as a binary classification task (up or down). The evaluation metrics are F1 and Matthews Correlation Coefficient (MCC). MCC is often reported in stock movement forecast because it can deal with the data imbalance problem.

### Result
![result](https://github.com/penguinwang96825/Made-with-ML-Incubator-Project/blob/master/image/result.png?raw=true)
![cm](https://github.com/penguinwang96825/Made-with-ML-Incubator-Project/blob/master/image/confusion%20matrix.png?raw=true)