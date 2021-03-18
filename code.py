#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis on Airline Tweets
# Cheng Zhong <br>
# cheng.zhong@columbia.edu

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import swifter
import gc
import nltk 
import sklearn 
import collections
import sys
import itertools
import string
import re
import emoji

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tqdm.notebook import tqdm
from nltk.stem import PorterStemmer

import pickle5 as pickle
import sklearn 
from collections import Counter
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from wordcloud import WordCloud, ImageColorGenerator
from tqdm.notebook import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv("tweets.csv")
df.head()


# In[3]:


# Number of missing complaint narratives
no_miss_narr = df['text'].isnull().sum()
pct_miss_narr = no_miss_narr/len(df.index) * 100
print(f"Number of missing texts without narratives: {no_miss_narr}")
print(f"Percentage of all texts without narratives: {pct_miss_narr:.2f}%")


# ### Distribution of Sentiments

# In[4]:


sns.factorplot(x="airline_sentiment", data=df, kind="count", size=6, aspect=1.5, palette="PuBuGn_d")
plt.show();


# ### Text Cleaning

# In[6]:


class TextCounts(BaseEstimator, TransformerMixin):
    
    def count_regex(self, pattern, tweet):
        return len(re.findall(pattern, tweet))
    
    def fit(self, X, y=None, **fit_params):
        # fit method is used when specific operations need to be done on the train data, but not on the test data
        return self
    
    def transform(self, X, **transform_params):
        count_words = X.apply(lambda x: self.count_regex(r'\w+', x)) 
        count_mentions = X.apply(lambda x: self.count_regex(r'@\w+', x))
        count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+', x))
        count_capital_words = X.apply(lambda x: self.count_regex(r'\b[A-Z]{2,}\b', x))
        count_excl_quest_marks = X.apply(lambda x: self.count_regex(r'!|\?', x))
        count_urls = X.apply(lambda x: self.count_regex(r'http.?://[^\s]+[\s]?', x))
        # We will replace the emoji symbols with a description, which makes using a regex for counting easier
        # Moreover, it will result in having more words in the tweet
        count_emojis = X.apply(lambda x: emoji.demojize(x)).apply(lambda x: self.count_regex(r':[a-z_&]+:', x))
        
        df = pd.DataFrame({'count_words': count_words
                           , 'count_mentions': count_mentions
                           , 'count_hashtags': count_hashtags
                           , 'count_capital_words': count_capital_words
                           , 'count_excl_quest_marks': count_excl_quest_marks
                           , 'count_urls': count_urls
                           , 'count_emojis': count_emojis
                          })
        
        return df
tc = TextCounts()
df_eda = tc.fit_transform(df.text)
df_eda['airline_sentiment'] = df.airline_sentiment
df_eda.head()


# In[7]:


class CleanText(BaseEstimator, TransformerMixin):
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)
    
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
    
    def emoji_oneword(self, input_text):
        # By compressing the underscore, the emoji is kept as one word
        return input_text.replace('_','')
    
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)
    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    def stemming(self, input_text):
        porter = PorterStemmer()
        words = input_text.split() 
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.emoji_oneword).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)
        return clean_X


# In[8]:


ct = CleanText()
sr_clean = ct.fit_transform(df.text)
sr_clean.sample(10)


# In[9]:


empty_clean = sr_clean == ''
print('{} records have no words left after text cleaning'.format(sr_clean[empty_clean].count()))
sr_clean.loc[empty_clean] = '[no_text]'


# In[10]:


sr_clean.head()


# ### Most Frequent Words (Top 20)

# In[11]:


cv = CountVectorizer()
bow = cv.fit_transform(sr_clean)
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(12, 10))
sns.barplot(x="word", y="freq", data=word_counter_df, ax=ax)
plt.show()


# ### Model 1: Sentiment Analysis w/ EmoLex and TfidfVectorizer

# In[35]:


content=sr_clean
df_model1=pd.DataFrame({'text': content})
df_model1.reset_index(drop=True, inplace=True)
df_model2 = df_model1
df_model1.head()


# In[25]:


#df_model1['tokenized_text'] = df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)


# In[26]:


#df_model1.head()


# In[27]:


from nrclex import NRCLex 
import glob
vec = CountVectorizer()


# In[28]:


filepath = "NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], skiprows=45, sep='\t', keep_default_na=False)
emolex_df = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
emolex_df.head()


# In[29]:


vec = TfidfVectorizer(vocabulary=emolex_df.word,
                      use_idf=False, 
                      norm='l1')
matrix = vec.fit_transform(df_model1.text)
vocab = vec.get_feature_names()
wordcount_df = pd.DataFrame(matrix.toarray(), columns=vocab)
wordcount_df


# In[30]:


negative_words = emolex_df[emolex_df.negative == 1].word

df_model1['negative'] = wordcount_df[negative_words].sum(axis=1)
df_model1.head(5)


# In[31]:


positive_words = emolex_df[emolex_df.positive == 1].word

df_model1['positive'] = wordcount_df[positive_words].sum(axis=1)
df_model1.head(10)


# In[32]:


angry_words = emolex_df[emolex_df.anger == 1]['word']
df_model1['anger'] = wordcount_df[angry_words].sum(axis=1)
df_model1.head(10)


# In[33]:


pd.options.mode.chained_assignment = None
df_model1.plot(x='positive', y='negative', kind='scatter')


# ### Model 2: Sentiment Analysis before Tokenization with TextBlob

# In[34]:


import re
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


# In[46]:


df_model2 = df


# In[47]:


df_model2['sentiment'] = df_model2['text'].apply(lambda narrative: TextBlob(narrative).sentiment)

df_model2['polarity'] = df_model2['sentiment'].apply(lambda sentiment: sentiment[0])
df_model2['subjectivity'] = df_model2['sentiment'].apply(lambda sentiment: sentiment[1])

df_model2.head()


# In[48]:


df_model2['positive'] = df_model2['polarity'].apply(lambda polarity: polarity>0)
df_model2['neutral'] = df_model2['polarity'].apply(lambda polarity: polarity==0)
df_model2['negative'] = df_model2['polarity'].apply(lambda polarity: polarity<0)

df_model2.head()


# In[49]:


print("Positive comments number: {}".format(df_model2['positive'].sum()))
print("Positive comments percentage: {} %".format(100*df_model2['positive'].sum()/df_model2['text'].count()))
print("Neutral comments number: {}".format(df_model2['neutral'].sum()))
print("Neutral comments percentage: {} %".format(100*df_model2['neutral'].sum()/df_model2['text'].count()))
print("Negative comments number: {}".format(df_model2['negative'].sum()))
print("Negative comments percentage: {} %".format(100*df_model2['negative'].sum()/df_model2['text'].count()))


# In[50]:


plt.figure(figsize=(50,30))
plt.margins(0.02)
plt.xlabel('Sentiment', fontsize=50)
plt.xticks(fontsize=40)
plt.ylabel('Frequency', fontsize=50)
plt.yticks(fontsize=40)
plt.hist(df_model2['polarity'], bins=50)
plt.title('Sentiment Distribution', fontsize=60)
plt.show()


# In[51]:


polarity_avg = df_model2.groupby('airline')['polarity'].mean().sort_values(ascending=False).plot(kind='bar', figsize=(50,30))
plt.xlabel('Airline', fontsize=45)
plt.ylabel('Average Sentiment Score', fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('Average Sentiment Score per Airline', fontsize=50)
plt.show()


# ### Model 3: MultinomailNB and Logistic Regression w/ CountVectorizer and TF-IDF Classifiers

# In[52]:


df_model = df_eda
df_model['clean_text'] = sr_clean
df_model.columns.tolist()


# In[53]:


class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X, **transform_params):
        return X[self.cols]

    def fit(self, X, y=None, **fit_params):
        return self


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(df_model.drop('airline_sentiment', axis=1), df_model.airline_sentiment, test_size=0.1, random_state=37)


# In[63]:


def grid_vect(clf, parameters_clf, X_train, X_test, parameters_text=None, vect=None, is_w2v=False):
    
    textcountscols = ['count_capital_words','count_emojis','count_excl_quest_marks','count_hashtags'
                      ,'count_mentions','count_urls','count_words']
    
    if is_w2v:
        w2vcols = []
        for i in range(SIZE):
            w2vcols.append(i)
        features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))
                                 , ('w2v', ColumnExtractor(cols=w2vcols))]
                                , n_jobs=-1)
    else:
        features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))
                                 , ('pipe', Pipeline([('cleantext', ColumnExtractor(cols='clean_text')), ('vect', vect)]))]
                                , n_jobs=-1)

    
    pipeline = Pipeline([
        ('features', features)
        , ('clf', clf)
    ])
    
    # Join the parameters dictionaries together
    parameters = dict()
    if parameters_text:
        parameters.update(parameters_text)
    parameters.update(parameters_clf)

    # Make sure you have scikit-learn version 0.19 or higher to use multiple scoring metrics
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)

    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best CV score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
    print("Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))
    print("\n")
    print("Classification Report Test Data")
    print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))
                        
    return grid_search


# In[56]:


# Parameter grid settings for the vectorizers (Count and TFIDF)
parameters_vect = {
    'features__pipe__vect__max_df': (0.25, 0.5, 0.75),
    'features__pipe__vect__ngram_range': ((1, 1), (1, 2)),
    'features__pipe__vect__min_df': (1,2)
}

# Parameter grid settings for MultinomialNB
parameters_mnb = {
    'clf__alpha': (0.25, 0.5, 0.75)
}

# Parameter grid settings for LogisticRegression
parameters_logreg = {
    'clf__C': (0.25, 0.5, 1.0),
    'clf__penalty': ('l1', 'l2')
}


# In[75]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib
from time import time
mnb = MultinomialNB()
logreg = LogisticRegression()


# In[76]:


countvect = CountVectorizer()
import os
np.random.seed(37)
KAGGLE_ENV = os.getcwd == '/kaggle/working'


# In[78]:


# MultinomialNB
best_mnb_countvect = grid_vect(mnb, parameters_mnb, X_train, X_test, parameters_text=parameters_vect, vect=countvect)


# In[79]:


# LogisticRegression
best_logreg_countvect = grid_vect(logreg, parameters_logreg, X_train, X_test, parameters_text=parameters_vect, vect=countvect)


# In[80]:


tfidfvect = TfidfVectorizer()


# In[81]:


# MultinomialNB
best_mnb_tfidf = grid_vect(mnb, parameters_mnb, X_train, X_test, parameters_text=parameters_vect, vect=tfidfvect)


# In[82]:


# LogisticRegression
best_logreg_tfidf = grid_vect(logreg, parameters_logreg, X_train, X_test, parameters_text=parameters_vect, vect=tfidfvect)


# ### Predict future negatives tweets

# In[84]:


textcountscols = ['count_capital_words','count_emojis','count_excl_quest_marks','count_hashtags'
                      ,'count_mentions','count_urls','count_words']
    
features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))
                         , ('pipe', Pipeline([('cleantext', ColumnExtractor(cols='clean_text'))
                                              , ('vect', CountVectorizer(max_df=0.5, min_df=1, ngram_range=(1,2)))]))]
                       , n_jobs=-1)

pipeline = Pipeline([
    ('features', features)
    , ('clf', LogisticRegression(C=1.0, penalty='l2'))
])

best_model = pipeline.fit(df_model.drop('airline_sentiment', axis=1), df_model.airline_sentiment)


# In[85]:


new_negative_tweets = pd.Series(["@VirginAmerica shocked my initially with the service, but then went on to shock me further with no response to what my complaint was. #unacceptable @Delta @richardbranson"
                      ,"@VirginAmerica this morning I was forced to repack a suitcase w a medical device because it was barely overweight - wasn't even given an option to pay extra. My spouses suitcase then burst at the seam with the added device and had to be taped shut. Awful experience so far!"
                      ,"Board airplane home. Computer issue. Get off plane, traverse airport to gate on opp side. Get on new plane hour later. Plane too heavy. 8 volunteers get off plane. Ohhh the adventure of travel ✈️ @VirginAmerica"])

df_counts_neg = tc.transform(new_negative_tweets)
df_clean_neg = ct.transform(new_negative_tweets)
df_model_neg = df_counts_neg
df_model_neg['clean_text'] = df_clean_neg

best_model.predict(df_model_neg).tolist()

