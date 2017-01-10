
# coding: utf-8

# # Predicting sentiment from product reviews
# 
# # Fire up GraphLab Create
# (See [Getting Started with SFrames](/notebooks/Week%201/Getting%20Started%20with%20SFrames.ipynb) for setup instructions)

# In[1]:

import graphlab


# In[2]:

# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)


# # Read some product review data
# 
# Loading reviews for a set of baby products. 

# In[3]:

products = graphlab.SFrame('amazon_baby.gl/')


# # Let's explore this data together
# 
# Data includes the product name, the review text and the rating of the review. 

# In[4]:

products.head()


# # Build the word count vector for each review

# In[5]:

products['word_count'] = graphlab.text_analytics.count_words(products['review'])


# In[6]:

products.head()


# In[7]:

graphlab.canvas.set_target('ipynb')


# In[8]:

products['name'].show()


# # Examining the reviews for most-sold product:  'Vulli Sophie the Giraffe Teether'

# In[9]:

giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']


# In[10]:

len(giraffe_reviews)


# In[11]:

giraffe_reviews['rating'].show(view='Categorical')


# # Build a sentiment classifier

# In[12]:

products['rating'].show(view='Categorical')


# ## Define what's a positive and a negative sentiment
# 
# We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.  Reviews with a rating of 4 or higher will be considered positive, while the ones with rating of 2 or lower will have a negative sentiment.   

# In[13]:

# ignore all 3* reviews
products = products[products['rating'] != 3]


# In[14]:

# positive sentiment = 4* or 5* reviews
products['sentiment'] = products['rating'] >=4


# In[15]:

products.head()


# ## Let's train the sentiment classifier

# In[16]:

train_data,test_data = products.random_split(.8, seed=0)


# In[56]:

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=['word_count'],
                                                     validation_set=test_data)


# # Evaluate the sentiment model

# In[18]:

sentiment_model.evaluate(test_data, metric='roc_curve')


# In[19]:

sentiment_model.show(view='Evaluation')


# # Applying the learned model to understand sentiment for Giraffe

# In[20]:

giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')


# In[21]:

giraffe_reviews.head()


# ## Sort the reviews based on the predicted sentiment and explore

# In[22]:

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)


# In[23]:

giraffe_reviews.head()


# ## Most positive reviews for the giraffe

# In[24]:

giraffe_reviews[0]['review']


# In[25]:

giraffe_reviews[1]['review']


# ## Show most negative reviews for giraffe

# In[26]:

giraffe_reviews[-1]['review']


# In[27]:

giraffe_reviews[-2]['review']


# In[28]:

selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']


# In[33]:

def select_words(words_count):
    selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']
    d = {}
    for word in selected_words:
        if word not in words_count: d[word] = 0
        else: d[word] = words_count[word]
    return d


# In[34]:

products['selected_word_count'] = products['word_count'].apply(select_words)


# In[35]:

products['selected_word_count']


# In[47]:

for word in selected_words:
    products[word] = products['word_count'].apply(lambda word_count: word_count[word] if word in word_count else 0)


# In[48]:

products


# ### Out of the selected_words, which one is most used in the dataset? Which one is least used?

# In[50]:

max_count, min_count = 0, 9223372036854775807
max_word, min_word = selected_words[0], selected_words[0]
for word in selected_words:
    total = products[word].sum()
    if total > max_count:
        max_word = word
        max_count = total
    if total < min_count:
        min_word = word
        min_count = total
print('most used: ', max_word, max_count)
print('least used: ', min_word, min_count)


# In[51]:

train_data,test_data = products.random_split(.8, seed=0)


# In[53]:

selected_words_model = graphlab.logistic_classifier.create(train_data,
                                                     target='sentiment',
                                                     features=selected_words,
                                                     validation_set=test_data)


# In[57]:

selected_words_model['coefficients']


# ### Out of the 11 words in selected_words, which one got the most positive weight? Which one got the most negative weight? Do these values make sense for you?

# In[64]:

most_positive = -1
most_positive_word = ''
most_negative = 1
most_negative_word = ''
for row in selected_words_model['coefficients']:
    if row['name'] == '(intercept)': continue
    if row['value']>most_positive:
        most_positive = row['value']
        most_positive_word = row['name']
    if row['value']<most_negative:
        most_negative = row['value']
        most_negative_word = row['name']
print('most positive: ', most_positive_word, most_positive)
print('most negative: ', most_negative_word, most_negative)


# In[66]:

sentiment_model.evaluate(test_data)


# In[67]:

selected_words_model.evaluate(test_data)


# In[68]:

diaper_champ_reviews = products[products['name']=='Baby Trend Diaper Champ']


# In[69]:

diaper_champ_reviews


# In[70]:

diaper_champ_reviews = diaper_champ_reviews.sort('rating', ascending = False)


# In[77]:

diaper_champ_reviews = diaper_champ_reviews.sort('sentiment', ascending = False)


# In[78]:

diaper_champ_reviews


# In[79]:

selected_words_model.predict(diaper_champ_reviews[0:10], output_type='probability')


# In[80]:

sentiment_model.predict(diaper_champ_reviews[0:10], output_type='probability')


# In[76]:

len(products[products['sentiment']==1])/float(len(products))

