
# coding: utf-8

# In[1]:

import graphlab
# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)


# In[3]:

sales = graphlab.SFrame('home_data.gl/')
sales


# In[5]:

graphlab.canvas.set_target('ipynb')
sales.show(view='BoxWhisker Plot', x='zipcode', y='price')


# #### The zipcode having the highest average price: 98039

# In[9]:

zip_98039 = sales[sales['zipcode'] == '98039']
zip_98039


# In[10]:

zip_98039['price'].mean()


# In[11]:

len(sales)


# In[14]:

sqft_living_2000_4000 = sales[(2000<sales['sqft_living']) & (sales['sqft_living']<=4000)]


# In[15]:

sqft_living_2000_4000


# In[22]:

frac = float(len(sqft_living_2000_4000))/len(sales)
frac


# In[24]:

advanced_features = [
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house				
'grade', # measure of quality of construction				
'waterfront', # waterfront property				
'view', # type of view				
'sqft_above', # square feet above ground				
'sqft_basement', # square feet in basement				
'yr_built', # the year built				
'yr_renovated', # the year renovated				
'lat', 'long', # the lat-long of the parcel				
'sqft_living15', # average sq.ft. of 15 nearest neighbors 				
'sqft_lot15', # average lot size of 15 nearest neighbors 
]


# In[25]:

train_data,test_data = sales.random_split(.8,seed=0)


# In[26]:

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']


# In[27]:

my_features_model = graphlab.linear_regression.create(train_data,target='price',features=my_features,validation_set=None)
advanced_features_model = graphlab.linear_regression.create(train_data,target='price',features=advanced_features,validation_set=None)


# In[28]:

print advanced_features_model.evaluate(test_data)
print my_features_model.evaluate(test_data)

