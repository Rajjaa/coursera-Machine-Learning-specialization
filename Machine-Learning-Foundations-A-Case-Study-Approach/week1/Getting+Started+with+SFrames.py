
# coding: utf-8

# # Fire up GraphLab Create
# 
# We always start with this line before using any part of GraphLab Create. It can take up to 30 seconds to load the GraphLab library - be patient!
# 
# The first time you use GraphLab create, you must enter a product key to license the software for non-commerical academic use. To register for a free one-year academic license and obtain your key, go to [turi.com](https://turi.com/download/academic.html).

# In[3]:

import graphlab
# Set product key on this computer. After running this cell, you will not need to re-enter your product key. 
graphlab.product_key.set_product_key('AB73-834E-66A3-257F-7F2F-3EDF-CECD-CC55')

# Limit number of worker processes. This preserves system memory, which prevents hosted notebooks from crashing.
graphlab.set_runtime_config('GRAPHLAB_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

# Output active product key.
graphlab.product_key.get_product_key()


# # Load a tabular data set

# In[4]:

sf = graphlab.SFrame('people-example.csv')


# # SFrame basics

# In[5]:

sf # we can view first few lines of table


# In[6]:

sf.tail()  # view end of the table


# # GraphLab Canvas

# In[7]:

# .show() visualizes any data structure in GraphLab Create
# If you want Canvas visualization to show up on this notebook, 
# add this line:
graphlab.canvas.set_target('ipynb')


# In[8]:

sf['age'].show(view='Categorical')


# # Inspect columns of dataset

# In[ ]:

sf['Country']


# In[ ]:

sf['age']


# Some simple columnar operations

# In[ ]:

sf['age'].mean()


# In[ ]:

sf['age'].max()


# # Create new columns in our SFrame

# In[ ]:

sf


# In[ ]:

sf['Full Name'] = sf['First Name'] + ' ' + sf['Last Name']


# In[ ]:

sf


# In[ ]:

sf['age'] * sf['age']


# # Use the apply function to do a advance transformation of our data

# In[ ]:

sf['Country']


# In[ ]:

sf['Country'].show()


# In[ ]:

def transform_country(country):
    if country == 'USA':
        return 'United States'
    else:
        return country


# In[ ]:

transform_country('Brazil')


# In[ ]:

transform_country('Brasil')


# In[ ]:

transform_country('USA')


# In[ ]:

sf['Country'].apply(transform_country)


# In[ ]:

sf['Country'] = sf['Country'].apply(transform_country)


# In[ ]:

sf


# In[ ]:



