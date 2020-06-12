#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
url= r'C:/Users/SAIDHANUSH/iris.csv'
spec= pd.read_csv(url)


# In[2]:


spec.head()


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=(8,6)
plt.rcParams['font.size']= 14


# In[4]:


spec=pd.get_dummies(spec)
spec.head()


# In[5]:


spec.plot(kind='scatter',x='petall' , y='species_setosa', alpha=0.2 )


# In[6]:


sns.lmplot(x='petall', y='species_setosa', data=spec, aspect=1.5, scatter_kws={'alpha':0.2})


# In[7]:


feature_cols = ['petall']
x=spec[feature_cols]
y=spec.species_setosa


# In[8]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(x,y)


# In[9]:


print(lr.intercept_)
print(lr.coef_)


# In[10]:


lr.intercept_+ lr.coef_*1.4


# In[ ]:




