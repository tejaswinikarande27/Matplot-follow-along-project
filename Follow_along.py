#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Import the libraries and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


### display all the rows and columns
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


# In[4]:


### Load the dataset
df=pd.read_csv("auto-mpg.csv")


# In[5]:


### Shape
df.shape


# In[6]:


### datatypes
df.dtypes


# In[7]:


### Missing values??
df.isnull().sum()


# There are no direct missing values

# In[8]:


df.head()


# In[9]:


df["hp"].value_counts()


# There are 6 ? in this dataset.
# We need to replace ? with median

# In[10]:


df["hp"]=df["hp"].replace("?",np.nan)
df["hp"]=df["hp"].astype(float)


# In[11]:


df.dtypes


# In[12]:


df.describe()


# In[13]:


df["hp"]=df["hp"].replace(np.nan,df["hp"].median())


# In[14]:


df.describe()


# In[15]:


### Change origin into catgoric->
df["origin"]=df["origin"].replace({1:"america",2:"europe",3:"asia"})


# In[16]:


df.sample(10)


# In[17]:


df.hist(figsize=(10,12))
plt.show()


# In[18]:


df.skew(numeric_only=True)


# In[19]:


sns.boxplot(x="mpg",data=df)


# In[20]:


sns.boxplot(x="cyl",data=df)


# In[21]:


import plotly.express as px
for column in df:
    fig=px.histogram(df,x=column,nbins=20)
    fig.show()


# In[22]:


for column in df:
    fig=px.box(df,x=column)
    fig.show()


# In[23]:


sns.pairplot(df)


# In[24]:


plt.scatter(df["hp"],df["mpg"])


# In[25]:


sns.countplot(x="origin",data=df)


# In[26]:


sns.violinplot(x="origin",y="mpg",data=df)


# In[27]:


sns.barplot(x="cyl",y="mpg",data=df)


# In[28]:


sns.lineplot(x="yr",y="mpg",data=df)


# In[29]:


sns.jointplot(x="wt",y="mpg",data=df)


# In[30]:


corr_matrix=df.corr(numeric_only=True)
corr_matrix


# In[31]:


sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")


# In[33]:


sns.kdeplot(x="wt", y="mpg", data=df)


# In[34]:


sns.boxenplot(x="origin", y="mpg", data=df)


# In[35]:


g = sns.FacetGrid(col='origin', row='yr',data=df)
g.map(sns.scatterplot, 'cyl', 'acc')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('FacetGrid Plot')
plt.show()


# In[37]:


sns.pairplot(data=df, hue="mpg", diag_kind="kde", palette="husl")


# In[ ]:




