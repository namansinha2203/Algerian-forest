#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = df.to_dict(orient = 'records')


# In[7]:


df.info()


# In[8]:


df[df.isnull().any(axis=1)]


# In[9]:


df.loc[:122,'Region']=1
df.loc[122:,'Region']=2
df[['Region']] = df[['Region']].astype(int)


# In[10]:


df.isnull().sum()


# In[11]:


df =df.dropna().reset_index(drop=True)
df.shape


# In[12]:


df.iloc[[122]]


# In[13]:


df= df.drop(122).reset_index(drop=True)


# In[14]:


df.columns


# In[15]:


df.columns = df.columns.str.strip()
df.columns


# In[16]:


df[['month', 'day', 'year', 'Temperature','RH', 'Ws']] = df[['month', 'day', 'year', 'Temperature','RH', 'Ws']].astype(int)


# In[17]:


objects = [features for features in df.columns if df[features].dtypes=='O']
for i in objects:
    if i != 'Classes':
        df[i] = df[i].astype(float)


# In[18]:


df.info()


# In[19]:


df.describe().T


# In[20]:


df.Classes.value_counts()


# In[21]:


df.Classes = df.Classes.str.strip()


# In[22]:


df.Classes.value_counts()


# In[23]:


df[:122]


# In[24]:


df[122:]


# In[25]:


df.to_csv('Algerian_forest_fires_dataset_CLEANED.csv', index=False)


# In[26]:


df1 = df.drop(['day','month','year'], axis=1)


# In[27]:


df1['Classes']= np.where(df1['Classes']== 'not fire',0,1)


# In[28]:


df1.Classes.value_counts()


# In[29]:


plt.style.use('seaborn')
df1.hist(bins=50, figsize=(20,15), ec = 'b')
plt.show()


# In[30]:


percentage = df.Classes.value_counts(normalize=True)*100
percentage


# In[31]:


classeslabels = ["FIRE", "NOT FIRE"]
plt.figure(figsize =(12, 7))
plt.pie(percentage,labels = classeslabels,autopct='%1.1f%%')
plt.title ("Pie Chart of Classes", fontsize = 15)
plt.show()


# In[32]:


k = len(df1.columns)
cols = corr.nlargest(k, 'Classes')['Classes'].index
cm = np.corrcoef(df1[cols].values.T)
sns.set(font_scale=1)
f, ax = plt.subplots(figsize=(20, 13))
hm = sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[33]:


ax = sns.boxplot(df['FWI'], color= 'red')


# In[34]:


dftemp= df.loc[df['Region']== 1]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data= df,ec = 'black', palette= 'Set2')
plt.title('Fire Analysis Month wise for Bejaia Region', fontsize=18, weight='bold')
plt.ylabel('Count', weight = 'bold')
plt.xlabel('Months', weight= 'bold')
plt.legend(loc='upper right')
plt.xticks(np.arange(4), ['June','July', 'August', 'September',])
plt.grid(alpha = 0.5,axis = 'y')
plt.show()


# In[35]:


dftemp= df.loc[df['Region']== 2]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data= df,ec = 'black', palette= 'Set2')
plt.title('Fire Analysis Month wise for Sidi-Bel Abbes Region', fontsize=18, weight='bold')
plt.ylabel('Count', weight = 'bold')
plt.xlabel('Months', weight= 'bold')
plt.legend(loc='upper right')
plt.xticks(np.arange(4), ['June','July', 'August', 'September',])
plt.grid(alpha = 0.5,axis = 'y')
plt.show()


# In[36]:


df.columns


# In[37]:


def barchart(feature,xlabel):
    plt.figure(figsize=[14,8])
    by_feature =  df1.groupby([feature], as_index=False)['Classes'].sum()
    ax = sns.barplot(x=feature, y="Classes", data=by_feature[[feature,'Classes']], estimator=sum)
    ax.set(xlabel=xlabel, ylabel='Fire Count')


# In[38]:


barchart('Temperature','Temperature Max in Celsius degrees')


# In[39]:


barchart('Rain', 'Rain in mm')


# In[40]:


barchart('Ws', 'Wind Speed in km/hr')


# In[42]:


barchart('RH','Relative Humidity in %')


# In[43]:


dftemp = df1.drop(['Classes', 'Region'], axis=1)
fig = plt.figure(figsize =(12, 6))
ax = dftemp.boxplot()
ax.set_title("Boxplot of Given Dataset")
plt.show()


# In[44]:


dftemp = dftemp = df1.drop(['Region','Temperature','Rain','Ws','RH'], axis=1)
for feature in dftemp:
    sns.histplot(data = dftemp,x=feature, hue = 'Classes')
    plt.legend(labels=['Fire','Not Fire'])
    plt.title(feature)
    plt.show()


# In[45]:


df.info()


# In[ ]:




