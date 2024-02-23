#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv("kc_house_data.csv")


# In[4]:


data.head()


# In[5]:


data.describe()


# In[6]:


data['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
sns.despine


# In[13]:


plt.figure(figsize=(10,10))
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
sns.despine


# In[14]:


plt.scatter(data.price,data.sqft_living)
plt.title("Price vs Square Feet")


# In[15]:


plt.scatter(data.price,data.long)
plt.title("Price vs Location of the area")


# In[16]:


plt.scatter(data.price,data.lat)
plt.xlabel("Price")
plt.ylabel('Latitude')
plt.title("Latitude vs Price")


# In[17]:


plt.scatter(data.bedrooms,data.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine


# In[18]:


plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])


# In[19]:


plt.scatter(data.waterfront,data.price)
plt.title("Waterfront vs Price ( 0= no waterfront)")


# In[20]:


train1 = data.drop(['id', 'price'],axis=1)


# In[21]:


train1.head()


# In[22]:


data.floors.value_counts().plot(kind='bar')


# In[23]:


plt.scatter(data.floors,data.price)


# In[24]:


plt.scatter(data.condition,data.price)


# In[25]:


plt.scatter(data.condition,data.price)


# In[26]:


plt.scatter(data.zipcode,data.price)
plt.title("Which is the pricey location by zipcode?")


# In[53]:


from sklearn.linear_model import LinearRegression


# In[54]:


reg = LinearRegression()


# In[55]:


labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)


# In[56]:


from sklearn.model_selection import train_test_split


# In[50]:


x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)


# In[51]:


reg.fit(x_train,y_train)


# In[43]:


reg.score(x_test,y_test)


# In[62]:


from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'squared_error')


# In[63]:


clf.fit(x_train, y_train)


# In[64]:


clf.score(x_test,y_test)


# In[68]:


from sklearn.linear_model import LinearRegression

# Assuming you've defined your training data as x_train and y_train

# Instantiate the LinearRegression model
reg = LinearRegression()

# Fit the model to the training data
reg.fit(x_train, y_train)

# Now you can make predictions on the test data
y_pred = reg.predict(x_test)



# In[69]:


y_pred = reg.predict(x_test)


# In[74]:


from sklearn.metrics import mean_squared_error

loss_values = []
for i, y_pred in enumerate(clf.staged_predict(x_test)):
    # Compute the mean squared error between true labels and predictions
    loss = mean_squared_error(y_test, y_pred)
    # Append the loss to the array
    loss_values.append(loss)


# In[75]:


testsc = np.arange((params['n_estimators']))+1


# In[76]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
plt.plot(testsc,t_sc,'r-',label = 'set dev test')


# In[77]:


from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


# In[78]:


pca = PCA()


# In[79]:


pca.fit_transform(scale(train1))


# In[ ]:




