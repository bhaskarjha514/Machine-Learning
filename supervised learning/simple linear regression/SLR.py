#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression


# In[8]:


data = pd.read_csv('Salary_Data.csv')
data.head(10)


# In[34]:


real_x = data.iloc[:,0].values
real_y = data.iloc[:,1].values

real_x = real_x.reshape(-1,1)
real_y = real_y.reshape(-1,1)


# In[22]:


training_x,testing_x,training_y,testing_y = train_test_split(real_x,real_y,test_size = 0.3, random_state=0)
training_x


# In[21]:


Lin = LinearRegression()
Lin.fit(training_x,training_y) #model ready now using training_x,training_y


# In[23]:


Pred_y = Lin.predict(testing_x)


# In[24]:


testing_y[3]


# In[26]:


Pred_y[3]


# In[27]:


plt.scatter(training_x,training_y,color= 'green')
plt.plot(training_x,Lin.predict(training_x),color= 'blue')
plt.title("Salary & Exp Training Plot")
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.show()


# In[28]:


plt.scatter(testing_x,testing_y,color= 'green')
plt.plot(training_x,Lin.predict(training_x),color= 'blue')
plt.title("Salary & Exp Testing Plot")
plt.xlabel('Exp')
plt.ylabel('Salary')
plt.show()

