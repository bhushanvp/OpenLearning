#!/usr/bin/env python
# coding: utf-8

# In[62]:


import river
import pandas as pd


# In[63]:


df = pd.read_csv("/home/bhushan/research/client_side_online_learning/DSBDALExam_DataSets/Hepatitis/hepatitis.csv")


# In[64]:


col = ["Class","AGE","SEX","TEROID","ANTIVIRALS","FATIGUE","MALAISE","ANOREXIA","LIVER BIG","LIVER FIRM","SPLEEN PALPABLE","SPIDERS","ASCITES","VARICES","BILIRUBIN","ALK PHOSPHATE","SGOT","ALBUMIN","PROTIME","HISTOLOGY"]
df.columns = col


# In[65]:


df.replace("?", pd.NA, inplace=True)
df = df.dropna()


# In[66]:


df = df.drop(["SGOT", "ALK PHOSPHATE", "FATIGUE", "ANOREXIA", "LIVER FIRM", "LIVER BIG", "SPLEEN PALPABLE", "SEX", "TEROID", "ANTIVIRALS"], axis=1)


# In[67]:


for i in df:
    df[i] = df[i].astype(float)
    


# In[68]:


y = df["Class"]
X = df.drop(["Class"], axis=1)



# In[69]:




# In[70]:


from river.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from river import stream
from river import preprocessing
from river import compose
import numpy as np


# In[71]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.95)


# In[72]:


scaler = preprocessing.StandardScaler()
gnb = GaussianNB()

model = compose.Pipeline(gnb)

new_x_train = x_train.values
new_y_train = y_train.values

for x, y in stream.iter_array(new_x_train, new_y_train):
    _ = model.learn_one(x, y)


# In[73]:


y_pred = []
new_x_test = x_test.values
new_y_test = y_test.values

acc = []

for x, y in stream.iter_array(new_x_test, new_y_test):
    y_pred.append(model.predict_one(x))
    _ = model.learn_one(x, y)
    #acc.append(accuracy_score(y_test[:len(y_pred)], y_pred))

print("Accuracy =", accuracy_score(y_pred, y_test))
