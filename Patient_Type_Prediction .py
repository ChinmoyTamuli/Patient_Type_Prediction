#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,f1_score

import warnings
warnings.filterwarnings(action='ignore')


# In[24]:


data=pd.read_csv('data-ori.csv')
data.head()


# In[25]:


data.info()


# In[26]:


data.isna().sum()


# # Preprocessing

# In[35]:


def preprocess_input(df):
    df=df.copy()
    
    #Binary Encoding
    df['SEX']=df['SEX'].replace({'F':0, 'M':1})
    
    #split df into X and y
    y=df['SOURCE']
    X=df.drop('SOURCE',axis=1)
    
    #Train_test_split
    X_train,X_test,y_train,y_test =train_test_split(X,y,train_size=0.7,shuffle=True,random_state=1)
    
    
    #Scale X
    scaler= StandardScaler()
    scaler.fit(X_train)
    X_train=pd.DataFrame(scaler.transform(X_train),index=X_train.index, columns=X_train.columns)
    X_test=pd.DataFrame(scaler.transform(X_test),index=X_test.index, columns=X_test.columns)
    
    return X_train,X_test,y_train,y_test


# In[36]:


X_train,X_test,y_train,y_test=preprocess_input(data)


# In[37]:


X_train


# In[38]:


y_train


# In[39]:


y_train.value_counts()


# # Training

# In[40]:


models = {
    "Logistic Regression": LogisticRegression(),
    "      Decision Tree": DecisionTreeClassifier(),
    "     Neural Network": MLPClassifier(),
    "      Random Forest": RandomForestClassifier(),
    "  Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name + " trained.")


# # Result

# In[41]:


for name,model in models.items():
    y_pred = model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    print(name+ "Accuracy: {:.2f}%".format(acc*100))


# In[42]:


for name,model in models.items():
    y_pred = model.predict(X_test)
    f1=f1_score(y_test,y_pred,pos_label='in')
    print(name+ "F1-Score: {:.5f}%".format(f1))


# In[ ]:




