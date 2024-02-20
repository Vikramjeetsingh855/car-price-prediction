#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# In[2]:


dataset = pd.read_csv("car data.csv")


# In[72]:


dataset.head(3)


# In[4]:


dataset.drop(columns='Selling_type',inplace=True)


# In[5]:


dataset.info()


# In[6]:


dataset.isnull().sum()


# In[7]:


dataset.select_dtypes(include="object").columns


# In[8]:


cn = LabelEncoder()
cn.fit(dataset["Car_Name"])
dataset["Car_Name"]  =  cn.transform(dataset["Car_Name"])


# In[9]:


ft = LabelEncoder()
ft.fit(dataset["Fuel_Type"])
dataset["Fuel_Type"]  =  ft.transform(dataset["Fuel_Type"])


# In[10]:


Tr = LabelEncoder()
Tr.fit(dataset["Transmission"])
dataset["Transmission"]  =  Tr.transform(dataset["Transmission"])


# In[ ]:





# In[ ]:





# In[11]:


plt.figure(figsize=(4,3))
sns.kdeplot(x="Driven_kms",data=dataset)
plt.show()


# In[12]:


plt.figure(figsize=(4,3))
sns.boxplot(x="Driven_kms",data=dataset)
plt.show()


# In[13]:


q1 = np.percentile(dataset["Driven_kms"],25)
q3 = np.percentile(dataset["Driven_kms"],75)
q1,q3


# In[14]:


iqr = q3-q1


# In[15]:


max_r = q3+(1.5*iqr)
min_r = q1-(1.5*iqr)
min_r,max_r


# In[16]:


dataset.loc[dataset["Driven_kms"] > max_r, "Driven_kms"] = max_r


# In[17]:


dataset.shape


# In[ ]:





# In[18]:


plt.figure(figsize=(4,3))
sns.kdeplot(x="Present_Price",data=dataset)
plt.show()


# In[19]:


plt.figure(figsize=(4,3))
sns.boxplot(x="Present_Price",data=dataset)
plt.show()


# In[20]:


q11 = np.percentile(dataset["Present_Price"],25)
q31 = np.percentile(dataset["Present_Price"],75)
q11,q31


# In[21]:


iqr1 = q31-q11


# In[22]:


max_r1 = q31+(1.5*iqr1)
min_r1 = q11-(1.5*iqr1)
min_r1,max_r1


# In[23]:


dataset.loc[dataset["Present_Price"] > max_r1, "Present_Price"] = max_r1


# In[ ]:





# In[24]:


# Present_Price_ft = FunctionTransformer(func= np.log1p )
# Present_Price_ft.fit(dataset["Present_Price"])
# dataset["Present_Price"] = Present_Price_ft.transform(dataset["Present_Price"])


# In[25]:


# Driven_kms_ft = FunctionTransformer(func= lambda x : x**0.5)
# Driven_kms_ft.fit(dataset["Driven_kms"])
# dataset["Driven_kms"] = Driven_kms_ft.transform(dataset["Driven_kms"])


# In[ ]:





# In[26]:


dataset.head(3)


# Car_Name	Year	Present_Price	Driven_kms	Fuel_Type	Selling_type	Transmission	Owner	Selling_Price

# In[27]:


sns.heatmap(data=dataset.corr(),annot=True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[53]:


x = dataset.iloc[:,:-1]
y = dataset["Selling_Price"]


# In[54]:


y.head(1)


# In[55]:


# sc = StandardScaler()
# sc.fit(x)
# x  = pd.DataFrame(sc.transform(x),columns=x.columns)


# In[ ]:





# In[56]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:





# In[57]:


dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)


# In[58]:


dt.score(x_test,y_test)*100,dt.score(x_train,y_train)*100


# In[59]:


dt.predict([[90,2014,5.59,27000.0,2,1,0]])


# In[60]:


dataset.head(1)


# In[36]:


y.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[38]:


new_data1 = pd.read_csv("car data.csv")


# In[39]:


new_data = new_data1.drop(columns=["Present_Price","Selling_type","Selling_Price"])


# In[61]:


new_data = new_data.head(1)
new_data


# In[41]:


sl = new_data1[(new_data1["Car_Name"]=="ritz")&(new_data1["Year"]==2014)]["Present_Price"][0]


# In[42]:


new_data.insert(2,"Present_Price",sl)


# In[43]:


new_data["Car_Name"] = cn.transform(new_data["Car_Name"])


# In[44]:


new_data["Fuel_Type"] = ft.transform(new_data["Fuel_Type"])


# In[45]:


new_data["Transmission"] = Tr.transform(new_data["Transmission"])


# In[62]:


# new_data = pd.DataFrame(sc.transform(new_data),columns=new_data.columns)


# In[63]:


dt.predict(new_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[64]:


from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose  import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


# In[65]:


new_data_x = new_data1.drop(columns=["Selling_type","Selling_Price"])
new_data_x.head(2)


# In[66]:


new_data_y = new_data1["Selling_Price"]


# In[ ]:


new_data_x.columns


# In[ ]:


num_data = new_data_x.select_dtypes(["int64","float64"]).columns
cat_data = new_data_x.select_dtypes(["object"]).columns


# In[67]:


c1 = ColumnTransformer([("t1",OrdinalEncoder(),[0,4,5])],remainder="passthrough")


# In[68]:


pipe = Pipeline(steps=[("en",ColumnTransformer([("t1",OrdinalEncoder(),[0,4,5])],remainder="passthrough"))
                       ,("sec",StandardScaler()),("lr_mo",DecisionTreeRegressor())])


# In[69]:


pipe.fit(new_data_x,new_data_y)


# In[ ]:





# In[70]:


import pickle


# In[71]:


p = open("mod.txt","wb")
pickle.dump(pipe,p)
p.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




