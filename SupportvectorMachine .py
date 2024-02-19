#!/usr/bin/env python
# coding: utf-8

# In[1]:


##importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('loan_approved.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


data.Loan_Amount_Term.value_counts()


# In[8]:


data.describe(include='O')


# In[9]:


get_ipython().system('pip install sweetviz')


# In[10]:


import sweetviz as sv #  library for univariant analysis

my_report = sv.analyze(data)## pass the original dataframe

my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"


# In[11]:


data1=data[['Gender','Married','Dependents','Education','Self_Employed','Property_Area']]
data2=data[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]


# In[12]:


plt.figure(figsize=(20,25), facecolor='white')#To set canvas 
plotnumber = 1#counter

for column in data1:#accessing the columns 
    if plotnumber<=16 :
        ax = plt.subplot(4,4,plotnumber)
        sns.countplot(x=data1[column],hue=data['Loan_Status (Approved)'])
        plt.xlabel(column,fontsize=20)#assign name to x-axis and set font-20
        plt.ylabel('Loan Status',fontsize=20)
    plotnumber+=1#counter increment
plt.tight_layout()


# In[13]:


plt.figure(figsize=(20,25), facecolor='white')#To set canvas 
plotnumber = 1#counter

for column in data2:#accessing the columns 
    if plotnumber<=16 :
        ax = plt.subplot(4,4,plotnumber)
        sns.histplot(x=data2[column],hue=data['Loan_Status (Approved)'])
        plt.xlabel(column,fontsize=20)#assign name to x-axis and set font-20
        plt.ylabel('Loan Status',fontsize=20)
    plotnumber+=1#counter increment
plt.tight_layout()


# In[14]:


data.isnull().sum()


# In[15]:


data.loc[data['Gender'].isnull()==True]


# In[16]:


data.Gender.value_counts()


# In[17]:


import seaborn as sns

sns.countplot(x='Gender',hue='Loan_Status (Approved)',data=data)


# In[18]:


data.loc[data['Gender'].isnull()==True,'Gender']='Male'


# In[19]:


data.Gender.isnull().sum()


# In[20]:


## Getting the values in Dependents
data.loc[data['Dependents'].isnull()==True]


# In[21]:


sns.countplot(x='Dependents',data=data,hue='Loan_Status (Approved)')


# In[22]:


data.loc[data['Dependents'].isnull()==True,'Dependents']='3+'


# In[23]:


data


# In[24]:


## For married feature
data.loc[data['Married'].isnull()==True]


# In[27]:


#sns.countplot(x='Married',hue='Loan_Status(Approved)',data=data)


# In[28]:


data.Married.value_counts()


# In[29]:


data.loc[data['Married'].isnull()==True,'Married']='Yes'


# In[30]:


data.loc[data['Self_Employed']=='No']


# In[32]:


data.Self_Employed.value_counts()


# In[33]:


sns.countplot(x='Self_Employed',data=data,hue='Loan_Status(Approved)')


# In[34]:


# Replace the nan values with mode
data.loc[data['Self_Employed'].isnull()==True,'Self_Employed']='No'


# In[35]:


data.isnull().sum()


# In[36]:


data.LoanAmount.hist()
plt.show()


# In[37]:


np.median(data.LoanAmount.dropna(axis=0))


# In[38]:


# Replace the nan values in LoanAmount column with median value
data.loc[data['LoanAmount'].isnull()==True,'LoanAmount']=np.median(data.LoanAmount.dropna(axis=0))


# In[39]:


data.Loan_Amount_Term.isnull().sum()


# In[40]:


data.Loan_Amount_Term.median()


# In[41]:


data.Loan_Amount_Term.hist()


# In[42]:


data.Credit_History.value_counts()


# In[44]:


sns.countplot(x='Credit_History',data=data,hue='Loan_Status(Approved)')


# In[45]:


pd.get_dummies(data['Gender'],prefix='Gender')


# In[46]:


pd.get_dummies(data['Gender'],prefix='Gender',drop_first=True)


# In[ ]:




