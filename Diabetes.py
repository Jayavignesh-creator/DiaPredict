

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("E:\\Diabetes.csv")


# In[3]:


df.head(20)


# In[4]:


df2 = df.iloc[:,:-1]
print((df2[:] == 0).sum())


# In[5]:


print(((df2[:] == 0).sum())/768*100)


# In[6]:


import seaborn as sns
def plotHistogram(values,label,feature,title):
    sns.set_style("whitegrid")
    plotOne = sns.FacetGrid(values, hue=label,aspect=2)
    plotOne.map(sns.distplot,feature,kde=False)
    plotOne.set(xlim=(0, values[feature].max()))
    plotOne.add_legend()
    plotOne.set_axis_labels(feature, 'Proportion')
    plotOne.fig.suptitle(title)
plotHistogram(df,"Outcome",'Insulin','Insulin vs Outcome (Blue = Healthy patients; Orange = Diabetic patients)')
plotHistogram(df,"Outcome",'SkinThickness','SkinThickness vs Outcome (Blue = Healthy patients; Orange = Diabetic patients)')


# In[7]:


g = sns.heatmap(df.corr(),cmap="BrBG",annot=False)


# In[9]:


df.drop(['Insulin','SkinThickness','DiabetesPedigreeFunction'],axis=1)


# In[57]:


import xgboost as xgb
from sklearn.model_selection import train_test_split

x = df[['Pregnancies', 'Glucose', 'BloodPressure','BMI','Age']]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=1)
classifier.fit(X_train, y_train)


# In[60]:


predictions = classifier.predict(X_test)


# In[61]:


import pickle
with open("Vignesh.pkl","wb") as file:
    pickle.dump(classifier, file)


# In[ ]:




