import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel(r'C:\Users\Lenovo\Desktop\Iris_Flask\Iris.xls')
df.head()
from sklearn.model_selection import train_test_split
#Splitting the data set into target and features
x=df.drop(['Classification'],axis=1) 
y=pd.DataFrame(df['Classification'])
# Spliting the dataset for training and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.25)
# logistic regression 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
m=model.fit(x_train, y_train)
pickle.dump(model,open('model.pkl','wb'))