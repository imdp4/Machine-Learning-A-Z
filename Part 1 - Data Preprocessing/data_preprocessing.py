#Data Preprocessing

"""
@author: Dhaval Pawar (imdp4)

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values

#Taking care of missing values 
from sklearn.impute import SimpleImputer
simple_imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
simple_imputer=simple_imputer.fit(X[:,1:3])
X[:,1:3]=simple_imputer.transform(X[:,1:3]) #transform will calculate mean and place it into the missing data

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features= [0])
X=onehotencoder.fit_transform(X).toarray(X)
labelencoder_y= LabelEncoder()
y=labelencoder_y.fit_transform(y)

#Test and Train Set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
