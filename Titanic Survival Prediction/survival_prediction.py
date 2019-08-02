# Importing important libraries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 

titanic_data=pd.read_csv('train.csv')
print(titanic_data.head(10))

print("No of passengers in original dataset:",len(titanic_data['PassengerId']))


#Analyzing data

sns.countplot(x="Survived", data=titanic_data)
plt.show()

sns.countplot(x="Survived", hue="Sex", data=titanic_data)
plt.show()

sns.countplot(x="Survived", hue="Pclass", data=titanic_data)
plt.show()

titanic_data['Age'].plot.hist()
plt.show()

titanic_data['Fare'].plot.hist()
plt.show()

# titanic_data['Fare'].plot.hist(bin=20,figsize=(10,5))
# plt.show()

# Data Wrangling
print(titanic_data.isnull().sum())

sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap="viridis")
plt.show()

titanic_data.drop('Cabin', axis=1, inplace=True)
# print(titanic_data.head(5))

# print(titanic_data[['Name','Sex','Age']])
titanic_data.dropna(inplace=True)
# print(titanic_data[['Name','Sex','Age']])

# sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap="viridis")
# plt.show()

# print(titanic_data.isnull().sum())

sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
print(sex.head(5))

embark=pd.get_dummies(titanic_data['Embarked'],drop_first=True)
print(embark.head(5))

Pc1=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
print(Pc1.head(5))

titanic_data=pd.concat([titanic_data,sex,embark,Pc1],axis=1)
print(titanic_data.head(5))

titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass'],axis=1,inplace=True)
print(titanic_data.head(5))

#train and test data
#Build the model on the train data and predict the output on the test data

# defining independent and dependent variable
x=titanic_data.drop('Survived',axis=1)
y=titanic_data['Survived']

#splitting data into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression
#created model
logmodel=LogisticRegression()
#fit the model
logmodel.fit(x_train,y_train)
#made predictions
predictions=logmodel.predict(x_test)

#Accuracy Check
from sklearn.metrics import classification_report
classification_report(y_test,predictions)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))