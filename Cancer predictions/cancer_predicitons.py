# Importing libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# Importing sklearn libraries for creating various classification model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Loading the breast cancer data 
bs = load_breast_cancer()

# Checking the Feature names in the dataset
x = bs.feature_names
print(x)

# Checking the Target names in the dataset
y = bs.target_names
print("*"*30)
print(y)
print("*"*30)

# Splitting the dataset into train and test data 
# Using 70% for training and rest for testing
x_train, x_test, y_train, y_test = train_test_split(bs.data, bs.target,  test_size=0.3, random_state=1)

# Making Support Vector Classifier
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
predic = clf.predict(x_test)
# Checking its accuracy
print("Using SVM Tree ", accuracy_score(y_test, predic)*100)

# Making Dexcision Tree classifier
clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=50)
clf.fit(x_train, y_train)
predic = clf.predict(x_test)
# Checking its accuracy
print("Using Decision Tree ", accuracy_score(y_test, predic)*100)

# Making KNN Classifier
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(x_train, y_train)
predic = clf.predict(x_test)
# Checking its accuracy
print("Using KNN Tree ", accuracy_score(y_test, predic)*100)

# Making Logistic Regression model
clf = LogisticRegression()
clf.fit(x_train, y_train)
predic = clf.predict(x_test)
# Checking its accuracy
print("Using Logistic Tree ", accuracy_score(y_test, predic)*100)

# Making Naive Bayes predicting model
clf = GaussianNB()
clf.fit(x_train, y_train)
predic = clf.predict(x_test)
# Checking its accuracy
print("Using Gaussian Tree ", accuracy_score(y_test, predic)*100)


# COCLUSION - SVM outperforms all other with an accuracy of 95.32%
