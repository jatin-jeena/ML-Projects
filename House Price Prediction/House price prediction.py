# Importing libraries
from sklearn.linear_model import LinearRegression
import pandas as pd 
from sklearn.model_selection import train_test_split

# Reading rthe csv file into a dataframe using pandas
data = pd.read_csv("Housing.csv")

# Displaying top 5 rows of the data read
print(data.head())

# Splitting the complete dataset into train and test
# Keeping the test size of data 30%
# and 70% traing data
x_train, x_test, y_train, y_test = train_test_split(data['lotsize'], data['price'], test_size= 0.3, random_state=1)

# Creating linear regression model
model = LinearRegression()

# Reshaping the data
x_train= x_train.values.reshape(-1,1)
y_train= y_train.values.reshape(-1,1)
x_test= x_test.values.reshape(-1,1)

# Fitting the training data into the model
model.fit(x_train, y_train)

# Making predictions on the test data
predictions = model.predict(x_test)

# Displaying predicted values
print(predictions)