# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.
2.Assign hours to X and scores to Y.
3.Implement training set and test set of the dataframe.
4.Plot the required graph both for test data and training data.
5.Find the values of MSE , MAE and RMSE.

## Program And Output:
# Developed by:SURIYA PRAKASH.S
# Reg no:212223100055
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')

#Displaying the content in datafile
df.head()
```
![image](https://github.com/user-attachments/assets/0d3ff5ec-fbbd-4860-bc4e-ba7da74d7c41)
```
#Last five rows
df.tail()
```
![image](https://github.com/user-attachments/assets/d4c2a29a-e721-4b34-98af-fd55e6be210f)

```
#Segregating data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
```
![image](https://github.com/user-attachments/assets/8c43bb04-b0ce-44b1-a523-d3f140063d20)

```
#Splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#Displaying predicted values
Y_pred
```

![image](https://github.com/user-attachments/assets/279d5db1-3638-4806-93e4-e9a29b06672a)

```
Y_test
```

![image](https://github.com/user-attachments/assets/5d025c9d-f82d-479d-b54d-f316b8fb5980)

```
#Graph plot for training data
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours VS Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
![image](https://github.com/user-attachments/assets/bbe0850b-5ed2-4b47-b85c-24fb40bef582)

```
#Graph plot for test data
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="orange")
plt.title("Hours VS Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

![image](https://github.com/user-attachments/assets/59a97443-048c-4a12-8a9c-f64ea94aba8f)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
