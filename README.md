# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Roshini RK
RegisterNumber: 212222230123 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


```

## Output:

### df.head()
![dfhead](https://user-images.githubusercontent.com/118956165/229857625-f2a8e49b-184d-4272-afe2-073c3a4bf63f.png)

### df.tail()
![dftail](https://user-images.githubusercontent.com/118956165/229857646-23adf949-d3eb-41c8-9c9b-d7fc688a5ce7.png)

### Array value of X
![xvalue](https://user-images.githubusercontent.com/118956165/229857674-b1f1b530-334a-4c2f-b7eb-25bf43c87e97.png)

### Array value of Y
![yvalue](https://user-images.githubusercontent.com/118956165/229857699-9d6650c6-2964-43a1-9c2f-747e41f9eb83.png)

### Values of Y prediction
![ypred](https://user-images.githubusercontent.com/118956165/229857786-b0b6ac5f-2eda-4689-be2e-4f0bbd4e60c7.png)

### Array values of Y test
![ytest](https://user-images.githubusercontent.com/118956165/229857802-ad68e4f0-11a1-4048-9174-dde763ed34ef.png)

### Training Set Graph
![train](https://user-images.githubusercontent.com/118956165/229857853-00e85b4c-3948-47da-9579-dcd3f2c6b807.png)

### Test Set Graph
![test](https://user-images.githubusercontent.com/118956165/229857871-95eaa08f-d163-498a-82d6-45095514f6dc.png)

### Values of MSE, MAE and RMSE
![mse](https://user-images.githubusercontent.com/118956165/229857913-1d68f680-18b8-4e33-952c-7497824c36a5.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
