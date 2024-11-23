# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10.Find the accuracy of our model and predict the require values.
## Program:
### 1.DATA SET:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: LINGESWARAN K
RegisterNumber: 212222110022
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Placement_Data.csv')
dataset
```
### 2.DATATYPES
```
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
```
```
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```

### 3.DATA STATUS:
```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
```
### 4.ARRAY:
```
Y
theta=np.random.randn(X.shape[1])
y=Y
```
### 5.THETA
```
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,X)
```
### 6.ACCURACY:
```
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
```
### 7.Y_PRED:
```
print(y_pred)
```
### 8.Y:
```
print(Y)
```
### 9.PREDNEW
```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
### 1.DATA SET:
![image](https://github.com/user-attachments/assets/6aa0652f-7130-487e-a03c-f552752ebbf8)
### 2.DATATYPES
![image](https://github.com/user-attachments/assets/e7f28b0e-33d3-49c2-af99-2b391215a8fe)
### 3.DATA STATUS:
![image](https://github.com/user-attachments/assets/cd8fe914-99af-40db-ac22-7fb04af91300)
### 4.ARRAY:
![image](https://github.com/user-attachments/assets/1caf5422-28e6-4f46-af7a-7a9622580961)
### 6.ACCURACY:
![image](https://github.com/user-attachments/assets/982399c3-adb1-41c4-bf2e-db984c63d1f3)
### 7Y_PRED:
![image](https://github.com/user-attachments/assets/74e2bc78-cf86-414f-a881-fab5ced4cca2)
### 8Y:
![image](https://github.com/user-attachments/assets/ad263174-24a9-4609-a691-764d413e0d20)
### 9.PREDNEW
![image](https://github.com/user-attachments/assets/f4cb1eb8-2d2f-4bc9-8cfa-64b3f33d2cb2)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

