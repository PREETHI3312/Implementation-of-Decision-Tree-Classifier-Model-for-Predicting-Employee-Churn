# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: PREETHI A K
RegisterNumber:  212223230156

import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
*/
```

## Output:
![image](https://github.com/user-attachments/assets/b17a69d7-d7ce-4d0c-b6c2-e9452d7bc2dc)

![image](https://github.com/user-attachments/assets/51fcf352-aa40-41c3-b89b-295521dcd5fb)

![image](https://github.com/user-attachments/assets/467c954e-132f-4d5a-b835-d57f98bcf475)

![image](https://github.com/user-attachments/assets/e71c7d5f-fc26-4a45-8fc5-3408582748ca)



![image](https://github.com/user-attachments/assets/32a8a33e-5f5b-482e-913a-2c59c6f5ca71)



![image](https://github.com/user-attachments/assets/3fef50fe-1f72-42fb-a019-4641a835b579)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
