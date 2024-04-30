# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: ARAVIND SAMY.P
RegisterNumber: 212222230011
*/

import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
### Encoding
![326333232-14c93d11-d8d5-4f91-8b80-1ba4f76af49d](https://github.com/Aravindsamy04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497037/0b4dfdfe-2b8c-4478-9db2-66e0d4d302d9)

### Head()
![326333270-49754a09-46dc-4838-8c17-7e6d9203f94a](https://github.com/Aravindsamy04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497037/d31e02c0-a1b1-475a-98be-9bfd7497b6b8)


### Info()

![326333320-3d140aa9-c186-4943-a70b-a56460a90375](https://github.com/Aravindsamy04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497037/f4531a1c-991d-4384-b4e9-2f465e0fb95b)


### isnull().sum()
![326333339-995fe78e-4661-46d3-84e3-fada28820617](https://github.com/Aravindsamy04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497037/866ba595-6e22-4e19-9060-e83486b8ff4e)

### Prediction of y
![326333367-3dc11444-a410-4950-a8d9-fde59ed078e9](https://github.com/Aravindsamy04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497037/a1fe4573-ac41-4e57-8469-27087ef1721c)



### Accuracy
![326333395-7f0a5fc6-1c65-44be-9217-370588a7bf9a](https://github.com/Aravindsamy04/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497037/28ccd9f4-25a1-4ba1-a8ef-0d3f12596fc8)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
