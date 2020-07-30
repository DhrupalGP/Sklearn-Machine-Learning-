import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

def read_file():
    
    dataset = pd.read_csv("Buy_Book.csv")
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
            
    x = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,4].values
    return x,y

def split_data(x,y):
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=0)
    return x_train,x_test,y_train,y_test

def model_train(x_train,x_test,y_train,y_test):
    
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train,y_train)
    
    value = np.array([[29,1,1,0]])
    val_pred = classifier.predict(value)
    
    y_pred = classifier.predict(x_test)
        
    accuracy = accuracy_score(y_test,y_pred)
    
    cn = confusion_matrix(y_test,y_pred)
    
    return classifier