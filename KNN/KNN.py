import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def file_read():
    
    dataset = pd.read_csv("Buy_Book.csv")
    
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
            
    x = dataset.iloc[:,0:4].values
    y = dataset.iloc[:,[-1]].values
    return x,y

def split_data():
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=0)
    return x_train,x_test,y_train,Y_test


def model_train():
    
    knn = KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
    knn.fit(x_train,y_train)

    y_pred = knn.predict(x_test)
    
    age = np.array([[55,0,0,0]])
    
    buy_pred = knn.predict(age)
    
    r_square = r2_score(y_test,y_pred)