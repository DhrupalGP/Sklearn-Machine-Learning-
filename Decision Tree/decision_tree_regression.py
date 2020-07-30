import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

def read_file():
    
    dataset = pd.read_csv("HousePrice.csv")
    
    x = dataset.iloc[:,[0]].values
    y = dataset.iloc[:,[1]].values
    return x,y

def slit_data():
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=0)
    
    return x_train,x_test,y_train,y_test

def model_train(x_train,x_test,y_train,y_test):
    
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(x_train,y_train)
    
    y_pred = tree_reg.predict(x_test)
    
    r_square = r2_score(y_test,y_pred)
    
    mod_acc = accuracy_score(y_test,y_pred)