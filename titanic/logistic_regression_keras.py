import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))+'/'
dataset = pd.read_csv(SCRIPT_PATH + "train.csv")
features = ['Sex', 'Age', 'Cabin', 'Embarked']


def fillna(dataframe):
    for item in features:
        uniq = dataframe[item].unique()
        index = random.randint(0,len(uniq)-1)  
        tmp = uniq[index]
        while tmp is np.nan:
            print('get new rnd data')
            index = random.randint(0,len(uniq)-1)  
            tmp = uniq[index]  
        dataframe = dataframe.fillna({item:tmp})
        dataframe[item] = LabelEncoder().fit_transform(dataframe[item])
    return dataframe


def train(x_train,y_train):
    model = Sequential()
    model.add(Dense(1,activation='sigmoid',input_dim=x_train.shape[1]))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=500)
    model.save('titanic.h5')
    del model

def predict(test_data):
    print(test_data.shape)   
    model = load_model('titanic.h5')
    res = model.predict_classes(test_data)    
    print(res.shape)
    res = np.reshape(res,res.shape[0])    
    results = pd.DataFrame({
    'PassengerId' : testset['PassengerId'],
    'Survived' : res
    })
    results.to_csv(SCRIPT_PATH + "/submission11.csv", index=False)





if __name__ == '__main__':
    training_data = dataset[features]
    training_label = dataset['Survived']
    training_data = fillna(training_data)
    print(training_data.info())

    testset = pd.read_csv(SCRIPT_PATH + "/test.csv")
    test_data = testset[features]
    test_data = fillna(test_data)
    print(training_data.info())

    #train(training_data.values,training_label.values)
    predict(test_data.values)
    pass

