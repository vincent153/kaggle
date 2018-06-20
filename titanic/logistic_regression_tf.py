import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))+'/'
dataset = pd.read_csv(SCRIPT_PATH + "train.csv")
features = ['Sex', 'Age', 'Cabin', 'Embarked']


learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

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
    # tf Graph Input
    input_feature_nums = x_train.shape[1]
    
    x = tf.placeholder(tf.float32, [None, input_feature_nums])
    y = tf.placeholder(tf.float32, [None, 1])
    pass

def predict(test_data):
    pass


if __name__ == '__main__':
    training_data = dataset[features]
    training_label = dataset['Survived']
    training_data = fillna(training_data)
    print(training_data.info())

    testset = pd.read_csv(SCRIPT_PATH + "/test.csv")
    test_data = testset[features]
    test_data = fillna(test_data)
    print(training_data.info())

    train(training_data.values,training_label.values)
    #predict(test_data.values)
    pass

