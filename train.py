# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:18:30 2022

@author: User
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime
from unseen_articles import ExploDataAnalysis,CreateModel, EvaluateModel

#%%CONSTANT
PATH_LOG = os.path.join(os.getcwd(),'log')
MODEL_PATH = os.path.join(os.getcwd(),'model','unseen_articles.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(),'model','tokenizer.json')
URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'

log_dir = os.path.join(PATH_LOG,
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

#%%EDA

#Step 1: Load Data
df = pd.read_csv(URL)

category = df['category'] #target
text = df['text'] #features
    

eda = ExploDataAnalysis()

#Step 2: Data cleaning

#converting all the texts into lower case
split_text =eda.lower_case_split(text)

token_data = eda.article_tokenizer(text,token_save_path=TOKENIZER_PATH)


[np.shape(i) for i in token_data]

pad_text = eda.article_padding(token_data)

#%% DATA PRE-PROCESSING

ohe = OneHotEncoder(sparse=False)
category_fitted = ohe.fit_transform(np.expand_dims(category,-1))

X_train, X_test, y_train, y_test = train_test_split(pad_text,
                                                    category_fitted,
                                                    test_size=0.3,
                                                    random_state=123)

#Expand dimension to 3D to fit into model
X_train=np.expand_dims(X_train,axis=-1)
X_test=np.expand_dims(X_test,axis=-1)

#%% MODEL CREATION
    
nb_categories = len(category.unique())
mc= CreateModel()
model = mc.lstm_layer(5000,nb_categories,nodes=60)

#compile and model fitting
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

#tensorboard
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

hist = model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test),
          callbacks=[tensorboard_callback,early_stopping])

#%% MODEL SAVE
model.save(MODEL_PATH)

#%% MODEL EVALUATION

#Prediction based on 5 category (Sport,Tech,Business, Entertainment, Politics)
predicted_advanced = np.empty([len(X_test),5]) 

for index, test in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(test,axis=0))

y_pred = np.argmax(predicted_advanced,axis=1)
y_true = np.argmax(y_test,axis=1)

#Reporting
#The accuracy score has been improved from 65% to 87%, in this case by two way:
# 1: After nodes has been increased from 30 to 60
# 2: While adding another neural networks extension to LSTM which is BiDirectional
me = EvaluateModel()
me.reporting_metrics(y_true,y_pred)
