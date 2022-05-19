# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:58:16 2022

@author: Mona
"""
import os
import pandas as pd
import re
import numpy as np

#Explotary Data Analysis
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

#Model Creation
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout, Bidirectional, Embedding

#Evaluate model
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

#%%

class ExploDataAnalysis():
       
    
    def lower_case_split(self,data):
        for index,text in enumerate(data):
            data[index] = re.sub('[^a-zA-Z]',' ',text).split()
            
        return data
    
    def article_tokenizer(self,data,token_save_path,
                          num_words=5000,oov_token='<OOV>'): 
        #tokenizer to vectorize the words
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)

        #to save the tokenizer for deployment
        token_json = tokenizer.to_json()
        
        with open(token_save_path,'w') as json_file:
            json.dump(token_json, json_file)
            
        #to vectorize the sequences of text
        data = tokenizer.texts_to_sequences(data)
        
        return data
    
    def article_padding(self,data):
        return pad_sequences(data,maxlen=300,padding='post',truncating='post')
#%%    
class CreateModel():
    def lstm_layer(self,num_words,nb_categories,nodes,embedding_output=64,
                          dropout=0.2):
        model = Sequential()
        model.add(Embedding(num_words,embedding_output)) #added embedded layer
        model.add(Bidirectional(LSTM(nodes, return_sequences=(True))))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes, return_sequences=(True))))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dense(nb_categories,activation='softmax'))
        model.summary()
        
        return model
#%%   
class EvaluateModel():
    def reporting_metrics(self,y_true,y_pred):
        print(classification_report(y_true,y_pred))
        print(confusion_matrix(y_true,y_pred))
        print(accuracy_score(y_true,y_pred))
        
    def tensorboard_graph(self,x,y):
        fig, axes = plt.subplots(ncols=5, sharex=(False), sharey=(True))

        for i in range(5):  
            axes[i].set_title(y[i])
            axes[i].imshow(x[i], cmap='gray')
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)

            plt.show()


#%%EDA
if __name__ == '__main__':
    
    PATH_LOG = os.path.join(os.getcwd(),'log')
    MODEL_PATH = os.path.join(os.getcwd(),'model','unseen_articles.h5')
    TOKENIZER_PATH = os.path.join(os.getcwd(),'model','tokenizer.json')
    URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
    
    df = pd.read_csv(URL)

    category = df['category']
    text = df['text']
        
    
    eda = ExploDataAnalysis()
    
    #data cleaning
    #converting all the texts into lower case
    split_text =eda.lower_case_split(text)
    
    #tokenizing
    token_data = eda.article_tokenizer(text,token_save_path=TOKENIZER_PATH)
    

    [np.shape(i) for i in token_data]
    
    #padding
    pad_text = eda.article_padding(token_data)
    #%%MODEL CREATION
    nb_categories = len(category.unique())
    mc= CreateModel()
    model = mc.lstm_layer(5000,nb_categories,nodes=60)
