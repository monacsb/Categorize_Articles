# Categorize_Articles
To categorize unseen articles into 5 categories namely Sport, Tech, Business, Entertainment and Politics.

# Description 

"Text documents are essential as they are one of the richest sources of data for businesses. Text documents often contain crucial information which might shape the market trends or influence the investment flows. Therefore, companies often hire analysts to monitor the trend via articles posted online, tweets on social media platforms such as Twitter or articles from newspaper. However, some companies may wish to only focus on articles related to technologies and politics. Thus, filtering of the articles into different categories is required."

Filtering acticles into 5 categories: <br />
1: Sport <br/>
2: Tech <br/>
3: Business <br/>
4: Entertainment <br/>
5: Politics <br/>

# How to use it 
Clone repo and run it. <br/>
To run the classes execute: unseen_articles.py
To train the model and generate accuracy report execute: train.py

# Requirement
Spyder  <br/>
Python 3.8  <br/>
Windows 10 or even latest version with CPU 2.22GHz and Memory 5GB  <br/> 
Anaconda prompt(anaconda 3)  <br/>

# Results 

The accuracy score has been improved from 65% to 87% by: <br/>
 1: After nodes has been increased from 30 to 60 <br/>
 2: While adding another neural networks extension to LSTM which is BiDirectional <br/>
<img width="359" alt="Accuracy_score_F1_2022_05_19" src="https://user-images.githubusercontent.com/103228610/169289836-803fdb28-6524-4e8a-81e8-5b70e0bd3c4f.png"> <br/>
Below is the tensorboard graph: <br>
As you can see, the below training validation lines show low accuracy and high in loss. But,once i adopted Bidirectional into the layers, my accuracy has increased
and my loss has reduced.<br/>
This shows that, the combination of LSTM and Bidirectional in this has improved the performance. <br/>
<img width="266" alt="graph_tensorboard" src="https://user-images.githubusercontent.com/103228610/169290270-49dd171f-92b2-406d-9635-d83d493585af.png">

# Model 
LSTM
# Credits 
Thanks to Susanli2016 for the dataset <br/>
https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv
