# International_football_prediction

# Background
Predicting the outcome of football matches has long been a subject of great interest and excitement among football enthusiasts, sports analysts, and betting enthusiasts. The ability to accurately forecast match results not only adds to the thrill of the game but also holds significant implications for various stakeholders involved.
The goal is to develop a prediction algorithm that can leverage historical data and current information to provide reliable insights into the probable outcome of future football matches. Such an algorithm would enable users to make informed decisions, whether for betting purposes, fantasy league competitions, or simply for enhancing their understanding and enjoyment of the sport.

# Team Members:

Adeniyi Olaolu Peter
ID: 16ecc37e2141f000

Ogunfuyi Taiwo Hassan 
ID: 1687e659be01f000

Semiu Biliaminu
ID: 16e7d4e4bcc1f000

Favour Sukat
ID: 16ee43e95341f000

Lukmon Kazeem
ID: 1487be7dc401f000

 Obiageli Jessica Okoli
16eb60ec64c1f000

 Emmanuel Mugabo
ID: 16dca05911c1f000

 Chinwendu Nweje
ID: 16bdd8696c01f000

 Chidozie Ahamefule 
ID: 11a24d9f3801f000

Morka esther eberechukwu 
ID: 16bfcdf4e141f000

AMADI CHIMEREMMA SANDRA
ID: 1572c1a4b801f000

VALENTINE NTHIGAH       
ID: 16fc514b8a01f000

Joseph Nwogwugwu 
ID:  16edea41fd41f000

Getrude Obwoge  
ID: 1691b4416e41f000


# Requirements

● Google COLAB

● Python Libraries;

● Numpy

● Pandas

● Pandas_profiling

● Matplotlib.pyplot

● Seaborn

● Plotly.express


# Motivation
There are many machine learning approaches for game prediction, however, we believe the XGBoost Classifier could be very helpful in this scenario leveraging historical data and advanced analytical techniques. The goal is to provide users with reliable insights into future match results, enhancing their understanding, enjoyment, and decision-making in relation to football matches

# Data Description
Data Set with the football matches of the International football matches has been created with the aim of opening a line of research in the Machine Learning, for the prediction of results of football matches from Kaggle

# Data Source
The dataset was obtained from Kaggle via the link :
https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017

# Data Pipeline

● Exploratory Data Analysis

● Data Preprocessing

● Feature Encoding

● Data Splitting

● Model Training and Evaluation

# Model Training and Evaluation
Model Training: Machine learning revolves around understanding the patterns and behaviors exhibited by a dataset and then testing this understanding on new data. To accomplish this, the dataset was divided into three distinct sets: the training dataset and the testing dataset and validation dataset.

Baseline Model : In our project, we have chosen the K-Nearest Neighbors (KNN) algorithm as our baseline model. 
We observed the following accuracy results: the training accuracy of the KNN model is 0.71, while the validation accuracy is 0.47.
The best performing model is the xgboost with a 100% accuracy on both the train and validation set. For that reason we will use it for our model evaluation.

# Model Evaluation
During this phase we assessed the performance and effectiveness of a machine learning model. It involved measuring how well the model performs on the test dataset or how accurately it can make predictions on new, unseen test data.
The results for the Model Evaluation using XGBoost Classifier which gave the highest accuracy on the training and testing accuracy are;
Training Accuracy : 1.0 Test Accuracy: 0.99.

Which implies that on the unseen test data, the model still performed exceptionally well. This demonstrates that the model has not been overfit.


