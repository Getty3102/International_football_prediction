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
https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-201 7/code

# Pipe Line

● Exploratory Data Analysis

● Data Preprocessing

● Feature Encoding

● Data Splitting

● Model Training and Evaluation

# Exploratory Data Analysis
Exploratory Data Analysis (EDA) is a crucial initial step in the data analysis process. It involves examining and understanding the structure, patterns, and characteristics of the dataset at hand. By exploring the data, we can uncover valuable insights, identify trends, detect anomalies, and gain a deeper understanding of the variables and relationships within the dataset

# Data Preprocessing
The following was done during the data preprocessing stage:

● The data column was dropped was dropped because it was irrelevant as we do not plan to carry out a time series analysis as the time spans are inconsistent.

● In our dataset, the distribution of the target column reveals a class imbalance issue. The home_team_wins class accounts for 48.71% of instances, while the draw and away_team_wins classes comprise 23% and 28.29% respectively. This imbalance may lead the model to overemphasize the trends of home team wins during training, resulting in poor performance on unseen data without similar patterns. To mitigate this, we will address the class imbalance by employing oversampling techniques to increase the representation of the minority classes.

● Data normalization was perfomed to ensure that all features have comparable ranges, preventing any particular feature from dominating the learning process. By bringing the features to a consistent scale, we enable the machine learning model to give equal consideration to each feature, avoiding biases that may arise due to variations in their original scales. Normalization plays a crucial role in creating a balanced learning environment, enhancing the model's ability to learn meaningful patterns and make accurate predictions.

# Model Training and Evaluation
Model Training: Machine learning revolves around understanding the patterns and behaviors exhibited by a dataset and then testing this understanding on new data. To accomplish this, the dataset was divided into three distinct sets: the training dataset and the testing dataset and validation dataset.

Baseline Model : In our project, we have chosen the K-Nearest Neighbors (KNN) algorithm as our baseline model. The KNN algorithm will serve as a foundational model from which we can assess the performance and effectiveness of more advanced techniques or models.The KNN model will provide a reference point to measure the progress and advancements made in our machine learning project.

We observed the following accuracy results: the training accuracy of the KNN model is 0.71, while the validation accuracy is 0.47.
It is evident that the KNN model demonstrates a higher accuracy on the training set compared to the testing set. This discrepancy indicates the presence of overfitting, whereby the model has learned the training data too well and struggles to generalize to new, unseen data.

We also used the Random Forest Classifier and XGBoost Classifier which gave the following results:
Random Forest Classifier: Training Accuracy : 1.0 , Validation Accuracy: 0.98. Which implies that the RF model performs well on training set and testing set.
XGBoost Classifier: Training Accuracy : 1.0 , Validation Accuracy: 1.0
The best performing model is the xgboost with a 100% accuracy on both the train and validation set. For that reason we will use it for our model evaluation.

# Model Evaluation
During this phase we assessed the performance and effectiveness of a machine learning model. It involved measuring how well the model performs on the test dataset or how accurately it can make predictions on new, unseen test data.
The results for the Model Evaluation using XGBoost Classifier which gave the highest accuracy on the training and testing accuracy are;
Training Accuracy : 1.0 Test Accuracy: 0.99.

Which implies that on the unseen test data, the model still performed exceptionally well. This demonstrates that the model has not been overfit


