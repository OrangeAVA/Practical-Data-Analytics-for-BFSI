import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Loading the dataset
data = pd.read_csv('Social_data.csv')
#If the data-set is not in the active directory, place the complete file path before the file name.

#Data Preprocessing
def create_Sentiment(Rating):
    if Rating == 1 or Rating == 2:
        return -1
    elif Rating == 3 or Rating == 4:
        return +1
    else: 
        return 0

df['Sentiments'] = df[Rating].apply(create_Sentiment)

#Split the data into training and testing sets
y = data['Sentiment']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=33)

#Training Logistic Regression Classifier Model 
lrm = LogisticRegression(solver='liblinear')
lrm.fit(x_train, y_train)

#Prediction of Sentiments using the test data set
y_pred = lrm.predict(x_test)
#Evaluating Model Accuracy 
Accuracy = accuracy_score(y_test, y_pred)
print(Accuracy)

Performance_Report = classification_report(Y_test, predicted)
print(Performance_Report)
