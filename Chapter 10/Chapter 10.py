#Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#Loading the dataset
data = pd.read_csv('MF_data.csv')
#If the data-set is not in the active directory, place the complete file path before the file name.

#Split the data into training and testing sets
X = data.drop('MF_Status', axis=1)
Y = data['MF_Status']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=33)

#Training Random Forest Model 
Model_RF = RandomForestClassifier(n_estimators=100, random_state=33)
Model_RF(x_train, y_train)

#Prediction of Mutual Fund using the test data set
y_pred = Model_RF.predict(x_test)

#Evaluating Model Accuracy and Classification Report
RF_Accuracy = accuracy_score(y_test, y_pred)
print(RF_Accuracy)
Performance_Report = classification_report(Y_test, predicted)
print(Performance_Report)
The final outputs will provide the model performance in terms of accuracy, precision, recall, and f1-score.
