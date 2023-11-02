#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


file1 = pd.read_csv('airlines.csv')
file2 = pd.read_csv('airports.csv')
file3 = pd.read_csv('flights.csv')



# In[3]:


# Read the CSV files
file1 = pd.read_csv('airlines.csv')
file2 = pd.read_csv('airports.csv')
file3 = pd.read_csv('flights.csv')

# Function to handle missing values in a DataFrame
def handle_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            # For categorical columns, fill missing values with the mode (most frequent value)
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            # For numerical columns, fill missing values with the mean
            df[column].fillna(df[column].mean(), inplace=True)

# Handle missing values for all DataFrames
handle_missing_values(file1)
handle_missing_values(file2)
handle_missing_values(file3)


# #### How can we develop a machine learning model that accurately predicts the likelihood of flight delays and cancellations based on historical flight data, weather conditions, airport information, and other relevant factors, with the goal of improving passenger satisfaction and airline operations?
# 

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Define X (features) and y (target variable)

X = file3[['DEPARTURE_DELAY', 'ARRIVAL_TIME', 'SCHEDULED_DEPARTURE', 'WEATHER_DELAY']]  
y = file3['CANCELLED'] 

# Handle missing values in X by filling NaN values with the mean of each column
X.fillna(X.mean(), inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# In[5]:


#KNN

from sklearn.neighbors import KNeighborsClassifier

# Create a KNN classifier with a specified number of neighbors (adjust n_neighbors as needed)
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Fit the KNN classifier on the training data
knn_classifier.fit(X_train, y_train)

# Make predictions on the test data
knn_predictions = knn_classifier.predict(X_test)

# Calculate the accuracy of the KNN model
knn_accuracy = knn_classifier.score(X_test, y_test)

print("KNN Accuracy:", knn_accuracy)


# In[6]:


#Random Forest:

from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest classifier (adjust parameters as needed)
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the Random Forest classifier on the training data
random_forest_classifier.fit(X_train, y_train)

# Make predictions on the test data
random_forest_predictions = random_forest_classifier.predict(X_test)

# Calculate the accuracy of the Random Forest model
random_forest_accuracy = random_forest_classifier.score(X_test, y_test)

print("Random Forest Accuracy:", random_forest_accuracy)


# In[7]:


#Decision Tree:

from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree classifier ( adjust parameters as needed)
decision_tree_classifier = DecisionTreeClassifier(random_state=42)

# Fit the Decision Tree classifier on the training data
decision_tree_classifier.fit(X_train, y_train)

# Make predictions on the test data
decision_tree_predictions = decision_tree_classifier.predict(X_test)

# Calculate the accuracy of the Decision Tree model
decision_tree_accuracy = decision_tree_classifier.score(X_test, y_test)

print("Decision Tree Accuracy:", decision_tree_accuracy)


# In[8]:


from sklearn.metrics import confusion_matrix

# Confusion matrix for KNN model
knn_confusion_matrix = confusion_matrix(y_test, knn_predictions)
print("Confusion Matrix for KNN Model:")
print(knn_confusion_matrix)

# Confusion matrix for Random Forest model
random_forest_confusion_matrix = confusion_matrix(y_test, random_forest_predictions)
print("\nConfusion Matrix for Random Forest Model:")
print(random_forest_confusion_matrix)

# Confusion matrix for Decision Tree model
decision_tree_confusion_matrix = confusion_matrix(y_test, decision_tree_predictions)
print("\nConfusion Matrix for Decision Tree Model:")
print(decision_tree_confusion_matrix)


# #### Based on these metrics, all three models (KNN, Random Forest, and Decision Tree) are performing exceptionally well with high accuracy and similar precision and recall scores. The differences are minimal, so choosing the best model may also depend on other factors like computational resources, model interpretability, and the specific requirements of your application.

# In[9]:


import pickle

# Save the KNN model to a .pkl file
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn_classifier, file)

# Save the Random Forest model to a .pkl file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(random_forest_classifier, file)

# Save the Decision Tree model to a .pkl file
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(decision_tree_classifier, file)


# In[14]:


# Load the KNN model
with open('knn_model.pkl', 'rb') as file:
    knn_loaded_model = pickle.load(file)

# Load the Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    random_forest_loaded_model = pickle.load(file)

# Load the Decision Tree model
with open('decision_tree_model.pkl', 'rb') as file:
    decision_tree_loaded_model = pickle.load(file)


# ##### 
# This solution focuses on the development of a predictive model that can help airlines, airports, and passengers anticipate and mitigate potential flight delays and cancellations, ultimately leading to a better travel experience and operational efficiency. It's a practical and relevant question for the aviation industry.
# 
# 
# 
