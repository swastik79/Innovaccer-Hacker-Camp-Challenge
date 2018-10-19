# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_excel('Dataset2.xlsx')
X = dataset.iloc[:,1:10].values
y = dataset.iloc[:,-1].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='median',axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Fitting the training set with RandomForest Classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100,criterion = 'entropy',max_depth=4,random_state=52)
classifier.fit(X_train,y_train)
confidence = classifier.score(X_test,y_test)
print("Initial accuracy before tuning the hyperparameters:", confidence)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Plotting confusion matrix on initial hyperparameters
cm_df = pd.DataFrame(cm,index = [2,4], columns = [2,4])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Random Forest Classifier Accuracy')
plt.ylabel('True value')
plt.xlabel('Predicted value')
plt.show()



#Increasing the efficiency by tuning the hyperparametrs using GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[100,200,210],
               'criterion':['gini','entropy'],
                'max_features':[1,2,3],'max_depth':[2,3,4]}]
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,
                           scoring = 'accuracy',cv =4,n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
confidence1 = grid_search.best_score_
gs = grid_search.best_estimator_
y_pred1 = gs.predict(X_test)
cm1 = confusion_matrix(y_test, y_pred1)
print("Final accuracy after tuning the hyperparameters:", confidence1)

#Plotting confusion matrix after tuning the hyperparameters

cm_df1 = pd.DataFrame(cm1,index = [2,4], columns = [2,4])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df1, annot=True)
plt.title('Random Forest Classifier Accuracy')
plt.ylabel('True value')
plt.xlabel('Predicted value')
plt.show()
