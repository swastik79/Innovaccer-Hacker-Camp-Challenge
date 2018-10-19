# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_excel('Dataset1.xlsx')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X[:,4] = labelencoder_X.fit_transform(X[:,4])
X[:,5] = labelencoder_X.fit_transform(X[:,5])
X[:,6] = labelencoder_X.fit_transform(X[:,6])
X[:,7] = labelencoder_X.fit_transform(X[:,7])
X[:,8] = labelencoder_X.fit_transform(X[:,8])
X[:,9] = labelencoder_X.fit_transform(X[:,9])
X[:,10] = labelencoder_X.fit_transform(X[:,10])
X[:,11] = labelencoder_X.fit_transform(X[:,11])
X[:,12] = labelencoder_X.fit_transform(X[:,12])
X[:,15] = labelencoder_X.fit_transform(X[:,15])
X[:,17] = labelencoder_X.fit_transform(X[:,17])
X[:,18] = labelencoder_X.fit_transform(X[:,18])
onehotencoder = OneHotEncoder(categorical_features=[0])
X= onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
onehotencoder = OneHotEncoder(categorical_features=[50])
X= onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
onehotencoder = OneHotEncoder(categorical_features=[53])
X= onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
onehotencoder = OneHotEncoder(categorical_features=[63])
X= onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
onehotencoder = OneHotEncoder(categorical_features=[85])
X= onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
onehotencoder = OneHotEncoder(categorical_features=[91])
X= onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
onehotencoder = OneHotEncoder(categorical_features=[102])
X= onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Applying LDA for dimensionality reduction to improve accuracy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components =50)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

#Applying XGBoost Classifier
from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators = 103)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
confidence= classifier.score(X_test,y_test)
print('Accuracy obtained by XGBoost classifier on test set :',confidence)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cm,
                     index = [1,2,3,4,5], 
                     columns = [1,2,3,4,5])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('XGBoost Classifier Accuracy')
plt.ylabel('True value')
plt.xlabel('Predicted value')
plt.show()


# Finding the important parameters of the dataset
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
important_features=model.feature_importances_


