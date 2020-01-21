import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

import joblib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class DataFrameSelector(BaseEstimator,TransformerMixin):
	def __init__(self,attri_name):
		self.attri_name = attri_name
	def fit(self,X,y=None):
		return self
	def transform(self,X,y=None):
		return X[self.attri_name]

class MostFrequentImputer(BaseEstimator,TransformerMixin):
	def fit(self,X,y=None):
		#self.most_frequent = X.value_counts().index[0]
		self.most_frequent = pd.Series([X[c].value_counts().index[0] for c in X],index=X.columns)
		self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],index=X.columns)
		return self
	def transform(self,X,y=None):
		return X.fillna(self.most_frequent)


data_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
y_train = np.array(data_train['Survived'])

"""
pipeline 
"""
num_pipeline = Pipeline([
	('selector',DataFrameSelector(['Age','SibSp','Parch','Fare'])),
	('imputer',SimpleImputer(strategy='median'))
])

cat_pipeline = Pipeline([
	('selector',DataFrameSelector(['Sex','Embarked','Pclass'])),
	('imputer',MostFrequentImputer()),
	('cat_encoder',OneHotEncoder(sparse=False)),
])

preprocessing_pipeline = FeatureUnion(transformer_list=[
	('num_pipeline',num_pipeline),
	('cat_pipeline',cat_pipeline),
])

X_train = preprocessing_pipeline.fit_transform(data_train)
print(X_train)

svm_clf = SVC(gamma='auto')
svm_clf.fit(X_train,y_train)

score = svm_clf.score(X_train,y_train)
scores_cross_val = cross_val_score(svm_clf,X_train,y_train,cv=10)
print(scores_cross_val.mean())