import matplotlib.pyplot as plt
#%matplotlib inline
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, cross_validation, tree, preprocessing, metrics
import sklearn.ensemble as ske
import tensorflow as tf
#from tensorflow.contrib import skflow
from tensorflow.contrib import learn as skflow


# importing excel
titanic_df = pd.read_excel('train.xls', 'train', index_col=None, na_values=['NA'])
titanic_df_test = pd.read_excel('test.xls', 'test', index_col=None, na_values=['NA'])

# for first five entries
#print("first five entries")
#print(titanic_df.head())
#print(titanic_df_test.head())

#print ("mean is----")
#print(titanic_df['survived'].mean())
#print(titanic_df_test['survived'].mean())

#print("grouped by pclass")
#print(titanic_df.groupby('pclass').mean())

print("grouping by pclass and sex")

class_sex_grouping = titanic_df.groupby(['pclass','sex']).mean()

print(class_sex_grouping)
print(titanic_df_test.groupby(['pclass','sex']).mean())

#print(titanic_df.count())


# preparing data to remove blank fields
titanic_df = titanic_df.drop(['PassengerId','name','Ticket','Embarked','cabin'], axis=1)
titanic_df_test = titanic_df_test.drop(['name','Ticket','Embarked','cabin'],axis=1)
#titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
titanic_df = titanic_df.dropna()
titanic_df_test = titanic_df_test.dropna()
print(titanic_df.count())
print(titanic_df_test.count())

# only for titanic_df_test, since there is a missing "Fare" values
titanic_df_test["Fare"].fillna(titanic_df_test["Fare"].median(), inplace=True)

# formatting data according to the need

#categorising on the basis of fare
titanic_df.loc[ titanic_df['Fare'] <= 7.91, 'Fare'] = 0
titanic_df.loc[(titanic_df['Fare'] > 7.91) & (titanic_df['Fare'] <= 14.454), 'Fare'] = 1
titanic_df.loc[(titanic_df['Fare'] > 14.454) & (titanic_df['Fare'] <= 31), 'Fare'] = 2
titanic_df.loc[ titanic_df['Fare'] > 31, 'Fare'] = 3
titanic_df_test.loc[ titanic_df_test['Fare'] <= 7.91, 'Fare'] = 0
titanic_df_test.loc[(titanic_df_test['Fare'] > 7.91) & (titanic_df_test['Fare'] <= 14.454), 'Fare'] = 1
titanic_df_test.loc[(titanic_df_test['Fare'] > 14.454) & (titanic_df_test['Fare'] <= 31), 'Fare'] = 2
titanic_df_test.loc[titanic_df_test['Fare'] > 31, 'Fare'] = 3

titanic_df['Fare'] = titanic_df['Fare'].astype(int)
titanic_df_test['Fare']    = titanic_df_test['Fare'].astype(int)


# age impute

titanic_df['age'] = titanic_df.groupby(['pclass'])['age'].transform(lambda x: x.fillna(x.mean()))
titanic_df_test['age'] = titanic_df_test.groupby(['pclass'])['age'].transform(lambda x: x.fillna(x.mean()))


# convert from float to int
titanic_df['age'] = titanic_df['age'].astype(int)
titanic_df_test['age']    = titanic_df_test['age'].astype(int)

titanic_df.loc[ titanic_df['age'] <= 16, 'age'] = 0
titanic_df.loc[(titanic_df['age'] > 16) & (titanic_df['age'] <= 32), 'age'] = 1
titanic_df.loc[(titanic_df['age'] > 32) & (titanic_df['age'] <= 48), 'age'] = 2
titanic_df.loc[(titanic_df['age'] > 48) & (titanic_df['age'] <= 64), 'age'] = 3
titanic_df.loc[(titanic_df['age'] > 64), 'age'] = 4

titanic_df_test.loc[ titanic_df_test['age'] <= 16, 'age'] = 0
titanic_df_test.loc[(titanic_df_test['age'] > 16) & (titanic_df_test['age'] <= 32), 'age'] = 1
titanic_df_test.loc[(titanic_df_test['age'] > 32) & (titanic_df_test['age'] <= 48), 'age'] = 2
titanic_df_test.loc[(titanic_df_test['age'] > 48) & (titanic_df_test['age'] <= 64), 'age'] = 3
titanic_df_test.loc[(titanic_df_test['age'] > 64), 'age'] = 4


# Family

# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]
titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0

titanic_df_test['Family'] =  titanic_df_test["Parch"] + titanic_df_test["SibSp"]
titanic_df_test['Family'].loc[titanic_df_test['Family'] > 0] = 1
titanic_df_test['Family'].loc[titanic_df_test['Family'] == 0] = 0

# drop Parch & SibSp
titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
titanic_df_test    = titanic_df_test.drop(['SibSp','Parch'], axis=1)



sexes = sorted(titanic_df['sex'].unique())
genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))
titanic_df['sex'] = titanic_df['sex'].map(genders_mapping).astype(int)
titanic_df_test['sex'] = titanic_df_test['sex'].map(genders_mapping).astype(int)


X_train = titanic_df.drop(['survived'], axis=1)
y_train = titanic_df['survived']

X_test=titanic_df_test.drop("PassengerId",axis=1).copy()
#X_train, y_train = cross_validation.train_test_split(X,y)
#X_test,y_test
clf_dt = tree.DecisionTreeClassifier(max_depth=10)

clf_dt.fit (X_train, y_train)
y_pred=clf_dt.predict(X_test)
print(y_pred)
print(clf_dt.score (X_train,y_train))


#shuffle_validator = cross_validation.ShuffleSplit(len(X), n_iter=20, test_size=0.2, random_state=0)

#test_classifier(clf_dt)
'''
tf_clf_dnn = skflow.TensorFlowDNNClassifier(hidden_units=[20, 40, 20], n_classes=2, batch_size=256, steps=1000, learning_rate=0.05)
tf_clf_dnn.fit(X_train, y_train)
tf_clf_dnn.score(X_test, y_test)

tf_clf_c = skflow.TensorFlowEstimator(model_fn=custom_model, n_classes=2, batch_size=256, steps=1000, learning_rate=0.05)
tf_clf_c.fit(X_train, y_train)
metrics.accuracy_score(y_test, tf_clf_c.predict(X_test))
'''