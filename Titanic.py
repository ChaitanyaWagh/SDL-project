# Importing Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset

test = pd.read_csv('test_clean.csv')
train = pd.read_csv('train_clean.csv')

name = test['Name']

X_train = train.drop(columns = ['Cabin','Embarked','Fare','Name','Parch','PassengerId','SibSp','Ticket','Title','Family_Size','Survived'])
y_train = train['Survived']
X_test = test.drop(columns = ['Name','Survived'])

# Classification Algo *Support Vector Machine*

from sklearn.svm import SVC
svc = SVC()

svc.fit(X_train, y_train)

SVMPrediction = svc.predict(X_test)

# Classification Algo *Decision Tree*

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

DTPrediction = clf.predict(X_test)

# Printing Accuracy

print(f"Accuracy of SVM = {(svc.score(X_train, y_train) * 100)} %")
print(f"Accuracy of Decision Tree = {(clf.score(X_train, y_train) * 100)} %")

# Visualization based on Feature: Gender

fig, ax = plt.subplots(1, 2, figsize = (18, 8))
train["Sex"].value_counts().plot.bar(color = "skyblue", ax = ax[0])
ax[0].set_title("Number Of Passengers By Sex")
ax[0].set_ylabel("Population")
sns.countplot("Sex", hue = "Survived", data = train, ax = ax[1])
ax[1].set_title("Sex: Survived vs Dead")
plt.show()

# Visualization based on Feature: Age

def ageG2int(data):
    data["Age_group"] = "NaN"
    data.loc[data["Age"] <= 16, "Age_group"] = 0 # Child
    data.loc[(data["Age"] > 16) & (data["Age"] <= 32), "Age_group"] = 1 # young teen and teen adult
    data.loc[(data["Age"] > 32) & (data["Age"] <= 48), "Age_group"] = 3 # middle age
    data.loc[(data["Age"] > 48) & (data["Age"] <= 64), "Age_group"] = 4 # young elderly
    data.loc[data["Age"] > 64, "Age_group"] = 5 # elderly
    return data

train = ageG2int(train)
sns.countplot("Age_group", hue = "Survived", data = train)
plt.show()

# Visualization based on Feature: Pclass

fig, ax = plt.subplots(1, 2, figsize = (18, 8))
train["Pclass"].value_counts().plot.bar(color = "skyblue", ax = ax[0])
ax[0].set_title("Number Of Passengers By Pclass")
ax[0].set_ylabel("Population")
sns.countplot("Pclass", hue = "Survived", data = train, ax = ax[1])
ax[1].set_title("Pclass: Survived vs Dead")
plt.show()

# Saving Result in Result.csv
submission = pd.DataFrame({
                           "Name": name,
                           "Survived by SVM": SVMPrediction,
                           "Survived by DT": DTPrediction
                           })
submission.to_csv('Result.csv', index=False)
