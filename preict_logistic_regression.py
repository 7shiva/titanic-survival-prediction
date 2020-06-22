import pandas as pd
import utils
from sklearn import linear_model, preprocessing

train=pd.read_csv("train.csv")
utils.clean_data(train)

target = train["Survived"].values
feature_names = ["Pclass","Age","Fare","Embarked","Sex", "SibSp", "Parch"]
features = train[feature_names].values

classifier =linear_model.LogisticRegression()
classifier_=classifier.fit(features,target)
print(classifier_.score(features,target))

poly = preprocessing.PolynomialFeatures(degree=2)
poly_features=poly.fit_transform(features)

classifier_=classifier.fit(poly_features,target)
print(classifier_.score(poly_features,target))
