import pandas as pd
import numpy as np
import utils
from sklearn import tree, model_selection

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
utils.clean_data(train)
utils.clean_data(test)

target = train["Survived"].values
feature_names = ["Pclass","Age","Fare","Embarked","Sex", "SibSp", "Parch"]
features = train[feature_names].values

decision_tree= tree.DecisionTreeClassifier(random_state=1)
decision_tree_=decision_tree.fit(features,target)

print(decision_tree_.score(features,target))

scores = model_selection.cross_val_score(decision_tree_,features,target,scoring='accuracy',cv=50)
print(scores)
print(scores.mean())

generalized_tree= tree.DecisionTreeClassifier(random_state=1,max_depth=7,min_samples_split=2)
generalized_tree_=generalized_tree.fit(features,target)

print(generalized_tree_.score(features,target))

scores = model_selection.cross_val_score(generalized_tree,features,target,scoring='accuracy',cv=50)
print(scores)
print(scores.mean())

tree.export_graphviz(generalized_tree_,feature_names=feature_names,out_file="tree.dot")
