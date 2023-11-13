import pandas as pd
import numpy as np
from sklearn import tree
from six import StringIO
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os

# create a pandas dataframe called "training" from the titanic-train.csv file
training = pd.read_csv("./Data/train.csv")

training["sex"] = training["sex"].apply(lambda toLabel: 0 if toLabel == "male" else 1)

training["age"].fillna(training["age"].mean(), inplace=True)


print(training.info())

y_target = training["survived"].values

columns = ["fare", "pclass", "sex", "age", "sibsp"]
X_input = training[list(columns)].values

# create clf_train as a decision tree classifier object
clf_train = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)

# train the model using the fit() method of the decision tree object.
# Supply the method with the input variable X_input and the target variable y_target
clf_train = clf_train.fit(X_input, y_target)

print(
    "percentage accuracy of the assignments made by the classifier {}".format(
        clf_train.score(X_input, y_target)
    )
)

with open("./Data/titanic.dot", "w") as f:
    f = tree.export_graphviz(clf_train, out_file=f, feature_names=columns)

os.system("dot -Tpng ./Data/titanic.dot -o ./Data/titanic.png")


image = mpimg.imread("./Data/titanic.png")
plt.imshow(image)
plt.show()


testing = pd.read_csv("./Data/test.csv")
testing["sex"] = testing["sex"].apply(lambda toLabel: 0 if toLabel == "male" else 1)

testing["age"].fillna(testing["age"].mean(), inplace=True)
testing["fare"].fillna(testing["fare"].mean(), inplace=True)

print(testing.info())

X_input = testing[list(columns)].values

target_labels = clf_train.predict(X_input)

# convert the target array into a pandas dataframe using the pd.DataFrame() method and target as argument
target_labels = pd.DataFrame({"Est_Survival": target_labels, "name": testing["name"]})


all_data = pd.read_csv("./Data/titanic.csv")
all_data["sex"] = all_data["sex"].apply(lambda toLabel: 0 if toLabel == "male" else 1)

all_data["age"].fillna(all_data["age"].mean(), inplace=True)


testing_results = pd.merge(target_labels, all_data[["name", "survived"]], on=["name"])

acc = np.sum(testing_results["Est_Survival"] == testing_results["survived"]) / float(
    len(testing_results)
)
print("model accuracy is {}".format(acc))
