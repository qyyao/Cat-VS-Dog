import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Use sklearn pre-defined dataset load_iris:
from sklearn.datasets import load_iris

iris = load_iris()

# Dataset consists of 3 different types of Irises: Setosa, Versicolour, Virginica and their petal and sepal length
# Form: 150 x 4 numpy.ndarray
# Rows: samples
# Columns: Sepal Length, Sepal Width, Petal Length, Petal Width

# Goal: Determine whether flower is Setosa or not

# STEP 1: separate petal length and width into variable X

X = iris.data[:, (2, 3)]  # petal length, petal width

# Since our target is Setosa, we want our target array to hold 1 for Setosa and 0 for everyone else

y = (iris.target == 0)  # find where Setosa is and convert if it is true, since its value is currently 0
y = (iris.target == 0).astype(np.int)  # convert true/false to type int 0 and 1

# Machine Learning Algorithm:
# 1) Create object of algorithm
# 2) Fit x and y into object
# 3) Use object to predict future values of y

per_clf = Perceptron(random_state=42)  # perceptron classifier
per_clf.fit(X, y)  # give model to object

y_pred = per_clf.predict(X)  # Now use our object to predict values!

# check our results:
print("predicted values are: ", y_pred)  # output predicted values
print("accuracy score is:", accuracy_score(y, y_pred))  # compare the accuracy of our model
print("co-efficients are: ", per_clf.coef_)  # co-efficient of line, AKA impact value
print("intercept is:", per_clf.intercept_)  # intercept of line
