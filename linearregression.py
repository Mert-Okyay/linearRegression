import numpy as np
import sklearn
from sklearn import linear_model as lm
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import pandas as pd
from matplotlib import pyplot, style
import pickle

# read the grades data
data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head())

# select desired attributes
data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]
# print(data.head())


predict = "G3"

# define axes
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# split data set into training and testing samples
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


best = 0

# create linear regression model
'''for m in range(100):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    linear = lm.LinearRegression()

    # create best fit line
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if best < accuracy:
        best = accuracy
        with open("studentmodelBest.pickle", "wb" ) as f:
            pickle.dump(linear, f)'''

pickle_in = open("studentmodelBest.pickle", "rb")

linear = pickle.load(pickle_in)

accuracy = linear.score(x_test, y_test)
print (accuracy)
print("co: ", linear.coef_)
print("interccept: ", linear.intercept_)

predictions = linear.predict(x_test)

# print results
for x in range(len(predictions)):
    grade = predictions[x]
    actualGrade = y_test[x]
    error = abs(grade - actualGrade)
    percError = error / actualGrade
    print(grade, x_test[x], actualGrade, round(error, 2), round(percError, 2), "%")

p = "absences"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel( "Final Grade")
pyplot.show()
