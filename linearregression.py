import numpy as np
import sklearn
from sklearn import linear_model as lm
from sklearn.utils import shuffle
import pandas as pd

data = pd.read_csv("student-mat.csv", sep=";")

#print(data.head())
data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]
#print(data.head())

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

linear = lm.LinearRegression()

linear.fit(x_train, y_train)

accuracy = linear.score(x_test, y_test)

print(accuracy)

print("co: ", linear.coef_)
print("interccept: ",linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    grade = predictions[x]
    actualGrade = y_test[x]
    error = abs(grade - actualGrade)
    percError = error / actualGrade
    print(grade, x_test[x], actualGrade , round(error, 2), round(percError, 2),"%")