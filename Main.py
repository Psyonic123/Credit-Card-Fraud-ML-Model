# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, \
    matthews_corrcoef, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Reading our credit card data set
data = pd.read_csv('creditcard.csv')
fraud = data[data["Class"] == 1]
valid = data[data["Class"] == 0]
outlier = len(fraud) / float(len(valid))
# Print sums of our classes
print("Number of fraud cases: ", format(len(data[data["Class"] == 1])))
print("Number of valid cases: ", format(len(data[data["Class"] == 0])))
print("Probability of fraud: ", outlier)
# Print a correlation matrix, attempt to see if any features are closely related
corrMatrix = data.corr()
sns.heatmap(corrMatrix, vmax=.8, square=True)
plt.show()
# dividing the X and the Y from the dataset
X = data.drop(["Class"], axis=1)
Y = data["Class"]
# getting just the values for the sake of processing (its a numpy array with no columns)
X_data = X.values
Y_data = Y.values
# Splitting our data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
# Building a MLP model


clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, Y_train)

# predictions
y_pred = clf.predict(X_test)

# Metrics
acc = confusion_matrix(Y_test, y_pred)


def accuracy(cm):
    diagonal = cm.trace()
    elements = cm.sum()
    return diagonal / elements


print("The accuracy is {}", accuracy(acc))
prec = precision_score(Y_test, y_pred)
print("The precision is {}".format(prec))
rec = recall_score(Y_test, y_pred)
print("The recall is {}".format(rec))
f1 = f1_score(Y_test, y_pred)
print("The F1-Score is {}".format(f1))
MCC = matthews_corrcoef(Y_test, y_pred)
print("The Matthews correlation coefficient is {}".format(MCC))
