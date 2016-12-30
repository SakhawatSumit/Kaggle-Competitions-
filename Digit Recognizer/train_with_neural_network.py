import numpy as np
import pandas as pd
import csv as csv
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

# Training data
train_df = pd.read_csv('train.csv', header=0)
# Test data
test_df = pd.read_csv('test.csv', header=0)
# Convert back to numpy array
train_data = train_df.values
test_data = test_df.values
# First column of dataset is target value & rest of the column is training value
# Separate them into another variables.
temp = range(1, 785)
x = np.array(train_data[:, temp])
y = np.array(train_data[:, 0])

print "Training..."
clf = (MLPClassifier(solver='adam', alpha=1e-5,
                     hidden_layer_sizes=(800, 800), random_state=1))
""""
scores = cross_val_score(clf, x, y, cv=10, scoring="accuracy")
print "Cross validation mean score..."
print scores.mean()
"""
clf = clf.fit(X=x, y=y)
print "Predicting..."
output = clf.predict(test_data).astype(int)
print "Score..."
print clf.score(X=x, y=y)

ids = range(1, 28001)
predictions_file = open("testresult.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId","Label"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()

print 'Done.'