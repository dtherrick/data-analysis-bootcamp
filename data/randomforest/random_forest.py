import pandas as pd
import numpy as np
import sqlite3 as lite
import matplotlib.pyplot as plt
import sklearn.ensemble as sk
import pylab as pl
import sklearn.metrics as skm


xt = []
yt = []
features = []
activity = {1: "walking", 2: "walking_upstairs", 3: "walking_downstairs", 4: "sitting", 5: "standing", 6: "laying"}
c = []
feat = open('UCI HAR Dataset/features2.txt', 'r')
for i,r in enumerate(feat):
    c.append("x" + str(i))
    features.append(r.split()[1])

dex = []
subj = open('UCI HAR Dataset/train/subject_train.txt', 'r')
for line in subj:
    dex.append(int(line))
xtrain = open('UCI HAR Dataset/train/X_train.txt', 'r')
for line in xtrain:
    #df.loc[i] = np.array([line.rstrip().split()])
    xt.append(map(lambda x: float(x) , line.split()))

ytrain = open('UCI HAR Dataset/train/y_train.txt', 'r')
for line in ytrain:
    yt.append(activity[int(line.rstrip())])
df = pd.DataFrame(xt, columns=c)
df['p_ID'] = dex
df['activity'] = yt

training_set = df.loc[df['p_ID'] >= 27]
train_feat = training_set.columns[:-2]
X = training_set[train_feat]
Y = training_set['activity']
rf = sk.RandomForestClassifier(n_estimators=50, oob_score=True)
rf.fit(X, Y)

# The oob score is a measure of accuracy for the random forest model to the training set.
print "oob score for the random forest model: ", rf.oob_score_

test_set = df.loc[df['p_ID'] <= 6]
testX = test_set[train_feat]
testY = test_set['activity']
print "mean accuracy score for the RF model on test data: ", rf.score(testX, testY)
print "test (prediction): ", rf.predict(testX)

# Find the 10 most important features for prediction in this model.
for n in range(50):
    z = (n+1)/1000.0 + .010
    important_features = [(features[i], n) for i, n in enumerate(rf.feature_importances_) if n >= z]
    if len(important_features) <= 10:
        print "the top %s features in the random forest model:" % (len(important_features)) 
        print important_features
        break

# Validating the model with participants 22-27.
validation_set = df.loc[(df["p_ID"] > 21) & (df["p_ID"] < 27)]
validationX = validation_set[train_feat]
validationY = validation_set['activity']
print "Mean accuracy score for the test set: ", rf.score(testX, testY)
print "Mean accuracy score for validation set: ", rf.score(validationX, validationY)

test_pred = rf.predict(testX)
test_cm = skm.confusion_matrix(testY, test_pred)

pl.matshow(test_cm)
pl.title('Confusion matrix for test data')
pl.colorbar()
pl.show()

# Precision score
print "Precision: %f" % (skm.precision_score(testY, test_pred))
# Accuracy score
print "Accuracy: %f" % (skm.accuracy_score(testY, test_pred))
# Recall score
print "Recall: %f" % (skm.recall_score(testY, test_pred))
# F1 score
print "F1: %f" % (skm.f1_score(testY, test_pred))