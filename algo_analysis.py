
import numpy as np
import pandas as pd
# from scipy.misc import imread
from matplotlib.pyplot import imread
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import time


df = pd.read_csv('devanagiri_data.csv')
df['character_class'] = LabelEncoder().fit_transform(df.character)
df.drop('character', axis=1, inplace=True)
df = df.astype(np.uint8)
df_sample = df.sample(frac=0.1, random_state=0)
df_train,df_test= train_test_split(df_sample,test_size=0.2)



#Algorithm Selection Using a Dataset Sample

names = ['RidgeClassifier', 'BernoulliNB', 'GaussianNB', 'ExtraTreeClassifier', 'DecisionTreeClassifier',
         'NearestCentroid', 'KNeighborsClassifier', 'ExtraTreesClassifier', 'RandomForestClassifier']
classifiers = [RidgeClassifier(), BernoulliNB(), GaussianNB(), ExtraTreeClassifier(), DecisionTreeClassifier(),
                NearestCentroid(), KNeighborsClassifier(), ExtraTreesClassifier(), RandomForestClassifier()]

test_scores,fit_time, score_time = [], [], []

for clf in classifiers:
    scores = cross_validate(clf, df_sample.iloc[:, :-1], df_sample.iloc[:, -1],return_train_score=True)
    test_scores.append(scores['test_score'].mean())
    fit_time.append(scores['fit_time'].mean())
    score_time.append(scores['score_time'].mean())

selection=pd.DataFrame({'Classifier': names,
              'Test_Score': test_scores,
              'Fit_Time': fit_time,
              'Score_Time': score_time})

print(selection)

"""MODEL NO 1"""
# Time==22 seconds
# K Nearest Neighbors
start=time.time()
print("Training the model using KNN")
parameters = {'n_neighbors': np.arange(1, 22, 4)}
clf = GridSearchCV(KNeighborsClassifier(), parameters)

clf.fit(df_train.iloc[:, :-1], df_train.iloc[:, -1])
end=time.time()
print(f"time took is {end-start}")
result = pd.DataFrame.from_dict(clf.cv_results_)

x, y = clf.best_params_['n_neighbors'], clf.best_score_
text = 'N Neighbors = {}, Score = {}'.format(x, y)

plt.figure()
plt.title('K Nearest Neighbors')
plt.xlabel('No. of Neighbors')
plt.ylabel('Accuracy Score')
plt.yticks(np.arange(0.6, 0.81, 0.02))

plt.plot(result.param_n_neighbors, result.mean_test_score, label='Mean Accuracy Score')
plt.plot(x, y, 'o', label=text)

plt.legend()
plt.show()


"""MODEL NO 2"""
# time == 25 minutes
# Extremely Randomized Trees

start=time.time()
print("Training the model , please wait ........")
parameters = {'n_estimators': np.arange(20, 310, 20)}
clf = GridSearchCV(ExtraTreesClassifier(), parameters)

clf.fit(df_train.iloc[:, :-1], df_train.iloc[:, -1])
result = pd.DataFrame.from_dict(clf.cv_results_)
end=time.time()
print(f"Training is completed in {end-start} seconds")

x, y = clf.best_params_['n_estimators'], clf.best_score_
text = 'No. of Trees = {}, Score = {}'.format(x, y)

plt.figure()
plt.title('Extremely Randomized Trees Classification')
plt.xlabel('No. of Trees')
plt.ylabel('Accuracy Score')
plt.yticks(np.arange(0.6, 0.81, 0.02))

plt.plot(result.param_n_estimators, result.mean_test_score, label='Mean Accuracy Score')
plt.plot(x, y, 'o', label=text)

plt.legend()
plt.show()



"""MODEL NO 3"""
# Random Forests
#time = 30 minutes
parameters = {'n_estimators': np.arange(20, 310, 20)}
clf = GridSearchCV(RandomForestClassifier(), parameters)

start=time.time()
print("Training the model , please wait ........")

clf.fit(df_train.iloc[:, :-1], df_train.iloc[:, -1])
result = pd.DataFrame.from_dict(clf.cv_results_)
end=time.time()
print(f"Training is completed in {end-start} seconds")

x, y = clf.best_params_['n_estimators'], clf.best_score_
text = 'No. of Trees = {}, Score = {}'.format(x, y)

plt.figure()
plt.title('Random Forests Classification')
plt.xlabel('No. of Trees')
plt.ylabel('Accuracy Score')
plt.yticks(np.arange(0.6, 0.81, 0.02))

plt.plot(result.param_n_estimators, result.mean_test_score, label='Mean Accuracy Score')
plt.plot(x, y, 'o', label=text)

plt.legend()
plt.show()


#Learning Curve Using Extremely Randomised Decision Trees Classification
df_sample = df.sample(frac=0.1, random_state=0)

start=time.time()
print("Training the model , please wait ........")

clf = ExtraTreesClassifier(n_estimators=280)

train_sizes, train_scores, test_scores = learning_curve(clf, df_sample.iloc[:, :-1], df_sample.iloc[:, -1])
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

end=time.time()
print(f"Training is completed in {end-start} seconds")

plt.figure()
plt.title('Learning Curve for Extra Trees Classification')
plt.xlabel('Training examples')
plt.ylabel('Score')

# plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

plt.legend()
plt.show()


# Testing the test data using extraTreesClassifier with 256 trees

df_sample = df.sample(frac=0.1, random_state=0)

clf = ExtraTreesClassifier(n_estimators=256)

scores = cross_validate(clf, df_sample.iloc[:, :-1], df_sample.iloc[:, -1])
print('Mean Accuracy Score:', scores['test_score'].mean())
