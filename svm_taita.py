# coding: utf-8
import itertools
import os
from skimage import data, color, exposure
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import pandas as pd
from sklearn import svm
import seaborn as sns
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import gc
from sklearn import datasets
from sklearn import preprocessing

df = pd.read_csv("train.csv")
df.Embarked = df.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
df.Sex = df.Sex.replace(['male', 'female'], [0, 1])
df.Age = df.Age.replace('NaN', 0)
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df = df.fillna(0)
print(df)

Y = df['Survived'].values
X = df
X.drop('Survived', axis=1, inplace=True)
X = X.values.astype(np.int64)

X = np.array(X)
Y = np.array(Y)
# np.savetxt("X_data.csv", X, fmt="%f", delimiter=",")

# 主成分分析する
pca = PCA(copy = False, n_components = 2)
pca.fit(X)
for s in range(len(pca.singular_values_)):
    print(s)
    print(pca.singular_values_[s])

# 分析結果を元にデータセットを主成分に変換する
X = pca.fit_transform(X)
print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))
del pca
gc.collect()
print(X.shape)
print(Y.shape)
print('X', X)
print('y', Y)

tuned_parameters = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
print('start Grid Search')
gscv = GridSearchCV(svm.LinearSVC(), tuned_parameters)
gscv.fit(X, Y)
GS_C = gscv.best_estimator_.C
tuned_parameters_second = {'C': [GS_C, 2*GS_C, 3*GS_C, 4*GS_C, 5*GS_C, 6*GS_C, 7*GS_C, 8*GS_C, 9*GS_C]}
gscv = GridSearchCV(svm.SVC(), tuned_parameters_second)
gscv.fit(X, Y)
svm_best = gscv.best_estimator_


print('searched result of  C =', svm_best.C)

# 最適(?)なパラメータを用いたSVMの再学習
print('start re-learning SVM with best parameter set.')
svm_best.fit(X, Y)

# 学習結果の保存
print('finish learning SVM　with Grid-search.')
joblib.dump(svm_best, 'taita.pkl', compress=9)


df_test = pd.read_csv("test.csv")

# 不要カラムの削除
df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 文字列の数値置換
df_test.Embarked = df_test.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
df_test.Sex = df_test.Sex.replace(['male', 'female'], [0, 1])
df_test.Age = df_test.Age.replace('NaN', 0)
df_test = df_test.fillna(0)
print(df_test)

X_test = df_test
X_test = X_test.values.astype(np.int64)
X_test = np.array(X_test)

pca_test = PCA(copy = False, n_components = 2)
pca_test.fit(X_test)
X_test = pca_test.fit_transform(X_test)

Y_test_pred = svm_best.predict(X_test)

df_out = pd.read_csv("test.csv")
df_out["Survived"] = Y_test_pred

# outputディレクトリに出力する
df_out[["PassengerId","Survived"]].to_csv("submission.csv",index=False)

# トレーニングデータに対する精度
pred_train = svm_best.predict(X)
accuracy_train = accuracy_score(Y, pred_train)
print('トレーニングデータに対する正解率： %.2f' % accuracy_train)

# テストデータに対する精度
# pred_test = svm_best.predict(X_test)
# accuracy_test = accuracy_score(Y_test, pred_test)
# print('テストデータに対する正解率： %.2f' % accuracy_test)
