#%%
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn import svm
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
X=iris.data
y=iris.target
# print(iris)
pca2 = PCA(n_components = 2) # 主成分分析，保留前2個eigenvectors
X_2 = pca2.fit_transform(X) # 將X降低為2維數據
# 拆分訓練與測試資料，train : test = 80% : 20%
X_train, X_test, y_train, y_test = train_test_split(X_2, y,test_size=0.2,random_state=0)

#%%
print('=========================3rd order polynomial function===========================')

clf3=svm.SVC(kernel='poly', C=1, degree=3) # 建立模型，kernel選用degree=3的polynomial
clf3.fit(X_train,y_train)

y_pred_3 = clf3.predict(X_test) # 測試資料放入模型預測

# Precision Score = TP / (FP + TP)
# Recall Score = TP / (FN + TP)
# F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score)

print('precision:', precision_score(y_test, y_pred_3, average='weighted'))
print('recall:', recall_score(y_test, y_pred_3, average='weighted'))
print('F1-score:', f1_score(y_test, y_pred_3, average='weighted'))

plot_decision_regions(X_train, y_train, clf=clf3, legend=2) # 畫平面圖
plt.xlabel('1st eigenvector')
plt.ylabel('2nd eigenvector')
plt.title('SVM on Iris with 3rd order polynomial function')
plt.show()

#%%
print('=========================10th order polynomial function===========================')

clf10=svm.SVC(kernel='poly', C=1, degree=10) # 建立模型，kernel選用degree=10的polynomial
clf10.fit(X_train,y_train)

y_pred_10 = clf10.predict(X_test)

print('precision:', precision_score(y_test, y_pred_10, average='weighted'))
print('recall:', recall_score(y_test, y_pred_10, average='weighted'))
print('F1-score:', f1_score(y_test, y_pred_10, average='weighted'))

plot_decision_regions(X_train, y_train, clf=clf3, legend=2)
plt.xlabel('1st eigenvector')
plt.ylabel('2nd eigenvector')
plt.title('SVM on Iris with 10th order polynomial function')
plt.show()

#%%
print('=========================RBF with Gamma = 2^-5====================================')

clfrbf=svm.SVC(kernel='rbf', C=1, gamma=2**-5) # 建立模型，kernel選用gamma=2^-5的rbf
clfrbf.fit(X_train,y_train)

y_pred_rbf = clfrbf.predict(X_test)

print('precision:', precision_score(y_test, y_pred_rbf, average='weighted'))
print('recall:', recall_score(y_test, y_pred_rbf, average='weighted'))
print('F1-score:', f1_score(y_test, y_pred_rbf, average='weighted'))

plot_decision_regions(X_train, y_train, clf=clf3, legend=2)
plt.xlabel('1st eigenvector')
plt.ylabel('2nd eigenvector')
plt.title('SVM on Iris with RBF with Gamma = 2^-5')
plt.show()

# %%
