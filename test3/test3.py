import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('titanic/train.csv')

print(df.isnull().sum())

# 取出稱謂 Mr., Mrs., Miss.：'空格' + 字母 + '.'
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)

def title_map(x):   # title mapping 'Mr' = 0 , 'Miss' = 1 , 'Mrs' = 2 and other = 3
    title = 0
    if x == 'Mr':
        title = 0
    elif x == 'Miss' :
        title = 1
    elif x == 'Mrs':
        title = 2
    else:
        title = 3
    return title
df['Title'] = df['Title'].map(title_map)

# Sex Mapping
sex_mapping = { 'male': 0 , 'female': 1 }
df['Sex'] = df['Sex'].map(sex_mapping)

# 計算個別 Title 的年齡中位數，補值
df['Age'].fillna(df.groupby('Title')['Age'].transform('median'), inplace = True)

# Age Mapping，取十分位
# print(df.Age.max()) # MAX 值為 80
df.loc[ df['Age'] < 10 , 'Age' ] = 0
df.loc[ (df['Age'] >= 10) & (df['Age'] < 20) , 'Age' ] = 1
df.loc[ (df['Age'] >= 20) & (df['Age'] < 30) , 'Age' ] = 2
df.loc[ (df['Age'] >= 30) & (df['Age'] < 40) , 'Age' ] = 3
df.loc[ (df['Age'] >= 40) & (df['Age'] < 50) , 'Age' ] = 4
df.loc[ (df['Age'] >= 50) & (df['Age'] < 60) , 'Age' ] = 5
df.loc[ (df['Age'] >= 60) & (df['Age'] < 70) , 'Age' ] = 6
df.loc[ (df['Age'] >= 70) & (df['Age'] < 80) , 'Age' ] = 7
df.loc[ (df['Age'] >= 80) , 'Age'] = 8

# 'Embarked' 登船地點，補上與下一筆相同值，若最後一筆是na則補上與前一筆相同值
df['Embarked'].fillna(method='bfill', inplace=True)
df['Embarked'].fillna(method='pad', inplace=True)

# Embarked Mapping
embarked_mapping = { 'S': 0 , 'C': 1 , 'Q': 2 }
df['Embarked'] = df['Embarked'].map(embarked_mapping)

# 計算個別 Pclass 的票價中位數，補值
# df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'), inplace = True)

# 取出船艙的第一個字母，表示船艙所在的區域
df['Cabin'] = df['Cabin'].str[:1]

# Cabin Mapping
cabin_mapping = { 'A': 1 , 'B': 2 , 'C': 3 , 'D': 4 , 'E': 5 ,
                  'F': 6 , 'G': 7 , 'T': 8 }
df['Cabin'] = df['Cabin'].map(cabin_mapping)


# print(df['Cabin'].value_counts())

# 計算個別 Pclass 的船艙區域中位數，補值
df['Cabin'].fillna(df.groupby('Pclass')['Cabin'].transform('median').astype(int), inplace = True)
# print(df['Cabin'].value_counts())

# 合併 Sibsp & Parch = FamilyNum
df['FamilyNum'] = df['SibSp'] + df['Parch']

# 刪除 PassengerId, Name, SibSp, Parch, Ticket, Title
col_drop = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Title']
df = df.drop(col_drop, axis = 1)

print('*************************************')
print(df)
print('*************************************')

target = df['Survived']
dataset = df.drop('Survived', axis = 1)

X = dataset.values
Y = target.values

# 平均 & 變異數標準化
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)
X_normalized = normalize(X_scaled)

# 主成分分析，數據轉2維
pca = PCA(n_components = 2) 
X_2 = pca.fit_transform(X_normalized)

# 座標圖
plt.scatter(X_2[:,0], X_2[:,1])
plt.show()

# 建立 SpectralClustering 模型
sc = SpectralClustering(n_clusters=2)
sc.fit(X_2, Y)
labels = sc.labels_
plt.scatter(X_2[:,0], X_2[:,1], c=labels)
plt.show()

from sklearn.metrics import pairwise_distances

# 計算各點之間的距離，距離 < mean 才視為相連，生成A矩陣
A = pairwise_distances(X_normalized, metric='euclidean')
A_mean = A.mean()
print('\nA mean:', A_mean)
vectorizer = np.vectorize(lambda x: 1 if (x > 0) & (x < A_mean) else 0)
A = np.vectorize(vectorizer)(A)
print('A:', A)

# Laplacian Matrix
from scipy.sparse import csgraph
L = csgraph.laplacian(A, normed=False)
print('L: ', L)

# 計算 eigenvalues & eigenvectors
eigval, eigvec = np.linalg.eig(L)
eigval = eigval.astype(float).round(5)
eigvec = eigvec.astype(float).round(5)
eigval = np.sort(eigval)
print('eigval: ', eigval)
# print('eigvec: ', eigvec)

# 取得最小的 nonzero eigenvalue
minval = np.min(eigval[np.nonzero(eigval)])

# 取得 corresponding eigenvector
def near(a, b, rtol = 1e-5, atol = 1e-8):
    return np.abs(a - b) < (atol + rtol * np.abs(b))

nonozero_eigvec = eigvec[near(eigval, minval)].astype(float).round(3)

print('\nThe smallest nonzero eigenvalue and the corresponding eigenvector:')
print('eigenvalue: \n', minval)
print('eigenvector: \n',nonozero_eigvec)

plt.plot(eigval)
plt.title('eigenvalues')
plt.show()
